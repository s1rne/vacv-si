"""Frozen, train-only chirality and scalar calibration ablation of v4."""
import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from threadpoolctl import threadpool_limits

from experiments.v2_analog import tanimoto
from experiments.v2_context_analog import contexts, same_context
from experiments.v4_pair_delta import fit_predict
from vacv.curate import standardize
from vacv.pipeline import scaffolds_of

PROTOCOL = Path('docs/v10-spec-protocol.json')
REFERENCE = Path('docs/v8-nested-protocol.json')
OUTPUT = Path('experiments/results/v10-spec-validation')
TRAIN = Path('data/raw_v2/train_v2.csv')
SCHEMES = ('identity', 'identity71', 'identity72', 'scaffold')
VARIANTS = ('raw', 'alpha', 'cap5000', 'alpha_then_cap5000')
CONFIG = {
    'representations': ['original', 'chiral'], 'radius': 2, 'fp_size': 2048,
    'fingerprints': 'RDKit GetCountFingerprintAsNumPy > 0, float64 binary',
    'chirality': {'original': False, 'chiral': True}, 'predictor': 'v4 ridge-log unchanged',
    'similarity': 'own representation Tanimoto', 'context': 'original strain/Culture match',
    'schemes': list(SCHEMES), 'folds': 'exact v8 outer and inner partitions',
    'variants': list(VARIANTS), 'cap': 5000.,
    'alpha': 'max(0, sum(y_train * inner_oof) / sum(inner_oof ** 2)); zero denominator => 1',
    'calibration': 'inner Group3 OOF on outer training rows only; refit predictor on outer train',
    'primary_metric': 'pooled outer OOF raw MSE, all 320 unchanged targets',
    'diagnostic_only': 'MSE excluding one argmax target row',
    'selection': 'none; all eight fixed variants reported on every scheme',
    'robust_recommendation': 'each fixed variant improves all three identity schemes vs original raw; scaffold does not worsen',
    'identity_mean': 'diagnostic only; not a selection criterion',
    'external_data': False, 'test_inference': False, 'submission': False,
    'limitations': 'historically reused dataset; repeated splits are not independent holdouts',
}
SOURCES = tuple(map(Path, (
    'experiments/v10_spec_validation.py', 'tests/test_v10_spec_validation.py',
    'experiments/v2_analog.py', 'experiments/v2_context_analog.py',
    'experiments/v4_pair_delta.py', 'vacv/curate.py', 'vacv/pipeline.py',
    'vacv/features.py', 'vacv/units.py', 'vacv/literature.py', 'vacv/dataset.py',
    'vacv/splits.py',
)))
BASELINES = tuple(Path(f'experiments/results/v8-nested-validation/oof-{s}.csv') for s in SCHEMES)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {p: importlib.metadata.version(p) for p in
            ('numpy', 'pandas', 'scikit-learn', 'rdkit', 'scipy', 'threadpoolctl', 'lightgbm')}


def read_train():
    frame = pd.read_csv(TRAIN, usecols=['SMILES', 'strain', 'Culture', 'SI VACV'])
    y = frame['SI VACV'].to_numpy(dtype=float)
    if len(y) != 320 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Expected all 320 finite nonnegative original targets')
    if frame[['SMILES', 'strain', 'Culture']].isna().any().any():
        raise ValueError('Missing allowed input fields')
    return frame, y


def fingerprints(smiles, chirality=False):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=chirality)
    rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        rows.append(np.zeros(2048) if mol is None else
                    (generator.GetCountFingerprintAsNumPy(mol) > 0).astype(float))
    return np.stack(rows)


def checked_reference(frame):
    ref = json.loads(REFERENCE.read_text())
    identities = []
    for smiles in frame.SMILES:
        key = standardize(smiles).inchikey
        if not key:
            raise ValueError('Cannot reproduce identity grouping')
        identities.append(key[:14])
    if identities != ref['identity_groups']:
        raise ValueError('Identity grouping drift')
    groups = np.asarray(ref['scaffold_groups'])
    # Check that frozen components preserve both relations; no new grouping is chosen.
    seen = {}
    for i, (identity, scaffold) in enumerate(zip(identities, scaffolds_of(frame), strict=True)):
        for key in [('identity', identity)] + ([('scaffold', scaffold)] if scaffold else []):
            if key in seen and groups[i] != groups[seen[key]]:
                raise ValueError('Frozen scaffold group breaks a shared relation')
            seen[key] = i
    identities = np.asarray(identities)
    for scheme in SCHEMES:
        coverage = np.zeros(len(frame), int)
        for outer in ref['folds'][scheme]:
            tr, va = np.asarray(outer['train']), np.asarray(outer['valid'])
            if set(tr) & set(va) or set(tr) | set(va) != set(range(len(frame))):
                raise ValueError('Invalid outer partition')
            coverage[va] += 1
            inner_coverage = np.zeros(len(frame), int)
            for split in [outer, *outer['inner']]:
                a, b = np.asarray(split['train']), np.asarray(split['valid'])
                if set(identities[a]) & set(identities[b]):
                    raise ValueError('Identity leakage')
                if scheme == 'scaffold' and set(groups[a]) & set(groups[b]):
                    raise ValueError('Scaffold leakage')
            for split in outer['inner']:
                a, b = split['train'], split['valid']
                if set(a) & set(b) or set(a) | set(b) != set(tr):
                    raise ValueError('Invalid inner partition')
                inner_coverage[b] += 1
            if not np.all(inner_coverage[tr] == 1) or inner_coverage[va].any():
                raise ValueError('Invalid inner coverage')
        if not np.all(coverage == 1):
            raise ValueError('Invalid outer coverage')
    return ref


def calibrate(y_train, inner_oof):
    if not np.isfinite(y_train).all() or not np.isfinite(inner_oof).all():
        raise ValueError('Calibration requires finite inner OOF and training labels')
    denominator = float(inner_oof @ inner_oof)
    return max(0., float(y_train @ inner_oof) / denominator) if denominator else 1.


def variants(predictions, alpha):
    return np.column_stack([predictions, alpha * predictions,
                            np.minimum(predictions, 5000.), np.minimum(alpha * predictions, 5000.)])


def evaluate_fold(y, tr, va, inner, predict):
    """Only explicit fitting rows are exposed to the predictor or calibrator."""
    inner_oof = np.full(len(y), np.nan)
    inner_fold = np.full(len(y), -1)
    for fold, split in enumerate(inner):
        a, b = np.asarray(split['train']), np.asarray(split['valid'])
        labels = np.full(len(y), np.nan)
        labels[a] = y[a]
        inner_oof[b] = predict(a, b, labels)
        inner_fold[b] = fold
    alpha = calibrate(y[tr], inner_oof[tr])
    labels = np.full(len(y), np.nan)
    labels[tr] = y[tr]
    raw = predict(tr, va, labels)
    return variants(raw, alpha), alpha, inner_oof[tr], inner_fold[tr]


def freeze():
    frame, _ = read_train()
    checked_reference(frame)
    protocol = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'config': CONFIG,
                'versions': versions(), 'fold_reference': str(REFERENCE),
                'fold_reference_sha256': sha(REFERENCE),
                'sha256': {str(p): sha(p) for p in (TRAIN, REFERENCE, *SOURCES, *BASELINES)}}
    with PROTOCOL.open('x') as handle:
        handle.write(json.dumps(protocol, indent=2) + '\n')
    print('Protocol frozen:', sha(PROTOCOL), flush=True)


def run(expected_sha):
    if sha(PROTOCOL) != expected_sha:
        raise ValueError('Protocol hash mismatch')
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['config'] != CONFIG or protocol['versions'] != versions():
        raise ValueError('Configuration or dependency version drift')
    for path, checksum in protocol['sha256'].items():
        if sha(path) != checksum:
            raise ValueError(f'Frozen source/input changed: {path}')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frame, y = read_train()
    reference = checked_reference(frame)
    ctx = contexts(frame)
    matching = same_context(ctx, ctx)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'protocol.json').write_bytes(PROTOCOL.read_bytes())
    metrics, calibrations, parity = [], [], []
    for representation in CONFIG['representations']:
        x = fingerprints(frame.SMILES, CONFIG['chirality'][representation])
        similarity = tanimoto(x, x)

        def predict(tr, va, labels):
            return fit_predict('ridge-log', x, labels, similarity, matching, tr, va)[0]

        for scheme in SCHEMES:
            oof = np.full((len(y), len(VARIANTS)), np.nan)
            assignment = np.full(len(y), -1)
            for fold, split in enumerate(reference['folds'][scheme]):
                tr, va = np.asarray(split['train']), np.asarray(split['valid'])
                result, alpha, inner_oof, inner_assignment = evaluate_fold(y, tr, va, split['inner'], predict)
                oof[va], assignment[va] = result, fold
                calibrations.append({'representation': representation, 'scheme': scheme, 'fold': fold,
                                     'alpha': alpha, 'train_rows': len(tr),
                                     'numerator': float(y[tr] @ inner_oof),
                                     'denominator': float(inner_oof @ inner_oof)})
                pd.DataFrame({'row': tr, 'target': y[tr], 'inner_fold': inner_assignment,
                              'raw': inner_oof}).to_csv(OUTPUT / f'inner-{representation}-{scheme}-{fold}.csv', index=False)
                print(representation, scheme, fold, 'alpha', alpha, flush=True)
            if not np.isfinite(oof).all() or (assignment < 0).any():
                raise ValueError('Incomplete OOF')
            pd.DataFrame({'row': np.arange(len(y)), 'target': y, 'fold': assignment,
                          **dict(zip(VARIANTS, oof.T))}).to_csv(OUTPUT / f'oof-{representation}-{scheme}.csv', index=False)
            if representation == 'original':
                baseline = pd.read_csv(Path(f'experiments/results/v8-nested-validation/oof-{scheme}.csv'))
                np.testing.assert_array_equal(baseline.row, np.arange(len(y)))
                np.testing.assert_array_equal(baseline.target, y)
                np.testing.assert_array_equal(baseline.fold, assignment)
                np.testing.assert_allclose(baseline.v4, oof[:, 0], rtol=1e-12, atol=1e-8)
                parity.append({'scheme': scheme, 'passed': True,
                               'max_abs_prediction_difference': float(np.max(np.abs(baseline.v4 - oof[:, 0])))})
            for name, p in zip(VARIANTS, oof.T):
                squared = (p - y) ** 2
                metrics.append({'representation': representation, 'scheme': scheme, 'variant': name,
                                'mse': float(squared.mean()),
                                'mse_without_max_diagnostic_only': float(np.delete(squared, np.argmax(y)).mean())})
    for name, value in [('metrics', metrics), ('calibration-audit', calibrations), ('baseline-parity', parity)]:
        (OUTPUT / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
    manifest = {'protocol_sha256': expected_sha, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                'rows': len(y), 'all_original_targets_retained': True, 'target_clipping': False,
                'test_inference': False, 'submission_created': False, 'uploaded': False,
                'baseline_parity_all_schemes': all(item['passed'] for item in parity),
                'artifacts_sha256': {p.name: sha(p) for p in sorted(OUTPUT.iterdir())}}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--freeze', action='store_true')
    choice.add_argument('--protocol-sha256')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        freeze() if args.freeze else run(args.protocol_sha256)
