"""Bounded paired external-SI augmentation experiment for official train_v2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from vacv.curate import standardize
from vacv.dataset import load
from vacv.pipeline import scaffolds_of, structure_matrix
from vacv.splits import fold_indices, scaffold_folds


SOURCE = Path('/Users/mike/Projects/vacv-si/data/processed/external_v2/measured.csv')
OUT = Path('experiments/results/luna-v2-external-models')
WEIGHTS = (0.10, 0.25)
SEEDS = (0, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load(Path('data/raw_v2'), require_v2=True)
    source_hash = sha256(SOURCE)
    # Only target, structure and declared eligibility/provenance fields are read.
    allowed = ['smiles', 'si', 'measurement_type', 'pairing_level',
               'provenance_tier', 'notes']
    raw = pd.read_csv(SOURCE, usecols=allowed)
    raw.to_csv(OUT / 'external-source-eligible-fields.csv', index=False)

    comp_smiles = set(data.train.SMILES.astype(str)) | set(data.test.SMILES.astype(str))
    comp_keys: set[str] = set()
    invalid_comp_exact = set()
    for smiles in comp_smiles:
        item = standardize(smiles)
        if item.inchikey:
            comp_keys.add(item.inchikey[:14])
        else:
            invalid_comp_exact.add(smiles)

    # Preserve an auditable reason for every source row without retaining disallowed fields.
    audit = raw.copy()
    audit['source_row'] = np.arange(len(audit))
    audit['exact_smiles'] = audit.smiles.astype(str)
    standardized = [standardize(s) for s in audit.smiles]
    audit['inchikey14'] = [x.inchikey[:14] if x.inchikey else '' for x in standardized]
    audit['standardization_reject'] = [x.reject or '' for x in standardized]
    audit['exclusion_reason'] = ''
    audit.loc[audit.standardization_reject.ne(''), 'exclusion_reason'] = 'invalid_external_structure'
    audit.loc[audit.exclusion_reason.eq('') & ~audit.measurement_type.eq('exact'), 'exclusion_reason'] = 'not_exact'
    audit.loc[audit.exclusion_reason.eq('') & ~audit.provenance_tier.isin(['A', 'E']), 'exclusion_reason'] = 'provenance_not_A_or_E'
    audit.loc[audit.exclusion_reason.eq('') & ~audit.pairing_level.isin(['strict', 'document']), 'exclusion_reason'] = 'pairing_not_strict_or_document'
    audit.loc[audit.exclusion_reason.eq('') & audit.notes.fillna('').astype(str).str.strip().ne(''), 'exclusion_reason'] = 'notes_not_empty'
    audit.loc[audit.exclusion_reason.eq('') & ~np.isfinite(pd.to_numeric(audit.si, errors='coerce')), 'exclusion_reason'] = 'nonfinite_si'
    audit.loc[audit.exclusion_reason.eq('') & pd.to_numeric(audit.si, errors='coerce').lt(0), 'exclusion_reason'] = 'negative_si'
    overlaps = audit.inchikey14.isin(comp_keys) | audit.exact_smiles.isin(comp_smiles | invalid_comp_exact)
    audit.loc[audit.exclusion_reason.eq('') & overlaps, 'exclusion_reason'] = 'competition_structure_overlap'
    audit.to_csv(OUT / 'filter-audit.csv', index=False)

    ext = audit[audit.exclusion_reason.eq('')].copy()
    # Deduplicate connectivity keys, deterministically retaining the first source row.
    ext = ext.drop_duplicates('inchikey14', keep='first').reset_index(drop=True)
    if ext.empty:
        raise ValueError('No eligible disjoint external SI rows')
    ext[['source_row', 'smiles', 'si', 'inchikey14']].to_csv(OUT / 'external-training-rows.csv', index=False)

    train = data.train.reset_index(drop=True)
    y = data.y
    x = structure_matrix(train)
    ext_frame = pd.DataFrame({'SMILES': ext.smiles.astype(str).to_numpy()})
    xe = structure_matrix(ext_frame)
    ye = np.log1p(pd.to_numeric(ext.si).to_numpy(dtype=float))
    names = ['baseline'] + [f'external_weight_{w:.2f}' for w in WEIGHTS]
    metrics: list[dict] = []
    manifest = {
        'objective': 'paired measured external-SI augmentation, exploratory only',
        'external_source': str(SOURCE), 'source_sha256': source_hash,
        'source_rows': len(raw), 'eligible_unique_rows': len(ext),
        'competition_train_rows': len(data.train), 'competition_test_rows': len(data.test),
        'competition_unique_smiles': len(comp_smiles), 'competition_inchikey14_count': len(comp_keys),
        'invalid_competition_structures_exact_fallback_count': len(invalid_comp_exact),
        'excluded_overlap_rows': int((audit.exclusion_reason == 'competition_structure_overlap').sum()),
        'post_filter_overlap_keys': len(set(ext.inchikey14) & comp_keys),
        'post_filter_exact_smiles_overlap': len(set(ext.smiles) & comp_smiles),
        'eligibility': {'measurement_type': 'exact', 'provenance_tier': ['A', 'E'],
                        'pairing_level': ['strict', 'document'], 'notes': 'empty',
                        'unit_verification': 'not independently proven; experimental only'},
        'external_target': 'SI only', 'competition_forbidden_columns_read': False,
        'estimator': 'ExtraTreesRegressor on Morgan+MACCS, log1p(SI), max_features=sqrt',
        'trees': 400, 'min_samples_leaf': 2, 'seeds': list(SEEDS),
        'external_sample_weights': list(WEIGHTS), 'selection': 'discovery split 18 only',
        'confirmations': [19, 20], 'scaffold': '5-fold scaffold split seed 0',
        'all_official_targets_retained': True, 'official_train_target_rows': len(y),
        'official_max_target': float(y.max()), 'kaggle_upload': 'prohibited/not attempted',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def folds_for(label: str):
        if label == 'scaffold':
            split = scaffold_folds(scaffolds_of(train), 5, seed=0)
            return [fold_indices(split, fold) for fold in range(5)]
        return list(KFold(5, shuffle=True, random_state=int(label)).split(y))

    def evaluate(label: str, eval_names: list[str]) -> None:
        predictions = {n: np.full(len(y), np.nan) for n in eval_names}
        fold_ids = np.full(len(y), -1, dtype=int)
        for fold, (tr, va) in enumerate(folds_for(label)):
            seed_preds = {n: [] for n in eval_names}
            for seed in SEEDS:
                base = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                           max_features='sqrt', random_state=seed, n_jobs=2)
                base.fit(x[tr], np.log1p(y[tr]))
                seed_preds['baseline'].append(base.predict(x[va]))
                for weight in WEIGHTS:
                    augmented_name = f'external_weight_{weight:.2f}'
                    if augmented_name not in eval_names:
                        continue
                    model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                                max_features='sqrt', random_state=seed, n_jobs=2)
                    xx = np.vstack([x[tr], xe])
                    yy = np.r_[np.log1p(y[tr]), ye]
                    sw = np.r_[np.ones(len(tr)), np.full(len(ye), weight)]
                    model.fit(xx, yy, sample_weight=sw)
                    seed_preds[f'external_weight_{weight:.2f}'].append(model.predict(x[va]))
            for name in eval_names:
                log_pred = np.mean(seed_preds[name], axis=0)
                predictions[name][va] = np.maximum(0.0, np.expm1(log_pred))
            fold_ids[va] = fold
            pd.DataFrame({'index': np.arange(len(y)), 'target': y, 'fold': fold_ids,
                          **predictions}).to_csv(OUT / f'oof-{label}-partial.csv', index=False)
            print(f'{label}: fold {fold + 1}/5 complete', flush=True)
        for name, pred in predictions.items():
            if not np.isfinite(pred).all():
                raise ValueError(f'incomplete OOF for {label}/{name}')
            err = (pred - y) ** 2
            metrics.append({'split': label, 'model': name, 'mse': float(err.mean()),
                            'tail_prediction': float(pred[y.argmax()]),
                            'tail_squared_error': float(err[y.argmax()]),
                            'mae': float(np.abs(pred - y).mean())})
        pd.DataFrame({'index': np.arange(len(y)), 'target': y, 'fold': fold_ids,
                      **predictions}).to_csv(OUT / f'oof-{label}.csv', index=False)
        (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(metrics[-len(eval_names):], flush=True)

    with threadpool_limits(limits=2):
        evaluate('18', names)
        winner = min((row for row in metrics if row['split'] == '18'), key=lambda r: r['mse'])['model']
        finalists = ['baseline', winner]
        manifest['frozen_discovery_winner'] = winner
        manifest['frozen_finalists'] = finalists
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # Confirmations and scaffold use only discovery-frozen finalist(s).
        for label in ('19', '20', 'scaffold'):
            # Keep artifacts complete and comparable for both baseline and winner.
            evaluate(label, finalists)
    manifest['completed_splits'] = ['18', '19', '20', 'scaffold']
    manifest['confirmation_gate_each_mse_le_3000000'] = {
        label: all(row['mse'] <= 3_000_000 for row in metrics if row['split'] == label)
        for label in ('19', '20')
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (OUT / 'report.md').write_text(render_report(metrics, manifest))


def render_report(metrics: list[dict], manifest: dict) -> str:
    lookup = {(r['split'], r['model']): r for r in metrics}
    lines = ['# Luna v2 external SI model combination', '',
             'Bounded exploratory paired CV; all official training rows and targets are retained.',
             'The external GLM export has no independent unit-verification proof, so this is experimental evidence only.',
             'No Kaggle upload was attempted. External information is used only for SI regression targets and eligibility filtering.', '',
             '| Split | Baseline MSE | Frozen augmented model MSE | Paired change |',
             '|---|---:|---:|---:|']
    for split in ('18', '19', '20', 'scaffold'):
        base = lookup[(split, 'baseline')]['mse']
        model = manifest.get('frozen_discovery_winner', 'baseline')
        aug = lookup[(split, model)]['mse']
        lines.append(f'| {split} | {base:,.0f} | {aug:,.0f} ({model}) | {(aug/base-1)*100:+.2f}% |')
    lines += ['', f"Eligible unique external rows: {manifest['eligible_unique_rows']}.",
              f"Source SHA256: `{manifest['source_sha256']}`.",
              f"Maximum official training target retained: {manifest['official_max_target']:,.3f}.",
              'The split scores are repeated CV on the same small competition training set, not independent external validation.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
