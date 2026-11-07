"""No-tuning identity-group stress test of the already submitted v4 model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context
from experiments.v4_pair_delta import fit_predict
from vacv.curate import standardize
from vacv.dataset import load


def main():
    out = Path('experiments/results/v5-identity-stress')
    if out.exists():
        raise FileExistsError(out)
    data = load(Path('data/raw_v2'), require_v2=True)
    keys, rejects = [], []
    for i, s in enumerate(data.train.SMILES):
        value = standardize(s)
        keys.append(value.inchikey[:14] if value.inchikey else 'raw:'+s)
        if not value.inchikey:
            rejects.append({'row': i, 'reason': value.reject})
    keys = np.asarray(keys)
    y, x = data.y, fingerprint(data.train)
    ctx = contexts(data.train)
    sim, match = tanimoto(x, x), same_context(ctx, ctx)
    pred, folds, audits = np.full(len(y), np.nan), np.full(len(y), -1), []
    with threadpool_limits(limits=2):
        for fold, (tr, va) in enumerate(GroupKFold(5).split(x, y, keys)):
            assert not set(keys[tr]).intersection(keys[va])
            pred[va] = fit_predict('ridge-log', x, y, sim, match, tr, va)[0]
            folds[va] = fold
            audits.append({'fold': fold, 'train_rows': tr.tolist(), 'validation_rows': va.tolist()})
    out.mkdir(parents=True)
    pd.DataFrame({'row': np.arange(len(y)), 'identity': keys, 'target': y,
                  'fold': folds, 'ridge-log': pred}).to_csv(out/'oof.csv', index=False)
    sizes = pd.Series(keys).value_counts()
    metrics = diagnostics(y, pred)
    metrics.update({'unique_identities': len(sizes), 'repeated_identity_groups': int((sizes>1).sum()),
                    'rows_in_repeated_identities': int(sizes[sizes>1].sum()),
                    'standardization_rejections': rejects, 'no_tuning': True,
                    'all_320_labels_retained': len(y)==320,
                    'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__), Path('experiments/v4_pair_delta.py'),
                                Path('data/raw_v2/train_v2.csv')]}})
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (out/'fold-audit.json').write_text(json.dumps(audits)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
