"""Predeclared equal v4/v5 family ensemble; no weight fitting or uploads."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v2_analog import diagnostics


def main():
    out = Path('experiments/results/v5-equal-ensemble')
    if out.exists():
        raise FileExistsError(out)
    roots = [Path('experiments/results')/name for name in
             ['v5-pair-regularization', 'v5-pair-bagging', 'v5-pair-features']]
    winners = [json.loads((p/'manifest.json').read_text())['winner'] for p in roots]
    sources = [Path(__file__), Path('docs/v5-protocol.md')]
    metrics, frames = [], {}
    for split in (42, 43, 44, 45, 'scaffold'):
        paths = [p/f'oof-{split}.csv' for p in roots]
        ds = [pd.read_csv(p) for p in paths]
        sources.extend(paths)
        first = ds[0]
        for d in ds:
            pd.testing.assert_frame_equal(first[['row','target','fold']], d[['row','target','fold']])
            assert np.array_equal(d.row, np.arange(320))
            base = next(c for c in d if c.startswith('v4-ridge-log'))
            np.testing.assert_allclose(first['v4-ridge-log'], d[base], rtol=1e-12, atol=1e-10)
        base = first['v4-ridge-log'].to_numpy()
        pred = np.mean([base]+[d[w].to_numpy() for d,w in zip(ds,winners)], axis=0)
        assert np.isfinite(pred).all() and (pred>=0).all()
        frames[split] = first[['row','target','fold']].assign(ensemble=pred, baseline=base)
        for name, p in [('ensemble',pred),('baseline',base)]:
            metrics.append({'split': split, 'model': name, **diagnostics(first.target.to_numpy(),p)})
    out.mkdir(parents=True)
    for split, frame in frames.items():
        frame.to_csv(out/f'oof-{split}.csv', index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({'members':['v4-ridge-log']+winners,
        'weights':[.25]*4,'predeclared_protocol':'docs/v5-protocol.md','no_upload':True,
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
