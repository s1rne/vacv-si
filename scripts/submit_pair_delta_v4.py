"""Build the explicitly authorized frozen Ridge-log submission; never upload."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.v2_analog import fingerprint, tanimoto
from experiments.v2_context_analog import contexts, same_context
from experiments.v4_pair_delta import fit_predict
from vacv.dataset import load, submission


def main():
    out = Path('experiments/results/v4-pair-delta-submission')
    if out.exists():
        raise FileExistsError(out)
    data = load(Path('data/raw_v2'), require_v2=True)
    frozen = Path('experiments/results/v4-pair-delta/frozen-winner.json')
    assert json.loads(frozen.read_text())['winner'] == 'ridge-log'
    frames = pd.concat([data.train, data.test], ignore_index=True)
    x = fingerprint(frames)
    ctx = contexts(frames)
    tr = np.arange(len(data.train))
    query = np.arange(len(data.train), len(frames))
    with threadpool_limits(limits=2):
        p, fired, pairs = fit_predict('ridge-log', x, data.y, tanimoto(x, x),
                                     same_context(ctx, ctx), tr, query)
    result = submission(data.test['index'].to_numpy(), p)
    template = pd.read_csv('data/raw_v2/sample_submit.csv')
    assert list(result.columns) == list(template.columns)
    assert np.array_equal(result['index'], template['index'])
    assert len(result) == len(data.test) == 597
    out.mkdir(parents=True)
    path = out / 'submission-pair-delta-v4.csv'
    result.to_csv(path, index=False)
    reread = pd.read_csv(path)
    assert np.array_equal(reread['index'], result['index'])
    np.testing.assert_allclose(reread['SI VACV'], p, rtol=1e-12, atol=1e-12)
    sources = [Path(__file__), Path('experiments/v4_pair_delta.py'),
               Path('experiments/v2_analog.py'), Path('experiments/v2_context_analog.py'),
               Path('vacv/dataset.py'), Path('vacv/pipeline.py'), frozen,
               Path('data/raw_v2/train_v2.csv'), Path('data/raw_v2/test.csv'), path]
    manifest = {'profile': 'ridge-log', 'training_rows': len(tr), 'test_rows': len(query),
                'directed_training_pairs': pairs, 'pair_expert_rows': int(fired.sum()),
                'prediction_quantiles': np.quantile(p, [0, .5, .9, .99, 1]).tolist(),
                'external_data': False, 'test_labels_used': False,
                'upload_gate_passed': False,
                'authorization': 'User explicitly approved this candidate after CV checkpoint',
                'sha256': {str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
