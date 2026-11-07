"""Fixed cross50 model: audited rebuild4 snapshot vs saved v3/v4 controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v3_intervals import read_snapshot
from experiments.v6_pair_external import fit_predict, source_context
from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    source = Path('experiments/results/glm-rebuild4-audit/accepted.csv')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == args.sha256
    out = Path('experiments/results/glm-rebuild4-ablation')
    if out.exists():
        raise FileExistsError(out)
    data = load(Path('data/raw_v2'), require_v2=True)
    ext, excluded = read_snapshot(source, data.train, data.test)
    assert not len(excluded) and ext.kind.eq('exact').all()
    meta = pd.read_csv(source, usecols=['SMILES','cell','strain'])
    ext = ext.merge(meta,on='SMILES',validate='one_to_one').rename(columns={'cell':'Culture'})
    frame = pd.concat([data.train,ext],ignore_index=True)
    n = len(data.y)
    y = np.r_[data.y,ext.lower_si.to_numpy(float)]
    x = fingerprint(frame)
    sim = tanimoto(x,x)
    ctx, normalized = contexts(frame), source_context(frame)
    match, source_match = same_context(ctx,ctx), same_context(normalized,normalized)
    metrics, audits, frames = [],[],{}
    with threadpool_limits(limits=2):
        for split in (52,53,54,55,'scaffold'):
            previous = pd.read_csv(Path('experiments/results/v6-pair-external')/f'oof-{split}.csv')
            p, fold_ids = np.full(n,np.nan),np.full(n,-1)
            iterator = (KFold(5,shuffle=True,random_state=split).split(data.y) if split!='scaffold'
                        else (fold_indices(scaffold_folds(scaffolds_of(data.train),5),f) for f in range(5)))
            for f,(tr,va) in enumerate(iterator):
                p[va],_,audit = fit_predict('cross50',x,y,sim,match,source_match,tr,va,n)
                fold_ids[va] = f
                audits.append({'split':split,'fold':f,'train_rows':tr.tolist(),
                               'validation_rows':va.tolist(),**audit})
            np.testing.assert_array_equal(fold_ids,previous.fold)
            np.testing.assert_allclose(data.y,previous.target)
            np.testing.assert_array_equal(previous.row,np.arange(n))
            predictions = {'rebuild4-cross50':p,'v3-cross50':previous.cross50.to_numpy(),
                           'v4-ridge-log':previous['v4-ridge-log'].to_numpy()}
            for name, values in predictions.items():
                assert np.isfinite(values).all() and (values>=0).all()
                metrics.append({'split':split,'model':name,**diagnostics(data.y,values)})
            frames[split] = pd.DataFrame({'row':np.arange(n),'target':data.y,'fold':fold_ids,**predictions})
    out.mkdir(parents=True)
    for split,d in frames.items():
        d.to_csv(out/f'oof-{split}.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'fold-audit.json').write_text(json.dumps(audits)+'\n')
    sources = [Path(__file__),source,Path('experiments/v6_pair_external.py'),
               Path('data/raw_v2/train_v2.csv')]
    sources += list(Path('experiments/results/v6-pair-external').glob('oof-*.csv'))
    (out/'manifest.json').write_text(json.dumps({'profile':'cross50','no_retuning':True,
        'external_rows':len(ext),'confirmation':[52,53,54,55],'no_upload':True,
        'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
