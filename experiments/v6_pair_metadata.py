"""Title morphology as pair-delta metadata; no source-label lookup."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from experiments.v2_analog import fingerprint, tanimoto, diagnostics
from experiments.v2_context_analog import contexts, same_context
from experiments.v4_pair_delta import fit_predict
from vacv.dataset import load
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices

PROFILES = ('title1', 'title3')


def representations(frame):
    x = fingerprint(frame)
    # Stateless hashing: vocabulary and normalization never depend on labels.
    title = HashingVectorizer(analyzer='char', ngram_range=(2, 5), n_features=256,
                              alternate_sign=False, norm='l2').transform(
                                  frame.Title.fillna('').astype(str).str.casefold()).toarray()
    return {'v4': x, 'title1': np.hstack([x,title]), 'title3': np.hstack([x,3*title])}


def main():
    out = Path('experiments/results/v6-pair-metadata')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    data = load(Path('data/raw_v2'), require_v2=True)
    reps, y = representations(data.train), data.y
    sim = tanimoto(reps['v4'],reps['v4'])
    ctx = contexts(data.train)
    match = same_context(ctx,ctx)
    metrics, audits, winner = [], [], None
    with threadpool_limits(limits=2):
        for split in (51,52,53,54,55,'scaffold'):
            names = (*PROFILES,'v4') if split==51 else (winner,'v4')
            pred = {n:np.full(len(y),np.nan) for n in names}
            folds = np.full(len(y),-1)
            iterator = (KFold(5,shuffle=True,random_state=split).split(y) if split!='scaffold'
                        else (fold_indices(scaffold_folds(scaffolds_of(data.train),5),f) for f in range(5)))
            for f,(tr,va) in enumerate(iterator):
                folds[va] = f
                for n in names:
                    pred[n][va] = fit_predict('ridge-log',reps[n],y,sim,match,tr,va)[0]
                audits.append({'split':split,'fold':f,'train_rows':tr.tolist(),'validation_rows':va.tolist()})
            for n,p in pred.items():
                metrics.append({'split':split,'model':n,**diagnostics(y,p)})
            pd.DataFrame({'row':np.arange(len(y)),'target':y,'fold':folds,**pred}).to_csv(out/f'oof-{split}.csv',index=False)
            if split==51:
                winner = min((m for m in metrics if m['model'] in PROFILES),key=lambda m:m['mse'])['model']
                (out/'frozen-winner.json').write_text(json.dumps({'winner':winner,'discovery':51})+'\n')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'fold-audit.json').write_text(json.dumps(audits)+'\n')
    (out/'manifest.json').write_text(json.dumps({'winner':winner,'confirmation':[52,53,54,55],
        'no_external_labels':True,'no_upload':True,'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),Path('experiments/v4_pair_delta.py'),Path('data/raw_v2/train_v2.csv')]}},indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
