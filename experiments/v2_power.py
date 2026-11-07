"""Six preregistered power profiles, discovery9 then frozen confirmation10/11.

Selection uses full raw MSE; tail-excluded diagnostics never select a model.
Run from repository root with PYTHONPATH=. python experiments/v2_power.py.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from vacv.dataset import load, submission
from vacv.pipeline import scaffolds_of
from vacv.splits import scaffold_folds, fold_indices
from vacv.power_regression import CONFIGS, PowerRegression

OUT = Path('experiments/results/v2-power')


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load(Path('data/raw_v2'), require_v2=True)
    y = data.y
    manifest = {'configs': CONFIGS, 'discovery': 9, 'confirmation': [10, 11],
                'selection_metric': 'mean raw squared error all training rows',
                'train_rows': len(y), 'target_max': float(y.max()),
                'train_sha256': hashlib.sha256(Path('data/raw_v2/train_v2.csv').read_bytes()).hexdigest(),
                'trees': 800, 'seeds': [0, 1, 2]}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    metrics = []

    def evaluate(label, folds, configs):
        pred = {name: np.full(len(y), np.nan) for name in configs}
        fold_id = np.full(len(y), -1)
        powers = sorted({p for name in configs for p, _ in CONFIGS[name]})
        names = {0.: 'log', .25: 'quarter', .5: 'sqrt', 1.: 'raw'}
        for fold, (tr, va) in enumerate(folds):
            base = {}
            for power in powers:
                model = PowerRegression(names[power]).fit(data.train.iloc[tr], y[tr])
                base[power] = model.predict(data.train.iloc[va])
            for name in configs:
                pred[name][va] = sum(weight*base[p] for p, weight in CONFIGS[name])
            fold_id[va] = fold
            pd.DataFrame({'index': np.arange(len(y)), 'target': y, 'fold': fold_id, **pred}).to_csv(OUT/f'oof-{label}-partial.csv', index=False)
            print(f'{label} fold {fold+1} complete', flush=True)
        for name, p in pred.items():
            error = (p-y)**2
            metrics.append({'split': label, 'model': name, 'mse': float(error.mean()),
                'tail_squared_error': float(error[y.argmax()]),
                'tail_prediction': float(p[y.argmax()]),
                'mse_without_largest_diagnostic_only': float(np.delete(error, y.argmax()).mean()),
                'mae': float(np.abs(p-y).mean())})
        pd.DataFrame({'index': np.arange(len(y)), 'target': y, 'fold': fold_id, **pred}).to_csv(OUT/f'oof-{label}.csv', index=False)
        (OUT/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        print(metrics[-len(configs):], flush=True)

    evaluate('9', KFold(5, shuffle=True, random_state=9).split(y), list(CONFIGS))
    winner = min(metrics, key=lambda row: row['mse'])['model']
    manifest['frozen_winner'] = winner
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    finalists = list(dict.fromkeys(['log', winner]))
    for seed in (10, 11):
        evaluate(str(seed), KFold(5, shuffle=True, random_state=seed).split(y), finalists)
    sf = scaffold_folds(scaffolds_of(data.train), 5, seed=0)
    evaluate('scaffold', [fold_indices(sf, i) for i in range(5)], finalists)
    template = pd.read_csv('data/raw_v2/sample_submit.csv')
    manifest['submissions'] = {}
    for name in finalists:
        model = PowerRegression(name).fit(data.train, y)
        result = submission(data.test['index'], model.predict(data.test))
        result = result.set_index('index').loc[template['index']].reset_index()
        path = OUT/f'submission-{name}.csv'
        result.to_csv(path, index=False)
        manifest['submissions'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

if __name__ == '__main__':
    main()
