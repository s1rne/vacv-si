# Frozen CheMeleon regression on official train_v2

The official refreshed target changes 81 old sentinel labels from 10000 to 0.
This experiment fits regression directly to that target. No old sentinel model,
SI substitution, manual literature mapping, forbidden competition concentration
column, or test target is used. All 320 labels, including 36066.667, remain in
training and raw-MSE evaluation.

Frozen cached CheMeleon embeddings are reduced to 32 principal components using
only each training fold. Permitted context is encoded using training categories.
Four fixed heads were evaluated: ExtraTrees raw, ExtraTrees log1p, Ridge log1p,
and ExtraTrees log1p augmented with the permitted external presence counts and
similarities. ExtraTrees averages two seeds, 400 trees each, leaf size 2 and
max_features .5. Ridge uses alpha 10 and training-only scaling. There was no
hyperparameter sweep. The presence features contain counts/similarities, not SI
or concentration values; they come from the existing collected-data artifact.

| Head | Discovery split 6 | Confirmation mean 7–8 | Scaffold |
|---|---:|---:|---:|
| ExtraTrees raw | 4,541,849 | 4,399,599 | 4,549,986 |
| ExtraTrees log | 4,245,596 | 4,283,416 | 4,261,772 |
| ExtraTrees log + external presence | **4,213,149** | **4,268,475** | 4,260,691 |
| Ridge log | 4,253,475 | 4,276,112 | **4,241,210** |

The discovery winner is ExtraTrees log + presence. Adding collected-data features
improves both confirmation seeds compared with the paired no-presence head;
mean gain is 0.35%, scaffold gain 0.025%. These are small differences on the
same 320 rows, not a guarantee of leaderboard superiority. Raw MSE is dominated
by the single extreme label; no label was removed to improve the report.
Log predictions estimate a transformed-target mean and may underpredict the
raw conditional mean; the comparison includes raw regression for that reason.

All four full-training prediction files have 597 finite nonnegative values in
the exact sample-submit order. `submissions.json` records data, embedding,
presence artifact, code and CSV hashes; `verification.json` records checks.
Selected candidate: `submission-chemeleon-et-log-presence.csv`. No Kaggle upload
was performed by this experiment.

Reproduction (from repository root with prepared cached embeddings):

```sh
PYTHONPATH=. python experiments/v2_embeddings.py --data-dir data/raw_v2 --out outputs/v2-embeddings
```

On this Mac the existing environment requires DYLD_LIBRARY_PATH pointing to
its sklearn/.dylibs. The run used the shared existing environment without
installing packages or modifying system libraries.
