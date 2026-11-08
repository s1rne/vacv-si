# Rebuild4 data-only ablation

Fixed cross50 pair-transfer model, unchanged hyperparameters and folds52–55
plus scaffold. The newly audited43-row snapshot replaces the previous47-row
snapshot. This is a data-source sensitivity check, not another model search.
All320 competition targets remain unchanged, including the maximum.

| Training variant | Mean fullMSE52–55 | Scaffold MSE |
|---|---:|---:|
| New43 external rows, cross50 | 3,690,145.067 | 4,255,114.770 |
| Previous47 external rows, cross50 | 3,690,391.235 | 4,255,568.093 |
| Submitted v4, no external labels | **3,595,103.060** | **4,253,853.162** |

The export change reduces the external model's mean MSE by246.168 (0.0067%),
but remains2.64% worse than v4 and loses to v4 on all four confirmation splits.
Do not promote or submit. No candidate CSV or Kaggle upload was created.

The43 structures already exist in the earlier47-row snapshot;42 SI values
agree. NIOCH-14 uses another valid same-document measurement, not a correction
to the old one. The omitted four previous pairs also reproduce from raw
records; inherited legacy variance flags do not prove those pairs invalid.
Thus this change is not additional chemical coverage or proof of cleaner data.

Full protocol equivalence is not established; endpoint exposure is known for
17 of43 exact records. The independent audit, accepted snapshot and provenance
are under `experiments/results/glm-rebuild4-audit/`.

Reproduction:

```sh
PYTHONPATH=. python experiments/glm_rebuild4_ablation.py --sha256 dddba1a7dc121b7d5baf6e4b5a4dc3eea8369dd85557e1e5478308fdc0e48a01
```

The output directory must not already exist. Source hashes and complete OOF
are saved. Verification: three external-model tests passed, all15 reported
model/split metrics independently recomputed from320-row OOF; fold IDs and
targets match the previous controls exactly.
