# Pair-delta query anchor gate

Frozen seed46 winner: gate60.

| Split | Gate | Full raw MSE | Expert rows |
|---|---|---:|---:|
| 46 | gate55 | 4115702.500 | 201 |
| 46 | gate60 | 4115033.996 | 182 |
| 46 | gate65 | 4115125.632 | 138 |
| 46 | v4-gate70 | 4021684.009 | 106 |
| 47 | gate60 | 3630449.626 | 187 |
| 47 | v4-gate70 | 3406357.889 | 122 |
| 48 | gate60 | 3619765.316 | 181 |
| 48 | v4-gate70 | 3239105.619 | 115 |
| 49 | gate60 | 3335998.356 | 178 |
| 49 | v4-gate70 | 3057714.173 | 119 |
| 50 | gate60 | 3530348.944 | 184 |
| 50 | v4-gate70 | 3306347.271 | 122 |
| scaffold | gate60 | 4206799.479 | 92 |
| scaffold | v4-gate70 | 4253853.162 | 35 |

Confirmation mean: 3529140.561; paired v4 mean: 3252381.238; relative change: 8.509%.
Improves every confirmation: False. Each confirmation <=3M: False.

Reject the relaxed global gate: all four confirmation splits worsen. Although coverage increases from115–122 to178–187 rows, additional weaker anchors dilute useful extrapolation: maximum-target predictions decline from4361/5654/5900/4520 to3258/4264/4470/3432 across47–50. The scaffold improvement does not rescue the failed confirmations. Two tests passed (exact equality with v4 at .70; heldout-label independence and threshold firing behavior). All30 fold audits exclude validation rows from training pair endpoints.

Only query anchor similarity threshold changes: fixed .55/.60/.65 versus v4 .70. Training pairs, Morgan radius2 binary2048, Ridge alpha10 no intercept, log1p deltas, top3 anchors, similarity^8 weights, structural-log fallback unchanged. All320 labels retained without clipping/removal, full raw MSE. Seed45 was hypothesis motivation only and is not evaluated or used in selection. Discovery46 chooses among exactly three profiles; frozen winner confirmed on47–50 with scaffold stress. No external data, forbidden fields, CSV or upload. Paired baseline is v4 Ridge-log, not the weaker context-only model.
