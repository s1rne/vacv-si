# External exact-SI pair transfer

Frozen seed51 winner: cross50.

Reject: confirmation mean3,690,391.235 versus pairedv4 3,595,103.060, worse2.65%; all four confirmations and scaffold worsen. No candidate or upload.

Coverage: 47 exact external rows,21 known strain/cell,19 used by at least one training pair. Every fold has64 directed external-external pairs; cross profiles add0–2 directed cross pairs. Seven external rows have SI>=1000;10 directed pairs include such rows. Their weighted squared log-delta contribution is4.220 at weight.1 and21.098 at weight.5, logged per fold rather than treating their large rawSI as ordinary bounded values.

Normalization rejects73 competition records containing a recognized strain plus mixed-pathogen/other-strain markers from external/cross pair matching; it changes neither competition-competition training pairs nor query anchors. No external exact rows have these mixed markers. Multiple recognized strain identities also become unknown. Unit tests cover this explicitly.

Provenance limitation: all47 exact rows state "same paper and known cell; full assay protocol equivalence not independently verified". Only17 have exposure_verified=true;30 are false despite nonmissing recorded time fields. Cell/strain equality is therefore not proof of equivalent assay conditions, and26/47 rows lack complete known normalized context. These uncertainties are retained, not imputed from labels. Three tests pass: snapshot hash and train/test identity overlap rejection; heldout-label independence and competition-only anchors; conservative normalization. All fold endpoints are restricted to outertrain plus audited disjoint external rows.

| Split | Profile | Full raw MSE |
|---|---|---:|
| 51 | external10 | 4082727.871 |
| 51 | cross10 | 4082608.706 |
| 51 | cross50 | 4064133.119 |
| 51 | v4-ridge-log | 4089231.368 |
| 52 | cross50 | 3915306.243 |
| 52 | v4-ridge-log | 3886034.771 |
| 53 | cross50 | 3612432.854 |
| 53 | v4-ridge-log | 3560667.962 |
| 54 | cross50 | 3036977.500 |
| 54 | v4-ridge-log | 2746470.846 |
| 55 | cross50 | 4196848.342 |
| 55 | v4-ridge-log | 4187238.661 |
| scaffold | cross50 | 4255568.093 |
| scaffold | v4-ridge-log | 4253853.162 |

{
  "winner": "cross50",
  "profiles": {
    "external10": [
      false,
      0.1
    ],
    "cross10": [
      true,
      0.1
    ],
    "cross50": [
      true,
      0.5
    ]
  },
  "discovery": 51,
  "confirmation": [
    52,
    53,
    54,
    55
  ],
  "confirmation_mean_mse": 3690391.234641375,
  "paired_v4_mean_mse": 3595103.0599298924,
  "improves_all": false,
  "all_confirmations_below3m": false,
  "external_exact_rows": 47,
  "external_known_context_rows": 21,
  "external_max_si": 157026.3157894737,
  "external_si_over1000": 7,
  "excluded_snapshot_rows": 0,
  "bounds_used": false,
  "query_external_anchors": false,
  "competition_rows": 320,
  "max_competition_si": 36066.667,
  "no_candidate_or_upload": true,
  "snapshot_sha256": "9ead9fc63139b88352584a39673f3458beaf1abb8c3ddd776c0de7d1729a2e23",
  "hashes": {
    "/Users/mike/Projects/vacv-si-recovery/experiments/v6_pair_external.py": "57dda46f9392f9704779d65a2e6ea528934f188b19b85e2bbef4a004d3492560",
    "experiments/v3_intervals.py": "56985282067f2d7f49e7a080306018060e165f23510fa9cb82b4a3c093a619cc",
    "experiments/v4_pair_delta.py": "33662660dfcb6e9a979129bc33c8ea3d36034ac9dea15a378a6ed0166326db63",
    "experiments/v2_analog.py": "4b8efa987e914341bd79ba1f736519fbc510aeb92f02ec268a0ea534cbbea976",
    "experiments/v2_context_analog.py": "74325bfbe9ddc2db28aa0c68646bf99777b8a39dfe75343a969deab37317ab77",
    "data/raw_v2/train_v2.csv": "ced712103d0f6687e790c327eedd93d2238da87c5a44f3e6210df75ef0a78b42"
  }
}

Immutable snapshot hash checked before reading context; read_snapshot independently blocks all competition train/test key14 and exact SMILES plus duplicate external identities. Only47 exact rows retained, no bounds converted to labels, no SI cap. Competition pair selection unchanged. External-only pairs and optional cross pairs require known named strain/cell and original similarity>=.5 top8 in either direction. Conservative aliases: Copenhagen, WR/Western Reserve, Lister; Vero, HFF/Human foreskin fibroblast, BSC-1/BSC1; unknown remains unavailable. Competition original context gate and query anchors remain unchanged, outer-train competition only. No external lookup or direct prediction substitution. Training pair weights:1 competition, .1 or .5 external-containing; alpha10 unchanged. Pair audits record support and high-external-target weighted squared-delta influence. Full320 raw MSE, no target removal/clipping. Discovery51 freezes before52–55; scaffold stress. No submissions generated or uploaded.
