# Authorized pair-delta submission

User explicitly approved this candidate after the local CV checkpoint, despite
the unmet <=3M-each gate. One upload was made on 2026-11-07.

- File: `submission-pair-delta-v4.csv`; 597 rows in exact sample index order.
- SHA256: `cbd1b6c1e7ebb858c9ca9bda38dbc24392c7ef81271228ef4edce96edd87ab60`.
- Frozen model: Ridge-log from commit3435851, trained on all320 officialv2 rows.
- External labels, test-label substitution and forbidden columns: none.
- Kaggle status observed: Success; new personal best.
- Public score: **1,710,149.26680**.
- Previous best: 2,038,068.29444; reduction **16.09%**.
- Rank observed: **16**, previously28; rank is time-sensitive.
- Public uses approximately30% of test; private score remains unknown.

Observed on the authenticated competition leaderboard:
https://www.kaggle.com/competitions/molecular-activity-prediction-hack/leaderboard

Verification: CSV shape/schema/index alignment, finite nonnegative predictions,
round-trip values and SHA256 checked; four pair-delta regression tests passed.
Read-only inspection found no blockers. No additional submissions authorized
by this one-candidate exception; the default local gate remains in force.
