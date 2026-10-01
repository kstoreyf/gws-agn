# Analysis 8 scripts

This directory is for thin experiment drivers around production `darksirens`.

Rules:

- do not implement a second population likelihood here;
- do not copy large pieces of `darksirens` into this directory;
- import the production likelihood/data/population machinery;
- keep seed-100 paths explicit and auditable;
- write exact CLI arguments and upstream SHAs into result metadata;
- preserve the signed-off seed-100 Analysis-2 files;
- any new mock/output must have an Analysis-8-specific name.

Expected eventual script roles are intentionally not pre-created because their exact names should follow the implementation discovered in Gate A:

- null/equivalence check;
- marked seed-100 generation/validation;
- marked \((f_{\rm AGN},\Delta\mu_\chi)\) scan;
- three-arm aggregation;
- event-level evidence decomposition;
- production figures.

Do not create a seed loop or a generic multi-seed queue in this campaign.
