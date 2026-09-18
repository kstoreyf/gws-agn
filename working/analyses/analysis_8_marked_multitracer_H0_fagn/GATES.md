# Analysis 8 gates

This file is the concise pass/fail ledger. Detailed evidence belongs in `diagnostics/`, `REPORT.md`, and `STATE.md`.

## Scope lock

- [x] Analysis defined as marked GAL/AGN multi-tracer HBI.
- [x] First mark is a GAL/AGN difference in \(\mu_{\chi_{\rm eff}}\) only.
- [x] Seed 100 only.
- [x] Complete catalogs only.
- [x] \(H_0=67.74\) fixed for this campaign.
- [x] Common mass, \(q\), redshift evolution and spin width fixed.
- [x] Multiple realizations require an explicit owner gate.

## Gate A — null/equivalence

Status: **PENDING**

Required:

- [ ] new tracer-dependent population path identified/implemented;
- [ ] \(\Delta\mu_\chi=0\) reproduces existing Analysis-2 likelihood;
- [ ] \(f=0\) reproduces GAL endpoint;
- [ ] \(f=1\) reproduces AGN endpoint;
- [ ] selection term reproduces existing K=2 result;
- [ ] seed-100 Analysis-2 \(f_{\rm AGN}\) posterior reproduced;
- [ ] evidence written to `diagnostics/null_equivalence.*`.

If Gate A fails, stop before generating a marked mock.

## Gate B — marked seed-100 mock integrity

Status: **BLOCKED ON A**

Registered mark:

\[
\Delta\mu_\chi^{\rm plant}=+0.10.
\]

Required:

- [ ] existing seed-100 LSS/catalog realization reused;
- [ ] existing signed-off files unchanged;
- [ ] one new marked seed-100 event family only;
- [ ] host fraction recorded;
- [ ] GAL/AGN truth spin means differ by the registered amount;
- [ ] masses and \(q\) have no planted channel difference;
- [ ] v3 PE contract unchanged;
- [ ] selection support validated;
- [ ] validation written to `diagnostics/marked_mock_validation.json`.

## Gate C — seed-100 marked recovery

Status: **BLOCKED ON B**

Exactly three scientific arms:

- [ ] S: spatial-only;
- [ ] I: intrinsic-only;
- [ ] J: joint spatial+intrinsic.

Required:

- [ ] joint \(f_{\rm AGN}\) recovery;
- [ ] joint \(\Delta\mu_\chi=+0.10\) recovery;
- [ ] spatial-only result;
- [ ] intrinsic-only result;
- [ ] joint result;
- [ ] selection guard valid over posterior support;
- [ ] event-level spatial/intrinsic evidence decomposition;
- [ ] production figures;
- [ ] `REPORT.md`.

## Owner gate

Status: **LOCKED**

After Gate C, stop.

Forbidden before explicit owner approval:

- seeds 101/102/103/105;
- any additional realization;
- free \(H_0\);
- mass marks;
- incompleteness;
- GP/HSGP population differences;
- GWTC data.

Completion line required from the driver:

> **OWNER GATE: seed-100 Analysis 8 is complete. I have not run additional realizations.**
