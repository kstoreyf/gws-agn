#!/usr/bin/env python3
"""Write SEEDS_FIX.md: pre-fix vs post-fix comparison of the 12-seed rerun."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"

pre = json.loads((RES / "seeds_summary.json").read_text())
post = json.loads((RES / "seeds_summary_fix.json").read_text())
chk_det = json.loads((RES / "fixcheck_detection_sigma_ang.json").read_text())
chk_reg = json.loads((RES / "fixcheck_s7301_regeneration.json").read_text())

KEYS = (("fscan_f", "f_AGN (f-scan, H0 fixed)", 0.30),
        ("joint_f", "f_AGN (joint)", 0.30),
        ("joint_H0", "H0 (joint) [km/s/Mpc]", 67.74))


def row(key, label):
    a, b = pre[key], post[key]
    def sig(st):
        s = st.get("sigma_from_zero")
        return f"{s:.1f}" if s else "-"
    return (f"| {label} | {a['mean']:+.4f} +- {a['sem']:.4f} ({sig(a)} sigma) "
            f"| {b['mean']:+.4f} +- {b['sem']:.4f} ({sig(b)} sigma) "
            f"| {a['sd']:.4f} / {a['mean_quoted_half_width']:.4f} = "
            f"{a['scatter_over_quoted_half_width']:.2f} "
            f"| {b['sd']:.4f} / {b['mean_quoted_half_width']:.4f} = "
            f"{b['scatter_over_quoted_half_width']:.2f} |")


guards = [r.get("passes_guard") for r in post["per_seed"]]
rej_f = [r.get("fscan_rejected") for r in post["per_seed"]]
rej_j = [r.get("joint_rejected") for r in post["per_seed"]]

txt = f"""# SEEDS_FIX: 12-seed two-tracer rerun with the sigma_ang-fixed generator

Rerun of the 12-seed (7301-7312) deep two-tracer campaign with the FIXED mock
generator (`darksirens-oraclefix` @ `fix/mock-observable-sky-width`, includes
merged #332/#334):

* `detection_data=observed` (PR #334): ONE measurement per source; the SNR
  threshold is applied to that measurement and the posterior conditions on the
  same record (`use_recorded_observation`).  `snr_ref = 6.278` (the matched-mock
  calibration) reproduces the historical detected fraction on this catalog
  family to 1.0% (pilot 2M draws on the s7301 catalog:
  {chk_det['V3_snr_ref_calibration']['frac_true_rule']:.6f} true rule vs
  {chk_det['V3_snr_ref_calibration']['frac_observed_rule_ref6.278']:.6f} observed rule).
* Sequential observable sky width (PR #335): sigma_ang = clip(35/rho, 1, 12) deg
  with rho the projection-free amplitude of the OBSERVED masses/distance, so the
  sky-noise width is a function of the recorded data, not of the latent true
  parameters (the -0.49 +- 0.08 exact-likelihood H0 bias mechanism).

## What was reused vs regenerated (recorded checks: results/fixcheck_*.json)

* REUSED: per-seed complete catalogs (`data_derived/s73XX/catalog_complete.h5`)
  -- the catalog draw involves no sky width; and the surveys are regenerated but
  verified BIT-IDENTICAL to the pre-fix ones (the AGN-subset draw precedes the
  event draws in the rng stream).
* Generator drift check: the oraclefix generator in default (`true`) mode
  reproduces the pre-fix s7301 events file bit-for-bit (all 25 datasets), i.e.
  the fix is inert unless the observed-data mode is selected.
* Detection provably does NOT involve the sky width: identical rng state with
  sequential vs fixed 5-deg sigma_ang gives bit-identical detection masks and
  observed dL/masses/SNR (sigma_ang enters only the sky scatter and the PE).
* Injections REDONE anyway: the detection RULE changed (projection-latent true-
  parameter SNR -> observed-data SNR), so the old selection campaign is not the
  events' selection function.  Old-vs-new s7301 detected sets share only
  {chk_reg['old_vs_new_detected_set']['n_shared_host_z_values']}/200 host
  redshifts (z_max {chk_reg['old_vs_new_detected_set']['z_max_old']:.3f} ->
  {chk_reg['old_vs_new_detected_set']['z_max_new']:.3f}).
  New injections: same 0.65/0.10/0.25 population/uniform/AGN-targeted mixture,
  120M proposals per seed, observed-data detection at snr_ref 6.278.

Estimator side unchanged: frozen darksirens master 2b86a2d, K=2
`dark_sirens` + field weighting, log10n0 = log10n0_c2 = -12, legacy guard
(`--selection_neff_guard hard --max_likelihood_variance 1e6`), same grids.

## Result table (mean offset from truth +- sem over 12 realisations)

| statistic | pre-fix offset | post-fix offset | pre sd/hw | post sd/hw |
|---|---|---|---|---|
{row('fscan_f', 'f_AGN (f-scan)')}
{row('joint_f', 'f_AGN (joint)')}
{row('joint_H0', 'H0 (joint)')}

(sd = seed-to-seed scatter; hw = mean per-seed quoted 68% half-width; the ratio
is the factor by which single-realisation intervals understate the
realisation-to-realisation uncertainty.)

Guard: passes_legacy_floor = {guards}; f-scan rejected cells per seed = {rej_f};
joint rejected cells per seed = {rej_j}.

## Reading

* Pre-fix reference (results/seeds_summary.json, UNTOUCHED): f-scan f
  -0.0537 +- 0.0090 (5.9 sigma), joint f -0.0273 +- 0.0130 (2.1 sigma), joint
  H0 -3.22 +- 0.55 (5.8 sigma) with H0 seed scatter 2.67x the quoted widths.
* Post-fix numbers above are the new scientific result; the H0 prediction was
  a collapse toward the darksirens estimator-overhead scale (-0.31 +- 0.13 on
  the K=1 oracle campaign, plus small documented latents ~ -0.05).

Figures: figs/seeds_fix_strip.png (per-seed offsets, paired pre->post),
figs/seeds_fix_widths.png (scatter vs quoted width).
Per-seed outputs: results/{{fscan,joint,guard}}_fix_s73XX.*,
data_derived/s73XX_fix/.
"""
(BASE / "SEEDS_FIX.md").write_text(txt)
print("wrote", BASE / "SEEDS_FIX.md")
