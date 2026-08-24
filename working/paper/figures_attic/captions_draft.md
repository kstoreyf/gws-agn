# Caption drafts (factual base; prose to be rewritten)

Every number below is read from the result file named in the provenance line
under each caption. Nothing is rounded by hand: the figures format the same
values the jsons carry.

---

## fig_pgm — the generative model

The graph of the simulated universe. Shaded nodes are observed, open nodes are
latent, small filled squares are fixed hyperparameters, and double-ringed nodes
are deterministic functions of their parents. One cosmology fixes the shell
geometry and the power spectrum, from which a single lognormal density field
$\delta$ is drawn; the tracer plate samples that one field twice, each tracer
with its own bias $b_k$ and mean comoving density $\bar{n}_k$, into a complete
catalog $C_k$, and the survey sees $\hat{C}_k$ — the same objects with photo-z
scatter $\sigma_z$ applied to every redshift and a flux limit $m_{\rm lim}$
applied to the magnitudes. In the event plate a tracer label $t_i$ is drawn
with probability $f_{\rm AGN}$, a host $h_i$ from that tracer's *complete*
catalog at its true redshift, and source parameters $\theta_i$ from the fixed
population. The measurement is the observed signal-to-noise ratio
$\hat{\rho}_i = \rho_i + \mathcal{N}(0, \sigma_\rho)$, and everything else
hangs off it: detection is the deterministic threshold
$\hat{\rho}_i \ge \rho_{\rm th}$, and every measurement width — chirp mass,
mass ratio, spin, sky — is $a_x (8/\hat{\rho}_i)$. There is no distance
measurement node anywhere in the graph: the optimal SNR is
$\rho_i \propto M_c^{5/6}/d_L$, so the SNR *is* the distance coordinate and
$d_L$ is recovered from $(\hat{M}_c, \hat{\rho}_i)$ rather than measured.

<!-- provenance: working/data/DESIGN_PE.md (v3 measurement family, §2.1-2.6);
     working/data/seed100/META.json stages.events.{measurement_model,pe_model,
     detection} and stages.catalogs.photoz_model. Constants shown as symbols:
     sigma_rho = 1.0, rho_th = 8, a_Mc = 0.08, a_q = 0.60, a_chi = 0.20,
     sky width = clip(35 deg/(1.83165 rho_obs), 1, 12) deg, sigma_z = 3e-3(1+z).
     Hand-drawn diagram, no data reduction. The analyses reported here use the
     complete-catalog limit of C_k; m_lim is the flux limit of the incomplete
     surveys the same generator writes. -->

---

## fig_single_tracer — one catalog at a time

The same 1000 dark sirens, analysed twice against the wrong universe. Against
the complete galaxy catalog alone the expansion rate comes back at
$H_0 = 69.9^{+1.7}_{-1.6}$ km s$^{-1}$ Mpc$^{-1}$, high of the true 67.74 by
about 1.3 of its own 68 % half-widths; against the complete AGN catalog alone
the posterior has no interior maximum at all — it climbs monotonically to the
top of the scanned range and is still rising at $H_0 = 100$, so no interval is
quoted for it. Both curves are scaled to their own peaks, the galaxy 68 %
interval is shaded, and neither analysis knows that the events were drawn from
a mixture of the two catalogs.

<!-- provenance: analysis_1_complete_catalog_H0/results/h0_gal_targeted.{h5,json}
     (median 69.9103, ci68 [68.3379, 71.6341], MAP 69.75) and
     h0_agn_targeted.{h5,json} (MAP 100.0, median 99.797, still rising at the
     grid top); h0_single_tracer.json records agn_h0_ci = null,
     agn_railed_at_grid_top = true. Seed 100, 201-point H0 grid on [50, 100],
     targeted injection lane, no cells rejected. Curves are peak-normalised
     because the two densities differ by a factor of 14 in peak height. -->

---

## fig_joint — both parameters at once

The joint fit on the two complete catalogs, with the mixture weight free. For
the reference realisation (blue, 68 % and 90 % credible regions) the expansion
rate and the AGN host fraction are recovered together,
$H_0 = 69.2^{+1.0}_{-1.0}$ km s$^{-1}$ Mpc$^{-1}$ and
$f_{\rm AGN} = 0.273^{+0.049}_{-0.047}$, against a true 67.74 and a realised
AGN host fraction of 0.295 (0.30 planted); the two parameters are only weakly
correlated. Four further realisations of the whole chain are drawn as outlines,
and the side panels are the two marginals of the same grids. Adding the sparse
tracer and its weight does not cost the expansion rate: this $H_0$ interval is
narrower than the galaxy-only interval of the previous figure (1.94 against
3.30 km s$^{-1}$ Mpc$^{-1}$ full 68 % width).

<!-- provenance: analysis_2_complete_catalog_H0_fagn/results/joint_s{100,101,
     102,103,105}.h5 (201 x 41 grid on [50,100] x [0,1], 8241 evaluations,
     0 rejected) and joint_s100.json (H0 median 69.2170, ci68 [68.2491,
     70.1915]; f median 0.27335, ci68 [0.22598, 0.32224]; rho = +0.0678; MAP
     (69.25, 0.275)); joint_summary.json seeds[].f_realised = 0.295, 0.326,
     0.308, 0.277, 0.289. Credible regions are highest-posterior-density
     contours of the flat-prior posterior; marginals are the same grids
     integrated along the other axis. The cross is at (67.74, 0.295): the
     realised fraction of the reference realisation, not the planted 0.30. -->

---

## fig_closure — five realisations

Each realisation is an independent universe — its own density field, catalogs
and 1000 events — put through the same joint fit. Points are the medians and
bars the 68 % intervals; the band is the five-realisation mean offset with its
standard error. The expansion rate returns $+0.41 \pm 0.55$ km s$^{-1}$
Mpc$^{-1}$ from truth and the AGN fraction $-0.012 \pm 0.020$ from the fraction
each realisation actually drew, both consistent with zero. Because the mock
draws its host labels binomially, the value the fit should return is not the
planted 0.30 but the realised fraction of that realisation (black ticks, 0.277
to 0.326); the band therefore steps with them.

<!-- provenance: analysis_2_complete_catalog_H0_fagn/results/joint_summary.json
     seeds[].joint.H0 and seeds[].joint.f_vs_realised (median, ci68) and
     closure.joint_H0 = {mean +0.40747, sem 0.55466, n 5} and
     closure.joint_f_vs_realised = {mean -0.011673, sem 0.020425, n 5}; the
     same two offsets appear in h0_fagn_joint.json. Realised fractions from
     seeds[].f_realised. Per-realisation binomial sd is 0.0145. -->

---

## fig_null — the sky-shuffle null

What the mixture weight is measuring. The blue curve is the AGN fraction
inferred from the events as recorded, with the expansion rate held at its true
value: median 0.266, 68 % interval [0.221, 0.313]. The orange curve is the
identical analysis of the identical events after the per-event sky samples have
been permuted between events, so every event keeps its own distance, masses,
spin and localisation area but no longer sits in its own host's patch of sky.
The weight collapses to 0.037, with 90 % of its posterior below 0.106: the
fraction is carried by host association and not by any difference in the two
tracers' global normalisations, which the permutation leaves untouched.

<!-- provenance: analysis_2_complete_catalog_H0_fagn/results/fscan_s100.{h5,json}
     (median 0.26650, ci68 [0.22120, 0.31295], ci90 [0.19221, 0.34375]) and
     fscan_null_s100.{h5,json} (median 0.037462, ci68 [0.011656, 0.076244],
     ci90 [0.0036476, 0.10580], MAP 0.01). Both are 101-point f scans at fixed
     H0 = 67.74, seed 100, 0 cells rejected; the null was run on
     data_derived/events_skyshuffled_s100.h5, written by that analysis's
     scripts/shuffle_event_sky.py. Curves are peak-normalised; the shaded
     bands are the 68 % intervals. -->

---

## fig_pure_tracer — one tracer at a time, at matched event count (appendix)

What each catalog can do on its own, measured on events it actually hosts. For
every realisation of the simulated universe the generator drew two further sets
of 1000 detected dark sirens on the same two catalogs — one set with every host
a galaxy, one with every host an AGN — and each set was analysed against its own
catalog alone, with the expansion rate the only free parameter. *Left:* the ten
posteriors, blue for the galaxy sets and orange for the AGN sets, each scaled to
its own peak because the AGN densities are about five times narrower and
correspondingly taller; the reference realisation is drawn at full strength in
each colour and the other four lighter and thinner, and the dashed line is the
true $H_0 = 67.74$ km s$^{-1}$ Mpc$^{-1}$. The horizontal range is trimmed to
$[56, 80]$, outside which every curve holds less than $10^{-4}$ of its mass. One
galaxy realisation is genuinely bimodal — a second mode at $H_0 = 62.00$ standing
at 0.70 of the peak at 69.75 — and it is drawn as it is, which is why its
interval is wide. *Right:* the same ten measurements as medians with their 68 %
intervals against the same truth line, galaxies offset left and AGN right at each
realisation, with each tracer's five-realisation mean offset drawn as a line and
its standard error as the band around it. At equal event count the AGN catalog
gives a mean 68 % half-width of 0.45 km s$^{-1}$ Mpc$^{-1}$ against the galaxy
catalog's 2.29, a factor of 5.1; both tracers return truth, the AGN sets by
$-0.001 \pm 0.227$ and the galaxy sets by $+0.594 \pm 0.864$ km s$^{-1}$
Mpc$^{-1}$. Five realisations fix a mean offset, not a coverage fraction: the
counts of intervals containing truth (3 of 5 at 68 % and 4 of 5 at 90 % for
galaxies, 2 of 5 and 5 of 5 for AGN) are consistent with the quoted intervals but
do not measure them.

<!-- provenance: analysis_0_pure_tracer_H0/results/h0_pure{gal,agn}_targeted_s{100,
     101,102,103,105}.h5 (H0_grid + log_likelihood, 201 points on [50,100], flat
     prior, 0 cells rejected in all 20 scans) for the curves, and the matching
     .json H0 blocks (median, ci68) for the right panel's points and bars.
     Aggregates from h0_pure_tracer.json: closure_gal.widths.mean_half68 =
     2.28759 and closure_agn.widths.mean_half68 = 0.45268 (ratio 5.0534);
     closure_gal.{mean_offset, sem_offset} = (+0.59402, 0.86380) and
     closure_agn = (-0.00137, 0.22654); coverage.n_truth_in_ci68 / ci90 = 3/4
     (gal) and 2/5 (agn). The bimodal realisation is seed 105, galaxy set:
     diagnostics.per_scan[h0_puregal_targeted_s105].mode_positions = [62.0,
     69.75] with relative heights [0.70428, 1.0]; its ci68 is [62.51, 71.00],
     which is the wide bar on the right. Every event set has n_events = 1000
     (closure_*.widths.n_events_per_seed). Record lane = targeted; the popuni
     lane is the cross-check and is not plotted (largest median shift 0.49 of a
     half-width for the galaxy sets, 0.85 for the AGN sets). Peak-normalisation
     is applied per curve in the left panel only; the right panel is in
     km/s/Mpc. Curves are the raw scan grids -- no smoothing, no interpolation
     beyond matplotlib's straight segments between grid points. -->

<!-- macros for the appendix text: \PureNseeds \PureNevents \PureGalHalfWidth
     \PureAgnHalfWidth \PureWidthRatio \PureWidthRatioPerSeed \PureGalOffset
     \PureAgnOffset \PureGalInSixtyEight \PureGalInNinety \PureAgnInSixtyEight
     \PureAgnInNinety \PureLaneMaxGal \PureLaneMaxAgn \PureBimodalSeed
     \PureBimodalModeLo \PureBimodalModeHi \PureBimodalHeight -->
