# GOAL — Multitracer dark sirens: inferring the AGN-hosted GW fraction (`fagn`)

**Repo:** `/hildafs/projects/phy230014p/magana/gws-agn`
**Workspace for this effort:** `./working/gw_agn/`
**Orchestration:** `paper-orchestrator` skill (gate-driven, multi-workstream). Fable orchestrates and
owns every number; Opus is peer tier (hard module design, adversarial review); Sonnet does
search / read / edit / mechanical implementation.
**Environments:** `glass-env` for `code/make_mocks.py` (GLASS lognormal mocks) **only**; `jax` for
everything else.

> **Status of this document.** Written *before* repo inspection. Every line tagged **[VERIFY P0]**
> is an assumption the orchestrator must confirm against the actual code/notes in Phase 0 before any
> downstream work depends on it. Do not treat unverified assumptions as true — that is the same
> discipline that keeps wrong numbers out of a manuscript.

---

## 1. One-line goal

Build and validate a **multitracer dark-sirens** pipeline that infers the mixing fraction
`fagn` — the fraction of GW events hosted by AGN rather than by the general galaxy field —
**jointly with H0**, using an AGN catalog and a galaxy catalog as independent redshift tracers of
the *same* underlying density field. Deliver it first as a proof-of-concept under `./working/gw_agn/`,
then as production `darksirens` code that generalizes to any tracer pair (AGN/BGS, BGS/LRG, …).

## 2. Why this is not trivial (the scientific risk to gate, not assume away)

`fagn` is a mixing weight between two host-probability fields that are **correlated** (AGN and
galaxies trace the same large-scale structure with different bias and number density). The
recoverable information about `fagn` comes only from where the two fields *differ*:

- **Number-density contrast** — AGN are rarer, so an event whose true host is an AGN localizes to a
  smaller effective host set → sharper redshift → distinguishable likelihood shape.
- **Clustering/bias contrast** — only present if the mocks encode different bias; a lognormal field
  with a single linear bias per tracer gives limited contrast on the scales GW localization probes.

If AGN are merely a sparse random subsample of the galaxy field, `fagn` may be **prior-dominated at
N=50**. This is the primary scientific risk. **It is gated, not assumed** (Gate G3). A null result
(`fagn` unconstrained) is a legitimate, publishable outcome and must be reported honestly, not
engineered away.

## 3. The math (single source of truth for validation)

All quoted numbers must trace to these expressions; the adversarial review reproduces at least one
headline number from raw catalog outputs against them.

### 3.1 Single-tracer dark-siren likelihood (complete catalog)

For GW event `i` with data `d_i` and tracer `X ∈ {GAL, AGN}` providing a normalized host-probability
field `p_X(z, Ω̂)` built from catalog members (weighted by whatever host-selection weight `w_j` the
A1 guidance specifies — **[VERIFY P0]** uniform-in-comoving-volume vs luminosity-weighted):

```
N_X(d_i | H0) = ∫ dz dΩ̂  p(d_i | z, Ω̂, H0)  p_X(z, Ω̂)          (unnormalized single-event integral)
β_X(H0)       = ∫ dz dΩ̂  P_det(z, Ω̂, H0)     p_X(z, Ω̂)          (selection normalization)
p(d_i | H0, X) = N_X(d_i | H0) / β_X(H0)
```

`p(d_i | z, Ω̂, H0)` maps the GW distance posterior to redshift through `d_L(z, H0)` at fixed
(pre-registered) population — masses enter **only** through `P_det` (SNR/detectability), never the H0
likelihood directly. **[VERIFY P0]** confirm masses are used solely for selection in the A1 path.

### 3.2 Mixture likelihood for `fagn`

Host drawn from the AGN field with probability `fagn`, else from the galaxy field:

```
                 fagn · N_AGN(d_i | H0) + (1 − fagn) · N_GAL(d_i | H0)
p(d_i|H0,fagn) = ─────────────────────────────────────────────────────
                 fagn · β_AGN(H0)      + (1 − fagn) · β_GAL(H0)

ln L(H0, fagn) = Σ_i ln p(d_i | H0, fagn)
```

**Convention (fixed for the proof of concept, [VERIFY P0] against existing notes):** `p_GAL` and
`p_AGN` are each independently normalized host-probability fields; `fagn ∈ [0,1]` is the mixing
weight; the injected truth `fagn_true` is the fraction of *planted* GW hosts drawn from the AGN field.
This makes recovery unambiguous and sidesteps the AGN⊂GAL double-counting question. If existing notes
define `fagn` differently (e.g. AGN vs non-AGN galaxies), adopt theirs and update this block.

> **Selection-normalization bug class (SEV-1 review target).** The denominator is the *mixture* of
> per-tracer `β`'s, not `β` of the mixture-of-fields computed some other way, and `β` must be applied
> **exactly once**. This is the direct analogue of the √2 channel double-count in the LISA work.
> Verified explicitly in Phase 6 adversarial module review.

## 4. Phase plan (mapped to paper-orchestrator)

| Phase | Work | Model tier |
|---|---|---|
| **P0 Recon** | Inspect repo: `make_mocks.py`, `src/darksirens/`, the **A1** scripts, existing `./working/gw_agn/*` notes. Resolve every **[VERIFY P0]**. Produce `RECON.md`. | Fable reads key files itself; Sonnet does breadth search |
| **P1 Mocks** | Generate GLASS mocks (`glass-env`): shared density field → GAL + AGN catalogs with **injectable, known `fagn_true`** for planted GW hosts. If planting machinery absent → build it (Gate G1 fallback). | Sonnet implements; Fable reviews design |
| **P2 GW samples + selection** | Per A1 guidance: draw GW events, assign masses, apply SNR selection, produce distance/sky posteriors. Translate GAL and AGN catalogs into `darksirens` input format. | Sonnet implements against A1; Opus reviews the selection module |
| **P3 Single-tracer H0** | Run `darksirens` H0-only (1-D scan) **independently** on GAL and on AGN. Must pass H0 coverage gate (G2) before proceeding. | Sonnet runs; Fable verifies posteriors |
| **P4 Joint (fagn, H0)** | Implement mixture likelihood (§3.2), 2-D grid over (fagn, H0). Ingest both catalogs. Recover injected `fagn_true` (Gate G3). | Opus designs the mixture module; Sonnet wires the grid; Fable owns verdicts |
| **P5 Productionize** | Refactor the validated mixture into `src/darksirens` as a general **multitracer** API (tracer-pair agnostic: AGN/BGS, BGS/LRG, …). Tests, docstrings, CLI/config. | Opus designs API; Sonnet implements/tests; Fable reviews |
| **P6 Close-out** | Full gate sweep (§6). Adversarial module review (Opus/Fable-tier fresh agent). Commit per workstream. **Never push.** Memory checkpoint. | per skill |

## 5. Constraints (from the request — hold these fixed)

- **Fixed population** — masses/spins pre-registered; no spectral-siren H0 information by construction.
- **Complete catalog / fixed survey** — DESI-like; galaxy selection = complete → only **GW** selection
  enters `β`. **[VERIFY P0]** the DESI-like depth and which tracer depths (BGS z≲0.4, LRG z≲1, QSO
  higher) the mock adopts.
- **Catalog-depth guard (Gate G0):** restrict GW events to the redshift shell where the catalog is
  complete; events beyond it silently re-introduce incompleteness and bias H0/`fagn`. Enforce and log
  the cut.
- **N = 50 events** as the nominal proof-of-concept, *but* size N from a quick Fisher/injection
  forecast (Gate G3a) — 50 may be too few for a non-null `fagn`; escalate N if the forecast says so.
- **Inference:** 1-D scan for H0-only (P3); 2-D grid for (fagn, H0) (P4). Grid over MCMC at PoC stage
  is correct — you want to *see* the joint shape and any fagn–H0 degeneracy, not just summary stats.

## 6. Gate ladder (numeric pass criteria stated before running; fallbacks pre-specified)

Record verdicts in `./working/gw_agn/GATES.md` + `gates_report.json` with environment freezes.

- **G0 — completeness shell.** GW events all within complete-catalog volume. *Pass:* 100% of retained
  events have true z < z_complete. *Fallback:* add explicit incompleteness term (out of PoC scope →
  document and restrict).
- **G1 — mock host planting exists & is correct.** `make_mocks.py` can plant GW hosts into a chosen
  field at a known `fagn_true`. *Pass:* recovered planted-host fraction (by construction, pre-inference)
  == `fagn_true` within Poisson error. *Fallback:* build the planting step in P1 before P4.
- **G2 — H0 coverage (single tracer).** Inject known H0; run P3 on ≥20 mock realizations per tracer.
  *Pass:* posterior coverage of true H0 within the 68%/90% credible interval consistent with nominal
  (e.g. 90% CI contains truth in 90%±sampling). *Fallback:* selection-function bug hunt (this is the
  √2-class gate) before any joint work.
- **G3 — `fagn` recovery.** Inject `fagn_true ∈ {0, 0.3, 0.7, 1.0}`; run P4. *Pass:* recovered `fagn`
  posterior covers truth with correct coverage across realizations AND is prior-informative (posterior
  narrower than prior) for at least the non-trivial injections. *If null* (posterior ≈ prior): that is
  a **result** — report the information limit, do not tune the mock to manufacture signal.
- **G3a — forecast before committing N.** Fisher or fast injection estimate of σ(fagn) vs N *before*
  the full P4 run. Sets nominal N; if σ(fagn) at N=50 spans most of [0,1] for the fiducial contrast,
  escalate N or document the null.
- **G4 — production parity.** The P5 multitracer API reproduces the P4 proof-of-concept numbers
  **bit-for-bit** on the same mocks. *Pass:* exact equality. *Fallback:* no refactor ships until parity.
- **G5 — tracer-pair generality.** P5 API runs end-to-end on a second tracer pair (e.g. BGS/LRG mock)
  without code change, only config. *Pass:* completes and returns a sane posterior.

## 7. Deliverables

1. `./working/gw_agn/RECON.md` — Phase-0 findings; every **[VERIFY P0]** resolved.
2. `./working/gw_agn/GATES.md` + `gates_report.json` — gate verdicts with provenance.
3. Mocks + planting machinery (P1), GW-sample/selection translation to `darksirens` format (P2).
4. Single-tracer H0 posteriors for GAL and AGN (P3) + coverage plots.
5. Joint (fagn, H0) 2-D grid posteriors (P4) + recovery/coverage plots.
6. **Production multitracer `darksirens` code** in `src/darksirens/` (P5): tracer-pair-agnostic
   likelihood, tests, docstrings, config-driven CLI. This is the real payoff — production level.
7. Short `RESULTS.md` written by Fable: the goal, the recovered numbers, the honest information limit,
   assumptions each claim costs, open questions.

## 8. Open scientific questions (surface, don't bury)

- **Multitracer payoff.** The point of two tracers is not just `fagn` — different-bias tracers of the
  same field can cancel sample variance on the joint (fagn, H0) posterior (cf. multitracer LSS,
  McDonald & Seljak 2009). Does adding the second tracer *tighten* H0 vs the single-tracer runs? Report
  the comparison; it is the argument for the production generalization.
- **fagn–H0 degeneracy.** Inspect the 2-D grid for correlation; both reshape the effective redshift
  prior. If strongly degenerate, `fagn` alone is not clean — say so.
- **Lognormal realism.** GLASS lognormal + linear bias will not capture AGN halo occupation / small-
  scale clustering. Fine for PoC; the production code must **not** bake in lognormal assumptions. Scope
  boundary, stated in `RESULTS.md`.
- **AGN⊂GAL vs disjoint.** Confirm the adopted convention doesn't double-count AGN in the GAL field, or
  that it washes out for the normalized-field mixture (§3.2).

## 9. References (anchor the framing; **orchestrator confirms exact IDs on arXiv/ADS — do not cite
from memory**)

Solid, high-confidence anchors:
- Schutz 1986, *Nature* 323, 310 — standard sirens.
- Chen, Fishbach & Holz 2018, *Nature* 562, 545 — dark-siren H0 forecast.
- MacLeod & Hogan 2008, *PRD* 77, 043512 — dark sirens with galaxy clustering.
- Gray et al. 2020, *PRD* 101, 122001 — `gwcosmo` dark-siren method / selection framework.
- Soares-Santos et al. 2019 (DES + GW170814) — statistical dark-siren H0.
- Magaña Hernández & Palmese 2024, *PRD* 111, 083031 — (author's own; dark-siren methodology).
- McDonald & Seljak 2009, *JCAP* 10, 007 — multitracer sample-variance cancellation.
- Tessore et al. 2023, *OJAp* — **GLASS** lognormal field generator.
- Graham et al. 2020, *PRL* 124, 251102 — GW190521 candidate AGN flare (motivation for AGN hosting).
- McKernan/Ford, Bartos et al. 2017, *ApJ* 835, 165 — AGN-disk BBH channel (motivation).

**Verify on arXiv before use (I am not confident of exact refs — do not fabricate):** the specific
`fagn`-from-GW–AGN-cross-correlation measurement papers (candidates to check: Bartos et al.;
Veronesi et al.; Vijaykumar/Fishbach et al.). Locate the actual method paper the existing notes rely on.
