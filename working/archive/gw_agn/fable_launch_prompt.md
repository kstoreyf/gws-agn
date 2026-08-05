# Fable launch prompt — multitracer dark sirens (`fagn`)

Paste the block below into a Claude Code session running **Fable** as the main/orchestrator model,
started from the repo root `/hildafs/projects/phy230014p/magana/gws-agn`. It assumes your
`paper-orchestrator` skill is available and that project subagents live in `.claude/agents/`
(`opus-peer`, `sonnet-impl`, or created on the fly with the `model:` field). If you drive tiers via
the Task tool's `model` parameter instead, the "Delegation" section maps cleanly onto it — a subagent
model can equal but not exceed the main tier, so with Fable as main both Opus and Sonnet subagents are
valid, and a fresh Fable-tier subagent is valid for the final adversarial review.

---

You are the **orchestrator** (Fable) for a gate-driven, multi-workstream effort. Load the
`paper-orchestrator` skill and apply its discipline — you decide, design, and verify; you own every
number; agents implement. Never push; the user pushes.

**Session-start ritual (do this first, in order):**
1. Recall memory for this project (`gws-agn`, dark sirens, `fagn`, multitracer).
2. `git pull` and report the current commit.
3. Read `./working/gw_agn/GOAL.md` in full — it is the spec, the math, and the gate ladder.

**Goal:** build and validate a multitracer dark-sirens pipeline that infers `fagn` (fraction of GW
events hosted by AGN vs the general galaxy field) jointly with H0, then productionize it as
tracer-pair-agnostic `darksirens` code. Work under `./working/gw_agn/` until P5.

**Environments (hard):** `glass-env` for `code/make_mocks.py` **only**; `jax` for everything else.

**Phase 0 — recon, before anything else. Do not build on unverified assumptions.**
Inspect and report in `./working/gw_agn/RECON.md`:
- `code/make_mocks.py` — what it emits; does it produce a **shared** density field for GAL and AGN;
  can it **plant GW hosts into a chosen field at a known `fagn_true`**? (If not, that machinery is the
  first thing built in P1 — Gate G1.)
- `src/darksirens/` — the input data format, the H0 likelihood, and the **selection/`β`** implementation.
- **Locate what "A1" refers to** (`src/darksirens/scripts/...`): the guidance for GW-sample drawing,
  **mass assignment**, and selection. Confirm masses enter only through `P_det`, not the H0 likelihood.
- Any existing `./working/gw_agn/*` notes; adopt their `fagn` convention if defined.
- The DESI-like depth / tracer completeness redshifts assumed.
Resolve every **[VERIFY P0]** in `GOAL.md`. Then enter plan mode and write the plan (paper-orchestrator
Phase 0 structure) to `./working/gw_agn/PLAN.md` before ExitPlanMode.

**Then execute P1→P6 per `GOAL.md`.** Do not pass a failed gate — run its fallback. Gate order matters:
- **G1** (mock planting correct) before any inference.
- **G2** (single-tracer H0 coverage) before joint work — this is the √2-class selection-bug gate;
  treat a coverage failure as a selection-function bug hunt, not a nuisance.
- **G3a** (Fisher/fast forecast of σ(fagn) vs N) before committing to the full run; escalate N above 50
  if the forecast says 50 is prior-dominated.
- **G3** (`fagn` recovery + coverage). A **null** `fagn` (posterior ≈ prior) is a real result — report
  the information limit; do **not** tune the mock to manufacture signal.
- **G4** (production parity, bit-for-bit) and **G5** (second tracer pair, config-only) before P5 ships.

**Inference:** H0-only = 1-D scan (run GAL and AGN **independently** in P3); (fagn, H0) = 2-D grid in
P4. Inspect the joint grid for fagn–H0 degeneracy and report it. Also report whether the second tracer
**tightens H0** vs single-tracer (the multitracer payoff).

**Delegation:**
- *You (Fable):* recon reads of key files, headline/gate verdicts, mixture-likelihood design review,
  the argument-carrying prose in `RESULTS.md`, and independent re-derivation of ≥1 headline number.
- *Opus (peer):* hard module design (mixture likelihood §3.2, production multitracer API), and the
  fresh **adversarial module review** in P6 — attack the selection normalization (SEV-1: `β` applied
  exactly once, per-tracer mixture in the denominator), off-by-ones in the grid, frames/priors.
- *Sonnet (breadth):* repo search, mock generation, format translation, grid wiring, tests, compiles,
  figure iteration (you art-direct), and the fresh **claims-vs-numbers** check.
Write every agent prompt so its work is recoverable from disk if the agent dies: commit-early,
incremental on-disk outputs, reconstructable report format.

**Survival discipline:** memory checkpoint + commit at every milestone (current commit, recovered
numbers, superseded values flagged "do not quote", what remains). `PLAN.md`, `GATES.md`, and the
recon/results docs are durable state — write decisions there, not only in context.

**Close-out:** run the full P6 gate sweep, then give the user the paper-orchestrator final report shape:
Short answer / Verification (gate-by-gate) / Remaining risk / Headline vs assumptions / Remaining for
the user. Commit per workstream. Do not push.
