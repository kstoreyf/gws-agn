fu_seed101 -- the m18 seed replication at seed 101 (owner-approved 2026-08-23).
Two dynesty arms, run sequentially on one RITA A100-80 (seed 102 runs its twin
on the other RITA GPU):

  1. campaign_m18_dynesty_s101    -- c_mode selection, the a5 redo cell verbatim
                                     with seed101 data.  Carries anchor recovery.
  2. campaign_m18_dynesty_pp_s101 -- c_mode per_pixel, same flags minus the fit.
                                     With (1) it carries the evidence separation
                                     (seed 100: per_pixel rails the GAL anchor at
                                     -1.808, selection recovers -3.118,
                                     delta lnZ = +17.7 -- but that number was
                                     archive-vs-new across SHAs; this pair is the
                                     clean same-SHA (0c5b3db) contrast).

Selection fits: seed 100's truez fits, held FIXED across seeds on purpose.
They estimate universe-level LF parameters (sigma(Mstar) ~ 0.005 mag from an
821k-galaxy fit), an order of magnitude below anything the replication tests,
and holding the assumed selection fixed isolates data-realisation scatter --
the axis this replication exists to measure.

rstate_seed 7 everywhere, matching the s100 cells, so dynesty noise is paired.
Both tasks checkpoint every 900 s and RESUME from ${RESULTS}/*.ckpt if the job
is requeued.
