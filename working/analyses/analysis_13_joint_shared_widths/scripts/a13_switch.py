"""Switch a restored dynesty (2.1.x) MultiEllipsoidSampler from multi/unif to rslice.

Owner decision 2026-10-05: seed 1 stalled twice under multi/unif (dynesty's default bootstrap = 5
inflates the ellipsoids; ~1,060 calls/it from it 2833 on, single iterations up to 11.4k calls).
Nested sampling only needs independent draws from the prior above the current threshold, so the
proposal can change mid-run; the dead points already collected keep their weights.

The new proposal uses dynesty's rslice defaults for this dimension: slices = 3 + ndim, enlarge 1.25,
no bootstrap, bound update every 2 * slices * nlive calls. The switch is idempotent: a sampler that
already uses the target method is left alone (so a later restore keeps its tuned slice scale).
"""
import numpy as np


def switch_sampler(s, method="rslice"):
    """Rewire ``s`` (restored, callables rebound) to ``method``; return a record of the switch."""
    from dynesty.nestedsamplers import _SAMPLING
    if method != "rslice":
        raise ValueError("only rslice is wired")
    if s.method == method:
        return None
    rec = {"from": s.method, "to": method, "it": int(s.it), "ncall": int(s.ncall),
           "bootstrap_before": int(s.bootstrap), "enlarge_before": float(s.enlarge)}
    slices = 3 + s.ncdim
    s.method = s.sampling = method
    s.evolve_point = _SAMPLING[method]
    s.propose_point = s.propose_live
    s.update_proposal = s.update_slice
    s.enlarge, s.bootstrap = 1.25, 0
    s.kwargs.update(enlarge=1.25, bootstrap=0, slices=slices)
    s.slices = slices
    s.scale = 1.0
    s.slice_history = {"ncontract": 0, "nexpand": 0}
    s.bound_update_interval = int(round(2.0 * slices * s.nlive))
    # discard any proposal queued under the old method, then rebuild the ellipsoids unbootstrapped
    s.queue, s.nqueue = [], 0
    s.update_bound_if_needed(-np.inf, force=True)
    rec.update(slices=slices, enlarge=1.25, bootstrap=0, bound_update_interval=s.bound_update_interval,
               n_ellipsoids=int(len(s.mell.ells)))
    return rec
