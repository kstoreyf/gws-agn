"""Analysis 16: the defensive mixture redraw of the posterior samples (driver option --redraw-mixture support).

The likelihood of merger i is the Monte Carlo sum (1/S) sum_j pi(x_ij | Lambda) / p_PE(x_ij) over its S samples. When the
population has a sharp mass edge (the observed-like power law ends at m_max = 87 Msun with a 0.01 Msun taper), a merger
whose posterior lies almost wholly beyond the edge has only a handful of samples inside the support, and ln L_i(H0) steps
as single samples cross the edge. The mixture proposal draws a fraction a_i of the samples from the posterior RESTRICTED to

    A = { x : the population's sharp edges can be satisfied for some H0 in the prior range [H0_lo, H0_hi] },
        upper edge (m_max, taper dm_max sharp):  m1det / (1 + z(dL; H0_hi)) < m_max + EDGE_MARGIN_WIDTHS * dm_max
        lower edge (m_min, taper dm_min sharp):  m2det / (1 + z(dL; H0_lo)) > m_min - EDGE_MARGIN_WIDTHS * dm_min

(source masses are smallest at H0_hi and largest at H0_lo at fixed dL; z(dL; H0) of flat LCDM at the run's Om0; an edge is
sharp when its taper is narrower than SHARP_EDGE_MAX_WIDTH), and 1 - a_i from the posterior itself. The mixture density
relative to the posterior is

    r_i(x) = (1 - a_i) + a_i 1_A(x) / P_A,i,        P_A,i = posterior probability of A (estimated: in-A / raw draws),

and the exact estimator is (1/S) sum_j [pi / p_PE](x_ij) / r_i(x_ij): the correction enters as p_PE -> p_PE * r_i per
sample (the store's p_pe column; core divides every sample by it after normalising it per event, an H0-independent
constant). It is exact for any population (the Gaussian peak's tail beyond m_max is carried by the unrestricted share).

Layout of one merger's S samples (S = nsamp, h = S // 2):
- columns 0..h-1: the production redraw's first h samples (blocks of 2000 with redraw_key + [b]), unchanged;
- a merger all of whose S production samples lie in A is left exactly as the production redraw (P_A,i = 1, a_i = 0,
  r_i = 1: bitwise the off path);
- otherwise a pilot of R_MIN raw draws of the merger's own stream, SeedSequence(redraw_key + [0, 1, i]) (i = the
  merger's index in the events file, so its draws depend neither on the other mergers nor on a subset), estimates P_A;
  with P_A >= P_REFILL the restricted posterior is within 1% of the posterior and the merger is kept as drawn (a_i = 0,
  r_i = 1); otherwise columns h..S-1 are refilled by rejection: the production samples h..S-1 that lie in A first, in
  order, then the in-A draws of the stream (the pilot, then blocks b = 1, 2, ... from redraw_key + [b, 1, i]) until
  S - h are accepted or CAP raw draws of the stream were made; P_A,i = in-A / raw over every posterior draw of the merger
  (its S production samples and the stream's); slots still empty at the cap are filled with unrestricted draws from
  SeedSequence(redraw_key + [0, 2, i]), and a_i = accepted / S (exact either way).
With no sharp edge in the run's population (the peak family: dm_max 10, dm_min 3) A is everything and the redraw is the
production one, untouched.
"""
from __future__ import annotations

import math
import time

import numpy as np

MODES = ("off", "support")
SHARP_EDGE_MAX_WIDTH = 1.0      # Msun: a taper narrower than this is a sharp edge
EDGE_MARGIN_WIDTHS = 5.0        # A reaches this many taper widths beyond the edge (core's filter is 0 at and beyond it)
R_MIN = 200_000                 # pilot: raw draws of the merger's own stream before the refill decision (P_A precision)
P_REFILL = 0.99                 # refill only when the pilot's P_A is below this (else the merger is kept as drawn)
CAP = 200_000_000               # raw posterior draws per merger at most
MAX_BLOCK = 2_000_000           # raw draws per call of the generator
STREAM_RESTRICTED = 1           # redraw_key + [b, 1, i]
STREAM_FILL = 2                 # redraw_key + [0, 2, i]
ZTAB_MAX = 5.0


def mode_of(cfg) -> str:
    m = cfg.get("redraw_mixture") or "off"
    if m not in MODES:
        raise SystemExit(f"[fatal] redraw_mixture {m!r} not in {MODES}")
    return m


def support_region(cfg) -> dict:
    """The region A of the run: edges and tapers of every class population the run uses, the H0 prior range."""
    free = set(cfg["free"])
    pops = [cfg["population_values"]] + list((cfg.get("class_populations") or {}).values())
    lo, hi = (float(x) for x in cfg["priors"]["H0"])
    reg = {"H0_range": [lo, hi], "Om0": float(cfg["Om0"]), "sharp_edge_max_width": SHARP_EDGE_MAX_WIDTH,
           "margin_widths": EDGE_MARGIN_WIDTHS, "upper": None, "lower": None}
    if not ({"m_max", "dm_max", "m_min", "dm_min"} & free):
        m_max = max(float(p["m_max"]) for p in pops)
        dm_max = max(float(p["dm_max"]) for p in pops if float(p["m_max"]) == m_max)
        if dm_max < SHARP_EDGE_MAX_WIDTH:
            reg["upper"] = {"m_max": m_max, "dm_max": dm_max, "H0": hi, "bound": m_max + EDGE_MARGIN_WIDTHS * dm_max,
                            "condition": f"m1det / (1 + z(dL; H0={hi:g})) < {m_max + EDGE_MARGIN_WIDTHS * dm_max:g}"}
        m_min = min(float(p["m_min"]) for p in pops)
        dm_min = max(float(p["dm_min"]) for p in pops if float(p["m_min"]) == m_min)
        if dm_min < SHARP_EDGE_MAX_WIDTH:
            reg["lower"] = {"m_min": m_min, "dm_min": dm_min, "H0": lo, "bound": m_min - EDGE_MARGIN_WIDTHS * dm_min,
                            "condition": f"m2det / (1 + z(dL; H0={lo:g})) > {m_min - EDGE_MARGIN_WIDTHS * dm_min:g}"}
        reg["edges_of_population"] = {"m_max": m_max, "dm_max": dm_max, "m_min": m_min, "dm_min": dm_min}
    else:
        reg["note"] = "a mass edge is sampled: A is everything"
    reg["active"] = reg["upper"] is not None or reg["lower"] is not None
    return reg


_ZTAB = {}


def z_of_dl(dL, H0, Om0):
    """z(dL; H0) of flat LCDM (a 40001-node table of prod_common's comoving distance; clamped at z = ZTAB_MAX)."""
    key = (float(H0), float(Om0))
    if key not in _ZTAB:
        import prod_common as P
        z = np.expm1(np.linspace(0.0, np.log1p(ZTAB_MAX), 40001))
        _ZTAB[key] = (z, (1.0 + z) * P.comoving_distance(z, H0, Om0))
    z, dl = _ZTAB[key]
    return np.interp(dL, dl, z)


def in_region(reg, m1det, m2det, dL):
    ok = np.ones(np.shape(m1det), bool)
    if reg["upper"] is not None:
        u = reg["upper"]
        ok &= m1det / (1.0 + z_of_dl(dL, u["H0"], reg["Om0"])) < u["bound"]
    if reg["lower"] is not None:
        lw = reg["lower"]
        ok &= m2det / (1.0 + z_of_dl(dL, lw["H0"], reg["Om0"])) > lw["bound"]
    return ok


def _draw(gen, obs, i, key, n):
    """n raw posterior draws of merger i from SeedSequence(key) (the generator's own posterior_samples_v3)."""
    o = {k: v[i:i + 1] for k, v in obs.items()}
    rng = np.random.default_rng(np.random.SeedSequence([int(x) for x in key]))
    return gen.posterior_samples_v3(rng, o, int(n))


def mixture_redraw(cfg, gen, obs, index, raw, ns):
    """Refill the restricted share of the samples in place.

    raw: {column: (nrows, >= ns)} production redraw rows of the mergers index (indices into obs, the events file's
    observations); the first ns columns are used and raw is replaced by its (nrows, ns) mixture. Returns (raw, r, rec):
    r (nrows, ns) the mixture density over the posterior per sample, or None when the region is everything (raw is then
    the production redraw, untouched); rec the region and per-merger record (refilled mergers only)."""
    t0 = time.time()
    reg = support_region(cfg)
    rec = {"mode": "support", "region": reg, "nsamp": int(ns), "R_MIN": R_MIN, "P_REFILL": P_REFILL, "CAP": CAP, "MAX_BLOCK": MAX_BLOCK,
           "streams": {"restricted": "redraw_key + [b, 1, i]", "fill": "redraw_key + [0, 2, i]"}}
    index = np.asarray(index, int)
    if not reg["active"]:
        rec["summary"] = {"active": False, "n_mergers": int(index.size), "n_refilled": 0, "n_capped": 0,
                          "seconds": time.time() - t0}
        rec["mergers"] = {}
        return raw, None, rec
    key = list(cfg["redraw_key"])
    cols = list(raw)
    raw = {k: np.array(v[:, :ns], dtype=float, copy=True) for k, v in raw.items()}
    nrows = index.size
    h = ns // 2
    need = ns - h
    inA = in_region(reg, raw["m1det"], raw["m2det"], raw["dL"])
    r = np.ones((nrows, ns))
    touched = np.where(~inA.all(axis=1))[0]
    mergers = {}
    for row in touched:
        tm = time.time()
        i = int(index[row])
        K_prod = int(inA[row].sum())
        cand = {k: raw[k][row, h:][inA[row, h:]] for k in cols}
        taken = [cand]
        n_taken = int(cand["dL"].size)
        D_s = K_s = 0
        b = 0
        P_pilot = None
        while D_s < CAP:
            if b == 0:
                n_b = R_MIN
            else:
                if n_taken >= need:
                    break
                p_hat = max(K_prod + K_s, 1) / (ns + D_s)
                n_b = max(int(math.ceil(1.25 * (need - n_taken) / p_hat)), 2000)
                n_b = int(min(MAX_BLOCK, CAP - D_s, 2000 * math.ceil(n_b / 2000)))
            p = _draw(gen, obs, i, key + [b, STREAM_RESTRICTED, i], n_b)
            ok = in_region(reg, p["m1det"], p["m2det"], p["dL"])
            D_s += n_b
            K_s += int(ok.sum())
            if n_taken < need:
                sel = np.where(ok)[0][: need - n_taken]
                taken.append({k: p[k][sel] for k in cols})
                n_taken += int(sel.size)
            b += 1
            if b == 1:
                P_pilot = K_s / D_s
                if P_pilot >= P_REFILL:
                    break
        D, K = ns + D_s, K_prod + K_s
        P_A = K / D
        common = {"P_A": P_A, "P_A_rel_err": math.sqrt(max(1.0 - P_A, 0.0) / max(P_A * D, 1e-300)),
                  "P_A_pilot": P_pilot, "raw_draws": int(D), "in_A_draws": int(K), "production_in_A": K_prod,
                  "stream_blocks": int(b)}
        if P_pilot >= P_REFILL:            # the restricted posterior is within 1 - P_REFILL of the posterior: kept as drawn
            mergers[str(i)] = {**common, "refilled": False, "restricted": 0, "a": 0.0, "capped": False,
                               "seconds": time.time() - tm}
            continue
        n_r = min(n_taken, need)
        restricted = {k: np.concatenate([t[k] for t in taken])[:n_r] for k in cols}
        capped = n_r < need
        if capped:
            fill = _draw(gen, obs, i, key + [0, STREAM_FILL, i], need - n_r)
            restricted = {k: np.concatenate([restricted[k], fill[k]]) for k in cols}
        for k in cols:
            raw[k][row, h:] = restricted[k]
        a = n_r / ns
        inA_row = in_region(reg, raw["m1det"][row], raw["m2det"][row], raw["dL"][row])
        r[row] = (1.0 - a) + a * inA_row / P_A
        mergers[str(i)] = {**common, "refilled": True, "restricted": int(n_r), "a": a, "capped": bool(capped),
                           "seconds": time.time() - tm}
    rec["mergers"] = mergers
    ref = {k: m for k, m in mergers.items() if m["refilled"]}
    rd = np.array([m["raw_draws"] for m in mergers.values()]) if mergers else np.zeros(0)
    rec["summary"] = {"active": True, "n_mergers": int(nrows), "n_with_production_samples_outside_A": len(mergers),
                      "n_refilled": len(ref), "n_kept_P_A_pilot_ge": [len(mergers) - len(ref), P_REFILL],
                      "n_capped": int(sum(m["capped"] for m in ref.values())),
                      "capped": [int(k) for k, m in ref.items() if m["capped"]],
                      "raw_draws_total": int(rd.sum()), "raw_draws_max": int(rd.max()) if rd.size else 0,
                      "P_A_min": float(min((m["P_A"] for m in mergers.values()), default=1.0)),
                      "seconds": time.time() - t0}
    return raw, r, rec


def corrected_p_pe(p_pe, r, nrows, ns):
    """p_PE -> p_PE * r per sample on the refilled mergers, renormalised to mean 1 there (the column's convention; core
    renormalises per merger); the other mergers' values are returned bitwise unchanged. r None: p_pe unchanged."""
    if r is None:
        return p_pe
    x = np.array(p_pe, dtype=float, copy=True).reshape(nrows, ns)
    rows = np.where((r != 1.0).any(axis=1))[0]
    if rows.size:
        y = x[rows] * r[rows]
        x[rows] = y / y.mean(axis=1, keepdims=True)
    return x.ravel()
