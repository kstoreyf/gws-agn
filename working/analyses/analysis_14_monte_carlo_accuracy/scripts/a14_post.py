"""Grid posterior summaries shared by the Analysis 14 driver and its aggregation."""
import numpy as np

QS = (0.05, 0.16, 0.5, 0.84, 0.95)


def _cdf(x, p):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return c / c[-1]


def summarize_1d(x, logp):
    """Median, 68% and 90% equal-tailed intervals of a gridded 1-D log density (trapezoid CDF)."""
    x = np.asarray(x, float)
    logp = np.asarray(logp, float)
    ok = np.isfinite(logp)
    if x.size < 2 or ok.sum() < 2:
        nan = float("nan")
        return {"median": nan, "ci68": [nan, nan], "ci90": [nan, nan], "hw68": nan, "w68": nan,
                "w90": nan, "mean": nan, "sd": nan, "map": nan, "mass_at_edges": [nan, nan]}
    p = np.where(ok, np.exp(logp - np.max(logp[ok])), 0.0)
    c = _cdf(x, p)
    q = np.interp(QS, c, x)
    norm = np.trapz(p, x)
    mean = float(np.trapz(p * x, x) / norm)
    sd = float(np.sqrt(max(np.trapz(p * (x - mean) ** 2, x) / norm, 0.0)))
    return {"median": float(q[2]), "ci68": [float(q[1]), float(q[3])], "ci90": [float(q[0]), float(q[4])],
            "hw68": float(0.5 * (q[3] - q[1])), "w68": float(q[3] - q[1]), "w90": float(q[4] - q[0]),
            "mean": mean, "sd": sd, "map": float(x[int(np.nanargmax(np.where(ok, logp, -np.inf)))]),
            "mass_at_edges": [float(p[0] / p.max()), float(p[-1] / p.max())]}


def marginals_2d(h, f, lnL):
    """Log marginals of a gridded 2-D log density lnL[i_h, j_f] (trapezoid over the other axis)."""
    lnL = np.asarray(lnL, float)
    m = np.nanmax(lnL)
    p = np.where(np.isfinite(lnL), np.exp(lnL - m), 0.0)
    ph = np.trapz(p, f, axis=1)
    pf = np.trapz(p, h, axis=0)
    with np.errstate(divide="ignore"):
        return np.log(ph), np.log(pf)


def summarize_2d(h, f, lnL):
    lh, lf = marginals_2d(h, f, lnL)
    sh, sf = summarize_1d(h, lh), summarize_1d(f, lf)
    lnL = np.asarray(lnL, float)
    p = np.exp(lnL - np.nanmax(lnL))
    p /= p.sum()
    H, F = np.meshgrid(h, f, indexing="ij")
    eh, ef = (p * H).sum(), (p * F).sum()
    cov = (p * (H - eh) * (F - ef)).sum()
    rho = cov / np.sqrt((p * (H - eh) ** 2).sum() * (p * (F - ef) ** 2).sum())
    return {"H0": sh, "f": sf, "rho": float(rho)}
