"""Analysis 14: redraw the per-event posterior samples of a stored v3 event set.

Each stored event's samples are the exact flat-prior posterior of ONE recorded measurement
(working/data/README.md section 2 and convention (c); DESIGN_PE.md 2.4-2.6). A redraw keeps
the recorded observation (``truth/obs_*``) and draws new samples from the same posterior with
a new random stream, through the generator's OWN functions, imported here:

    generate_dataset.posterior_samples_v3   the exact posterior draws (all six channels)
    generate_dataset.p_pe_v3                the PE-prior column, rho/(dL m1det q), mean-1 per event
    gmd._build_cosmology/_cosmology_grids   the z(dL) table the generator uses for m1src/m2src

exactly as ``stage_events`` post-processes them. Nothing about the measurement model is
re-implemented here.

Common random numbers: samples are drawn in blocks of 2000 per event, block b of redraw r
from ``SeedSequence([ROOT, set_code, r, b])``. Redraw r at N_s = 8000 is blocks 0..3, so its
first 2000 samples per event ARE redraw r at N_s = 2000; 32000 likewise contains 8000.
The only N_s-dependent step is the per-event mean-1 normalisation of p_pe, which darksirens
re-normalises per event anyway (only the shape enters).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
GMD_WORKTREE = Path("/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d")  # META.json's SHA
BLOCK = 2000
ROOT = 14_100_2026          # Analysis 14's stream root (any fixed integer)
SET_CODES = {"pureagn": 1, "mixed": 2}
OBS_KEYS = ("rho", "lnmc", "lnq", "chieff", "dec", "ra",
            "sig_lnmc", "sig_lnq", "sig_chieff", "sigma_ang", "sig_ra", "sigma_rho")
STORE_COLUMNS = ("ra", "dec", "dL", "m1det", "m2det", "chieff", "p_pe", "m1src", "m2src")

_GEN = None
_GRIDS = None


def generator():
    """The dataset generator module (imported once)."""
    global _GEN
    if _GEN is None:
        if str(DATA) not in sys.path:
            sys.path.insert(0, str(DATA))
        import generate_dataset as g  # noqa: E402
        _GEN = g
    return _GEN


def zdl_grids():
    """The generator's own cosmology grid (stage_events: gmd._cosmology_grids at the fiducial)."""
    global _GRIDS
    if _GRIDS is None:
        g = generator()
        gmd = g.import_gmd(GMD_WORKTREE)
        cosmo = gmd._build_cosmology(g.H0_FID, g.OM0_FID, g.W0_FID, g.WA_FID)
        _GRIDS = gmd._cosmology_grids(cosmo, g.ZMAX_GRID)
    return _GRIDS


def read_event_set(path):
    """The recorded observations and the scalar attributes of a stored events file."""
    import h5py
    with h5py.File(path, "r") as f:
        t = f["truth"]
        obs = {k: np.asarray(t[f"obs_{k}"], dtype=float) for k in OBS_KEYS}
        attrs = {k: f.attrs[k] for k in f.attrs}
        nobs, nsamp = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
    return obs, attrs, nobs, nsamp


def read_stored_samples(path, keys=STORE_COLUMNS + ("mc_det", "q", "rho")):
    import h5py
    with h5py.File(path, "r") as f:
        nobs, nsamp = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
        return {k: np.asarray(f[k]).reshape(nobs, nsamp) for k in keys}


def block_rng(set_code: int, r: int, b: int, seed: int = 100):
    """Seed 100 keeps its original entropy [ROOT, set, r, b]; other catalogs append the seed."""
    key = [ROOT, int(set_code), int(r), int(b)] + ([] if int(seed) == 100 else [int(seed)])
    return np.random.default_rng(np.random.SeedSequence(key))


def postprocess(post, nobs, nsamp):
    """stage_events' post-processing, verbatim: m1src/m2src from z(dL) and p_pe (mean-1)."""
    g = generator()
    grids = zdl_grids()
    z_pe = np.interp(post["dL"], grids["dl"], grids["z"])
    post["m1src"] = post["m1det"] / (1.0 + z_pe)
    post["m2src"] = post["m2det"] / (1.0 + z_pe)
    raw = g.p_pe_v3(post["m1det"], post["q"], post["dL"], mc_det=post["mc_det"])
    raw_e = raw.reshape(nobs, nsamp)
    post["p_pe"] = (raw_e / raw_e.mean(axis=1, keepdims=True)).ravel()
    return post


def redraw(obs, nsamp: int, set_code: int, r: int, events=None, seed: int = 100):
    """N_s = nsamp new samples per event (flat arrays, event-major), CRN-nested in blocks."""
    g = generator()
    if nsamp % BLOCK:
        raise ValueError(f"nsamp must be a multiple of {BLOCK}")
    nobs_all = obs["rho"].shape[0]
    sel = np.arange(nobs_all) if events is None else np.asarray(events)
    o = {k: v[sel] for k, v in obs.items()}
    nobs = sel.size
    blocks = []
    for b in range(nsamp // BLOCK):
        p = g.posterior_samples_v3(block_rng(set_code, r, b, seed), o, BLOCK)
        blocks.append({k: v.reshape(nobs, BLOCK) for k, v in p.items()})
    post = {k: np.concatenate([bl[k] for bl in blocks], axis=1).ravel() for k in blocks[0]}
    return postprocess(post, nobs, nsamp)


def store_bytes(cols, attrs, nobs, nsamp):
    """An in-memory gwcat-1.0 HDF5 file (never written to disk) with the stored file's attrs."""
    import h5py
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        for k, v in attrs.items():
            if k == "metadata_json":
                continue
            f.attrs[k] = v
        f.attrs["nobs"] = int(nobs)
        f.attrs["nsamp"] = int(nsamp)
        f.attrs["a14_redraw"] = True
        for k in STORE_COLUMNS:
            f.create_dataset(k, data=np.asarray(cols[k], dtype=np.float64))
    bio.seek(0)
    return bio


def load_store(ds, cols, attrs, nobs, nsamp):
    """darksirens' own loader (its file contract and per-event p_pe normalisation) on memory."""
    return ds.load_events(store_bytes(cols, attrs, nobs, nsamp))
