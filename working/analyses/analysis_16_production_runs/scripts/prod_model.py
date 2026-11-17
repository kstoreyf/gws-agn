"""Analysis 16: build the darksirens-core likelihood of one production run (core venv only).

    cfg = prod_driver.resolve(args)        # every option resolved to explicit values
    m = Model(cfg); m.build(); m.bind()    # load stores, redraw the samples, bind, check
    out = m.evaluate({"H0": 68.0, "f_agn": 0.3, ...})

Settings fixed here for every run (owner's rule: nothing left to a core default):
- darksirens-core e7c3007, imported from its worktree (refused otherwise); DARKSIRENS_ZMAX is set
  by the driver before the import and checked here;
- catalog evaluation "historical" (Analyses 13-15): kernel_window 1e-10, kernel_layout padded,
  missing_density grid; population normalisation grids pairing_norm per_sample, pairing_scale
  analytic;
- Cosmology(H0 sampled on the prior range, Om0 0.3075); powerlaw+peak, shared beta/spin/gamma,
  every population value either sampled or fixed at the value in cfg["population_values"];
- catalogs: field sky weighting; per catalog delta = sigma_kde = 0; kernel_pin "auto";
  complete catalogs completeness "complete" (Z_k = N_obs, no missing hosts, no density) unless
  cfg says otherwise (the Analysis 14 closure uses the count ratio at log10 n0 = -24);
  flux-limited catalogs completeness "selection" with the Schechter fit of that limit,
  field_normalizer "moments", n0_units from cfg (h_scaled in production), log10 n0 fixed or
  sampled per cfg;
- no catalog: core's catalog-free (spectral) likelihood, p(z) proportional to dV_c/dz on the
  redshift grid (hosts uniform in comoving volume; the population carries (1+z)^(gamma-1));
- host-dependent populations: per_catalog_population {2: [mu_G, mu_chi]}, the AGN copies passed
  as absolute values mu + dmu (unit Jacobian from the sampled offsets);
- bind_analysis: max_likelihood_variance from cfg (20 in production), sel_batch_size 5000,
  pe_event_block from cfg (5), selection_neff_soft_guard False.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402

P.ensure_paths()


def log(msg):
    P.log("a16", msg)


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ds = None
        self.an = None
        self.b = None
        self.diag = None
        self.labels = None
        self.info = {}

    # ------------------------------------------------------------------ import + settings
    def import_core(self):
        want = float(self.cfg["zmax"])
        env = os.environ.get("DARKSIRENS_ZMAX")
        if env is None or abs(float(env) - want) > 1e-12:
            sys.exit(f"[fatal] DARKSIRENS_ZMAX={env!r} but the run needs the redshift grid to {want}; "
                     "the driver sets it before importing darksirens")
        import darksirens as ds
        from darksirens.cosmology import _grid, distances
        if f"darksirens-core-{P.CORE}" not in ds.__file__:
            sys.exit(f"[fatal] darksirens imported from {ds.__file__}, not core {P.CORE}")
        if abs(float(_grid.zMax) - want) > 1e-12 or abs(float(distances.zMax) - want) > 1e-12:
            sys.exit(f"[fatal] darksirens grid zMax {_grid.zMax} / distance zMax {distances.zMax} != {want}")
        self.ds = ds
        self.info["darksirens_file"] = ds.__file__
        self.info["zmax_grid"] = float(_grid.zMax)
        self.info["grid_nodes"] = int(_grid.zgrid.shape[0])
        self.info["distance_table_nodes"] = int(distances.zgrid.shape[0])
        try:
            import subprocess
            head = subprocess.run(["git", "-C", str(P.CORE_SRC), "rev-parse", "HEAD"], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(P.CORE_SRC), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            head, dirty = f"unknown ({e})", ""
        if not head.startswith(P.CORE):
            sys.exit(f"[fatal] core worktree HEAD {head} is not the pinned {P.CORE}")
        self.info["core_commit"] = head
        self.info["core_worktree_dirty"] = bool(dirty)
        return ds

    def configure_settings(self):
        from darksirens.catalog import settings as cs
        from darksirens.population.utils import configure_normalization_grids
        ev = self.cfg["catalog_evaluation"]
        cs.configure_catalog_evaluation(kernel_window=ev["kernel_window"], kernel_layout=ev["kernel_layout"],
                                        missing_density=ev["missing_density"])
        ng = configure_normalization_grids(pairing_norm=ev["pairing_norm"], pairing_scale=ev["pairing_scale"])
        st = cs.catalog_evaluation_settings()
        settings = {k: str(getattr(st, k)) for k in st.__dataclass_fields__}
        settings.update({"pairing_norm": str(getattr(ng, "pairing_norm", "?")),
                         "pairing_scale": str(getattr(ng, "pairing_scale", "?"))})
        for k in ("kernel_window", "kernel_layout", "missing_density", "pairing_norm", "pairing_scale"):
            if str(settings.get(k)) != str(ev[k]):
                sys.exit(f"[fatal] catalog evaluation {k}={settings.get(k)} after configuring {ev[k]}")
        self.info["catalog_evaluation_in_effect"] = settings

    # ------------------------------------------------------------------ the analysis
    def build(self, catalog_stores=None):
        cfg = self.cfg
        ds = self.ds or self.import_core()
        self.configure_settings()
        lo, hi = cfg["priors"]["H0"]
        cos = ds.Cosmology(H0=(float(lo), float(hi)), Om0=P.OM0)
        free = set(cfg["free"])
        pv = cfg["population_values"]
        fixed_pop = {P.POP_LABELS[k]: float(v) for k, v in pv.items() if k not in free}
        pop = ds.Population("powerlaw+peak", fixed=fixed_pop)
        tracers = P.CATALOG_SETS[cfg["catalogs"]]
        kw = {}
        if tracers:
            stores = []
            for t in tracers:
                path = cfg["surveys"][t]
                st = (catalog_stores or {}).get(path) or ds.load_catalog(path)
                if getattr(st, "z_depth", None) is not None:
                    sys.exit(f"[fatal] {path} carries z_depth {st.z_depth}; production runs use none")
                stores.append(st)
            self.catalog_stores = dict(zip([cfg["surveys"][t] for t in tracers], stores))
            comp = cfg["completeness_model"]
            fixed_survey = {}
            survey_priors = {}
            sel = None
            norm = None
            n0_units = None
            for k, t in enumerate(tracers):
                sfx = "" if k == 0 else f"_c{k + 1}"
                fixed_survey["delta" + sfx] = 0.0
                fixed_survey["sigma_kde" + sfx] = 0.0
                if comp != "complete":
                    pname = "log10n_gal" if t == "gal" else "log10n_agn"
                    if pname in free:
                        survey_priors["log10n0" + sfx] = tuple(float(x) for x in cfg["priors"][pname])
                    else:
                        fixed_survey["log10n0" + sfx] = float(cfg["densities"][t]["log10n_used"])
            if comp == "selection":
                from darksirens.selection.catalog import SchechterMagnitudeSelection
                sel = [SchechterMagnitudeSelection(**{k: v for k, v in cfg["selection_fits"][t]["values"].items()})
                       for t in tracers]
                norm = "moments"
                n0_units = cfg["density_units"]
            elif comp == "count_ratio":
                norm = "direct"
                n0_units = cfg["density_units"]
            elif comp == "complete":
                norm = "direct"
            else:
                sys.exit(f"[fatal] completeness model {comp!r}")
            core_comp = {"complete": "complete", "selection": "selection", "count_ratio": "incomplete"}[comp]
            kw = dict(catalog=stores, catalog_sky_weighting="field", completeness=[core_comp] * len(stores),
                      field_normalizer=[norm] * len(stores), fixed_survey=fixed_survey, kernel_pin="auto")
            if sel is not None:
                kw["selection"] = sel
            if n0_units is not None:
                kw["n0_units"] = n0_units
            if survey_priors:
                kw["survey_priors"] = survey_priors
            if set(P.OFFSETS) & free:
                kw["per_catalog_population"] = {2: [P.POP_LABELS["mu_G"], P.POP_LABELS["mu_chi"]]}
        an = ds.model(cosmology=cos, population=pop, allow_out_of_prior=True, **kw)
        self.an = an
        self.info["core_plan_labels"] = list(an.parameters.labels)
        self.info["core_population_labels"] = list(an.parameters.population_labels)
        missing = set(P.POP_LABELS.values()) ^ set(an.parameters.population_labels)
        if missing:
            sys.exit(f"[fatal] core population labels differ from prod_common.POP_LABELS: {missing}")
        self.info["fixed_population_in_plan"] = {k: float(v) for k, v in an.parameters.fixed_population_values} \
            if an.parameters.fixed_population_values else (
            dict(zip(an.parameters.population_labels, map(float, an.parameters.fixed_population)))
            if an.parameters.fixed_population is not None else {})
        self.info["fixed_survey_in_plan"] = {k: float(v) for k, v in an.parameters.fixed_survey}
        self.info["n0_units"] = an.parameters.n0_units
        self.info["kernel_pin_active"] = bool(an.parameters.kernel_pin_active)
        self.info["catalog_model"] = str(an.parameters.catalog_model)
        return an

    # ------------------------------------------------------------------ data
    def load_events(self):
        """The event store: the file's own samples (nsamp 0) or a redraw at cfg['nsamp'] per event."""
        cfg = self.cfg
        ds = self.ds
        path = cfg["events_file"]
        n_lim = cfg.get("limit_events")
        t0 = time.time()
        if int(cfg["nsamp"]) == 0:
            store = events_file_subset(ds, path, int(n_lim)) if n_lim else ds.load_events(path)
            self.info["events_samples"] = "stored"
        else:
            import mc_redraw as M
            obs, attrs, nobs, _ = M.read_event_set(path)
            if n_lim:
                nobs = int(n_lim)
                obs = {k: v[:nobs] for k, v in obs.items()}
            ns = int(cfg["nsamp"])
            if ns % P.REDRAW_BLOCK:
                sys.exit(f"[fatal] nsamp {ns} is not a multiple of {P.REDRAW_BLOCK}")
            g = M.generator()
            blocks = []
            for b in range(ns // P.REDRAW_BLOCK):
                rng = np.random.default_rng(np.random.SeedSequence(list(cfg["redraw_key"]) + [b]))
                p = g.posterior_samples_v3(rng, obs, P.REDRAW_BLOCK)
                blocks.append({k: v.reshape(nobs, P.REDRAW_BLOCK) for k, v in p.items()})
            post = {k: np.concatenate([bl[k] for bl in blocks], axis=1).ravel() for k in blocks[0]}
            post = M.postprocess(post, nobs, ns)
            store = M.load_store(ds, post, attrs, nobs, ns)
            self.info["events_samples"] = "redrawn"
            del post, blocks
        self.info["events_load_seconds"] = time.time() - t0
        return store

    def load_injections(self):
        inj = self.ds.load_injections(self.cfg["injections_file"])
        n_lim = self.cfg.get("limit_injections")
        if n_lim:
            import dataclasses
            n = int(n_lim)
            cols = {k: np.asarray(v)[:n] for k, v in inj.columns.items()}
            inj = dataclasses.replace(inj, columns=cols, n_injections=n, prior_wt=np.asarray(inj.prior_wt)[:n])
        return inj

    def bind(self, events_store=None, inj_store=None):
        from darksirens.runtime_binding import bind_analysis
        cfg = self.cfg
        t0 = time.time()
        ev = events_store if events_store is not None else self.load_events()
        inj = inj_store if inj_store is not None else self.load_injections()
        self.events_store, self.inj_store = ev, inj
        b = bind_analysis(self.an, events=ev, injections=inj, max_likelihood_variance=float(cfg["max_var"]),
                          sel_batch_size=int(cfg["sel_batch"]), pe_event_block=int(cfg["pe_event_block"]),
                          selection_neff_soft_guard=False)
        self.b = b
        self.labels = list(b.labels)
        self.n_events = int(b.n_events)
        self.nsamp = int(b.nsamp)
        self.n_draw = float(b.n_draw)
        self.info.update({"n_events": self.n_events, "nsamp": self.nsamp, "n_draw": self.n_draw,
                          "n_detected_injections": int(inj.n_injections), "core_labels": self.labels,
                          "bind_seconds": time.time() - t0})
        expect = sorted(self.core_labels_expected())
        if sorted(self.labels) != expect:
            sys.exit(f"[fatal] core labels {self.labels}, expected {expect}")
        self.diag = make_diag(self.an, self.n_events, self.nsamp, self.n_draw, int(cfg["pe_event_block"]),
                              int(cfg["sel_batch"]), float(cfg["max_var"]))
        self.ops = (b.gw_pe, b.gw_selection) if b.model_operands is None else \
            (b.gw_pe, b.gw_selection, b.model_operands)
        return b

    # ------------------------------------------------------------------ coordinates
    def core_labels_expected(self):
        cfg = self.cfg
        free = cfg["free"]
        tracers = P.CATALOG_SETS[cfg["catalogs"]]
        out = []
        for p in free:
            if p == "H0":
                out.append("H0")
            elif p == "f_agn":
                out.append("fcat_2")
            elif p in P.POP_LABELS:
                out.append(P.POP_LABELS[p])
            elif p == "dmu_G":
                out.append(P.POP_LABELS["mu_G"] + "_c2")
            elif p == "dmu_chi":
                out.append(P.POP_LABELS["mu_chi"] + "_c2")
            elif p in ("log10n_gal", "log10n_agn"):
                t = "gal" if p == "log10n_gal" else "agn"
                k = tracers.index(t)
                out.append("log10n0" + ("" if k == 0 else f"_c{k + 1}"))
        return out

    def core_values(self, coords):
        """Sampler coordinates (ASCII names) -> {core label: value}; AGN copies made absolute."""
        tracers = P.CATALOG_SETS[self.cfg["catalogs"]]
        v = {}
        for p, x in coords.items():
            if p == "H0":
                v["H0"] = x
            elif p == "f_agn":
                v["fcat_2"] = x
            elif p in P.POP_LABELS:
                v[P.POP_LABELS[p]] = x
            elif p in ("log10n_gal", "log10n_agn"):
                k = tracers.index("gal" if p == "log10n_gal" else "agn")
                v["log10n0" + ("" if k == 0 else f"_c{k + 1}")] = x
        if "dmu_G" in coords:
            base = coords.get("mu_G", self.cfg["population_values"]["mu_G"])
            v[P.POP_LABELS["mu_G"] + "_c2"] = base + coords["dmu_G"]
        if "dmu_chi" in coords:
            base = coords.get("mu_chi", self.cfg["population_values"]["mu_chi"])
            v[P.POP_LABELS["mu_chi"] + "_c2"] = base + coords["dmu_chi"]
        return v

    def theta(self, coords):
        import jax.numpy as jnp
        v = self.core_values(coords)
        miss = [l for l in self.labels if l not in v]
        if miss:
            raise KeyError(f"no value for core label(s) {miss} from coordinates {sorted(coords)}")
        return jnp.asarray([float(v[l]) for l in self.labels], dtype=jnp.float64)

    def evaluate(self, coords):
        """dict: lnL (core's, guard included), pe_var = sum_i sigma_i^2, n_eff, log_mu, sel_corr,
        lnL_corrected (half-variance correction), sigma2_total = pe_var + N^2/N_eff."""
        r = self.diag(self.theta(coords), *self.ops)
        lnL, pv, ne, lmu, sc = (float(x) for x in r[:5])
        corr = lnL + 0.5 * pv - 0.5 * self.n_events / ne if np.isfinite(lnL) else lnL
        return {"lnL": lnL, "pe_var": pv, "n_eff": ne, "log_mu": lmu, "sel_corr": sc,
                "lnL_corrected": float(corr), "sigma2_total": pv + self.n_events ** 2 / ne if ne > 0 else np.inf}

    def bound_call(self, coords):
        return float(self.b(self.theta(coords)))

    def check_diag_equals_bound(self, points):
        rows = []
        for c in points:
            vb = self.bound_call(c)
            vd = self.evaluate(c)["lnL"]
            rows.append({"coords": c, "bound": vb, "diag": vd, "abs_diff": abs(vb - vd) if np.isfinite(vb) else None})
            ok = (vb == vd) if not np.isfinite(vb) else abs(vd - vb) <= 1e-8 * max(1.0, abs(vb))
            log(f"diag vs bound at {c}: {vd:.10f} vs {vb:.10f}")
            if not ok:
                sys.exit(f"[fatal] the diagnostics program differs from the bound call at {c}: {vd} vs {vb}")
        return rows


def make_diag(an, n_events, nsamp, n_draw, peb, sel_batch, maxvar):
    """The bound call's program with return_diagnostics=True (Analyses 14/15), spectral or field."""
    import jax
    import jax.numpy as jnp
    from darksirens.analysis import FieldCatalogMixtureRedshift, SpectralRedshift
    from darksirens.cosmology.distances import threads_distance_table
    from darksirens.runtime_binding import _decode_theta
    pop = an.population
    common = dict(pop_model=pop.model_name, shared_beta=pop.shared_beta, shared_spin=pop.shared_spin,
                  shared_gamma=pop.shared_gamma, sel_batch_size=sel_batch, selection_neff_soft_guard=False,
                  max_likelihood_variance=maxvar, pe_event_block=peb, angular_model=an.angular_model,
                  return_diagnostics=True)
    if isinstance(an.redshift, SpectralRedshift):
        from darksirens.likelihood.hierarchical import spectral_siren_log_likelihood

        @threads_distance_table()
        def diag(theta, gw_pe, gw_sel, distance_table=None):
            cosmology, population, _, angular = _decode_theta(an, theta, z_depth=None)
            d = spectral_siren_log_likelihood(cosmology, population, gw_pe, gw_sel, n_events, nsamp, n_draw,
                                              angular_params=angular, **common)
            return (d.log_likelihood, jnp.sum(d.event_mc_variance), d.n_eff, d.log_mu,
                    d.selection_log_correction, d.event_log_evidence, d.event_mc_variance)
    elif isinstance(an.redshift, FieldCatalogMixtureRedshift):
        from darksirens.likelihood.mixture import field_mixture_log_likelihood

        @threads_distance_table()
        def diag(theta, gw_pe, gw_sel, model_operands, distance_table=None):
            cosmology, population, catalog_params, angular = _decode_theta(an, theta, z_depth=None)
            d = field_mixture_log_likelihood(
                cosmology, population, catalog_params, gw_pe, gw_sel, model_operands,
                n_events, nsamp, n_draw, completeness=an.redshift.completeness,
                normalizer=an.redshift.normalizer, angular_params=angular, **common)
            return (d.log_likelihood, jnp.sum(d.event_mc_variance), d.n_eff, d.log_mu,
                    d.selection_log_correction, d.event_log_evidence, d.event_mc_variance)
    else:
        raise TypeError(f"unsupported redshift model {type(an.redshift).__name__}")
    return jax.jit(diag)


EV_COLUMNS = ("ra", "dec", "dL", "m1det", "m2det", "chieff", "p_pe", "m1src", "m2src")


def events_file_subset(ds, path, n_ev):
    """The first n_ev events of an events file (stored samples) as an in-memory store (CPU checks;
    Analysis 15's events_subset)."""
    import h5py
    bio = io.BytesIO()
    with h5py.File(path, "r") as src, h5py.File(bio, "w") as dst:
        nsamp = int(src.attrs["nsamp"])
        for k, v in src.attrs.items():
            if k != "metadata_json":
                dst.attrs[k] = v
        dst.attrs["nobs"] = int(n_ev)
        for k in EV_COLUMNS:
            dst.create_dataset(k, data=np.asarray(src[k][: n_ev * nsamp], dtype=np.float64))
    bio.seek(0)
    return ds.load_events(bio)
