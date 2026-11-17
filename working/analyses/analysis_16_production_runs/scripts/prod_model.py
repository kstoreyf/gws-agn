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
            if self.per_catalog_active():
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

    def per_catalog_active(self):
        """The AGN catalog (catalog 2) carries its own copies of mu_G, mu_chi: sampled offsets
        (host_dependent_populations) or offsets fixed at given values (two_tracers_two_populations)."""
        return bool(set(P.OFFSETS) & set(self.cfg["free"])) or bool(self.cfg.get("fixed_offsets"))

    # ------------------------------------------------------------------ data
    def load_events(self, select=None):
        """The event store: the file's own samples (nsamp 0) or a redraw at cfg['nsamp'] per event.

        select (boolean mask over the file's first n events, n = limit_events or all): keep those
        events only. The redraw is always made for every event with the set's own key and then
        subset, so an event's samples are the same in every run on its set (hosts_known's subsets
        carry exactly the samples of the full-set runs)."""
        cfg = self.cfg
        ds = self.ds
        path = cfg["events_file"]
        n_lim = cfg.get("limit_events")
        t0 = time.time()
        if int(cfg["nsamp"]) == 0:
            if select is None:
                store = events_file_subset(ds, path, int(n_lim)) if n_lim else ds.load_events(path)
            else:
                store = events_file_subset(ds, path, int(n_lim) if n_lim else None, select=select)
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
            if select is not None:
                select = np.asarray(select, bool)
                if select.shape != (nobs,):
                    sys.exit(f"[fatal] event selection of shape {select.shape} for {nobs} events")
                post = {k: v.reshape(nobs, ns)[select].ravel() for k, v in post.items()}
                nobs = int(select.sum())
            store = M.load_store(ds, post, attrs, nobs, ns)
            self.info["events_samples"] = "redrawn"
            del post, blocks
        self.info["events_load_seconds"] = time.time() - t0
        return store

    def load_injections(self):
        """The run's injection store: the realization's lane, or several of its injection sets pooled
        (cfg["injections_files"]): every set is drawn from the same proposal, so the pool is one set
        of sum_k ndraw_k draws whose detected rows are the union (mu = sum_k sum_j w_kj / sum_k ndraw_k)."""
        files = self.cfg.get("injections_files") or [self.cfg["injections_file"]]
        stores = [self._load_one_injection_set(f) for f in files]
        if len(stores) == 1:
            return stores[0]
        import dataclasses
        keys = list(stores[0].columns)
        for st in stores[1:]:
            if list(st.columns) != keys or st.fit_columns != stores[0].fit_columns:
                sys.exit(f"[fatal] injection sets to pool differ in columns: {st.path}")
        cols = {k: np.concatenate([np.asarray(st.columns[k]) for st in stores]) for k in keys}
        pooled = dataclasses.replace(
            stores[0], columns=cols, n_injections=int(sum(st.n_injections for st in stores)),
            ndraw=int(sum(st.ndraw for st in stores)),
            prior_wt=np.concatenate([np.asarray(st.prior_wt) for st in stores]),
            path=";".join(st.path for st in stores))
        self.info["injection_pool"] = [{"path": st.path, "ndraw": int(st.ndraw), "n_detected": int(st.n_injections)}
                                       for st in stores]
        return pooled

    def _load_one_injection_set(self, path):
        inj = self.ds.load_injections(path)
        n_lim = self.cfg.get("limit_injections")
        if n_lim:
            import dataclasses
            n = int(n_lim)
            cols = {k: np.asarray(v)[:n] for k, v in inj.columns.items()}
            inj = dataclasses.replace(inj, columns=cols, n_injections=n, prior_wt=np.asarray(inj.prior_wt)[:n])
        return inj

    def bind(self, events_store=None, inj_store=None, select=None):
        from darksirens.runtime_binding import bind_analysis
        cfg = self.cfg
        t0 = time.time()
        ev = events_store if events_store is not None else self.load_events(select=select)
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
        if cfg.get("fixed_offsets"):
            # the AGN catalog's own mu_G, mu_chi: core coordinates held at base + the given offsets
            out += [P.POP_LABELS["mu_G"] + "_c2", P.POP_LABELS["mu_chi"] + "_c2"]
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
        fo = self.cfg.get("fixed_offsets") or {}
        dmu_G = coords["dmu_G"] if "dmu_G" in coords else fo.get("dmu_G")
        dmu_chi = coords["dmu_chi"] if "dmu_chi" in coords else fo.get("dmu_chi")
        if dmu_G is not None:
            base = coords.get("mu_G", self.cfg["population_values"]["mu_G"])
            v[P.POP_LABELS["mu_G"] + "_c2"] = base + dmu_G
        if dmu_chi is not None:
            base = coords.get("mu_chi", self.cfg["population_values"]["mu_chi"])
            v[P.POP_LABELS["mu_chi"] + "_c2"] = base + dmu_chi
        return v

    def theta(self, coords):
        import jax.numpy as jnp
        v = self.core_values(coords)
        miss = [l for l in self.labels if l not in v]
        if miss:
            raise KeyError(f"no value for core label(s) {miss} from coordinates {sorted(coords)}")
        return jnp.asarray([float(v[l]) for l in self.labels], dtype=jnp.float64)

    extra_keys = ()          # per-cell quantities beyond the five every model stores

    def evaluate(self, coords):
        """dict: lnL (core's, guard included), pe_var = sum_i sigma_i^2, n_eff, log_mu, sel_corr,
        lnL_corrected (half-variance correction), sigma2_total = pe_var + N^2/N_eff."""
        r = self.diag(self.theta(coords), *self.ops)
        return self._pack(r)

    def _pack(self, r):
        lnL, pv, ne, lmu, sc = (float(x) for x in r[:5])
        corr = lnL + 0.5 * pv - 0.5 * self.n_events / ne if np.isfinite(lnL) else lnL
        return {"lnL": lnL, "pe_var": pv, "n_eff": ne, "log_mu": lmu, "sel_corr": sc,
                "lnL_corrected": float(corr), "sigma2_total": pv + self.n_events ** 2 / ne if ne > 0 else np.inf}

    def evaluate_events(self, coords):
        """evaluate() plus the per-event ln evidence and Monte Carlo variance (numpy arrays)."""
        r = self.diag(self.theta(coords), *self.ops)
        out = self._pack(r)
        out["event_log_evidence"] = np.asarray(r[5], dtype=float)
        out["event_mc_variance"] = np.asarray(r[6], dtype=float)
        return out

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


def events_file_subset(ds, path, n_ev, select=None):
    """The first n_ev events of an events file (stored samples) as an in-memory store (CPU checks;
    Analysis 15's events_subset); select (mask over those n_ev, or over all events when n_ev is
    None) keeps a subset of them."""
    import h5py
    bio = io.BytesIO()
    with h5py.File(path, "r") as src, h5py.File(bio, "w") as dst:
        nsamp = int(src.attrs["nsamp"])
        n_all = int(src.attrs["nobs"]) if n_ev is None else int(n_ev)
        for k, v in src.attrs.items():
            if k != "metadata_json":
                dst.attrs[k] = v
        keep = np.ones(n_all, bool) if select is None else np.asarray(select, bool)
        if keep.shape != (n_all,):
            sys.exit(f"[fatal] event selection of shape {keep.shape} for {n_all} events")
        dst.attrs["nobs"] = int(keep.sum())
        for k in EV_COLUMNS:
            a = np.asarray(src[k][: n_all * nsamp], dtype=np.float64).reshape(n_all, nsamp)
            dst.create_dataset(k, data=a[keep].ravel())
    bio.seek(0)
    return ds.load_events(bio)


def event_host_types(path, n_lim=None):
    """True host class of every event (1 = AGN, 0 = galaxy), from the events file."""
    import h5py
    with h5py.File(path, "r") as f:
        ht = np.asarray(f["truth/host_type"] if "truth/host_type" in f else f["host_type"], dtype=int)
    return ht[: int(n_lim)] if n_lim else ht


# ============================================================================ composite models
class HostsKnownModel:
    """hosts_known: the true host class of every merger is given.

    AGN-hosted mergers are analysed with the AGN catalog and the AGN-class population, galaxy-hosted
    mergers with the galaxy catalog and the galaxy-class population; each subset is its own core
    analysis (Model) bound to that subset's events and to the realization's full injection set, so
    its selection term is N_c ln mu_c(theta), with mu_c the detectable fraction of mergers of class c
    (class-c population x class-c catalog's redshift distribution) and N_c the subset's size. The
    likelihoods add (the H0 posteriors multiply):
        ln L(H0) = sum_c [ sum_{i in c} ln L_i,c(H0) - N_c ln mu_c(H0) ].
    This is the two-class mixture likelihood conditioned on the labels and on the number of mergers
    of each class: f_AGN drops out (it would enter only through the binomial factor of N_c, which
    conditioning on N_c removes). The half-variance correction is the sum of the subsets':
    sum_c [ sum_i sigma_i^2 / 2 - N_c / (2 N_eff,c) ], i.e. the per-cell n_eff stored is the
    equivalent N / sum_c (N_c / N_eff,c) and pe_var the sum, so the driver's correction applies."""

    extra_keys = ("lnL_agn", "lnL_gal", "pe_var_agn", "pe_var_gal", "n_eff_agn", "n_eff_gal",
                  "log_mu_agn", "log_mu_gal", "sigma2_sel")

    def __init__(self, cfg):
        self.cfg = cfg
        self.subs = {}
        self.info = {}
        self.labels = None

    def import_core(self):
        self.ds = Model(self.cfg).import_core()
        return self.ds

    def sub_cfg(self, cls):
        """A subset's own run configuration. The AGN-hosted subset may carry its own Monte Carlo settings
        (cfg["agn_subset"]: samples per merger and pooled injection sets; Analysis 14)."""
        cfg = dict(self.cfg)
        t = "agn" if cls == "agn" else "gal"
        cfg["catalogs"] = "agn" if cls == "agn" else "galaxy"
        cfg["tracers"] = [t]
        cfg["surveys"] = {t: self.cfg["surveys"][t]}
        cfg["population_values"] = dict(self.cfg["class_populations"][cls])
        cfg["population_fixed"] = {k: v for k, v in cfg["population_values"].items() if k not in cfg["free"]}
        cfg["fixed_offsets"] = None
        sub = self.cfg.get("agn_subset") if cls == "agn" else None
        if sub:
            cfg["nsamp"] = int(sub["nsamp"])
            cfg["injections_files"] = list(sub["injections_files"])
        return cfg

    def build(self):
        ht = event_host_types(self.cfg["events_file"], self.cfg.get("limit_events"))
        self.host_types = ht
        self.masks = {"agn": ht == 1, "gal": ht == 0}
        stores = {}
        for cls in ("agn", "gal"):
            if not self.masks[cls].any():
                continue
            m = Model(self.sub_cfg(cls))
            m.ds = self.ds
            m.build(catalog_stores=stores)
            stores.update(m.catalog_stores)
            self.subs[cls] = m
        self.info["n_events_by_class"] = {c: int(v.sum()) for c, v in self.masks.items()}
        self.info["subsets"] = {c: {"catalogs": m.cfg["catalogs"], "population": m.cfg["population_values"],
                                    "core_plan_labels": m.info.get("core_plan_labels")}
                                for c, m in self.subs.items()}

    def bind(self):
        injs = {}
        events_all = None
        for cls, m in self.subs.items():
            key = tuple(m.cfg.get("injections_files") or [m.cfg["injections_file"]])
            if key not in injs:
                injs[key] = m.load_injections()
            inj = injs[key]
            # one redraw of the whole set at the largest sample count, subset per class and cut to the
            # subset's count (the redraw is CRN-nested in blocks of 2000, so a subset's first N samples are
            # exactly an N-sample run's: identical to the full-set runs on the set)
            if events_all is None and int(self.cfg["nsamp"]) != 0:
                events_all = self._redraw_all(max(int(sm.cfg["nsamp"]) for sm in self.subs.values()))
            if events_all is not None:
                store = self._subset_store(m, events_all, self.masks[cls], int(m.cfg["nsamp"]))
                m.bind(events_store=store, inj_store=inj)
            else:
                m.bind(inj_store=inj, select=self.masks[cls])
            if m.n_events != int(self.masks[cls].sum()):
                sys.exit(f"[fatal] {cls} subset bound {m.n_events} events, expected {int(self.masks[cls].sum())}")
        self.n_events = sum(m.n_events for m in self.subs.values())
        any_m = next(iter(self.subs.values()))
        self.nsamp, self.n_draw = any_m.nsamp, any_m.n_draw
        self.labels = ["H0"]
        for cls, m in self.subs.items():
            self.info[f"{cls}_model"] = m.info

    def _redraw_all(self, ns):
        """The raw redraw (before post-processing) of every event at ns samples: {column: (nobs, ns)}."""
        import mc_redraw as M
        cfg = self.cfg
        obs, attrs, nobs, _ = M.read_event_set(cfg["events_file"])
        n_lim = cfg.get("limit_events")
        if n_lim:
            nobs = int(n_lim)
            obs = {k: v[:nobs] for k, v in obs.items()}
        if ns % P.REDRAW_BLOCK:
            sys.exit(f"[fatal] nsamp {ns} is not a multiple of {P.REDRAW_BLOCK}")
        g = M.generator()
        blocks = []
        for b in range(ns // P.REDRAW_BLOCK):
            rng = np.random.default_rng(np.random.SeedSequence(list(cfg["redraw_key"]) + [b]))
            p = g.posterior_samples_v3(rng, obs, P.REDRAW_BLOCK)
            blocks.append({k: v.reshape(nobs, P.REDRAW_BLOCK) for k, v in p.items()})
        raw = {k: np.concatenate([bl[k] for bl in blocks], axis=1) for k in blocks[0]}
        return raw, attrs, nobs, ns

    def _subset_store(self, m, events_all, mask, ns):
        """The subset's store at ns samples: its events' first ns redrawn samples, post-processed (per
        event, so exactly as a full-set ns-sample run)."""
        import mc_redraw as M
        raw, attrs, nobs, ns_max = events_all
        n_sub = int(mask.sum())
        post = {k: np.ascontiguousarray(v[mask, :ns]).ravel() for k, v in raw.items()}
        post = M.postprocess(post, n_sub, ns)
        sub = {k: v for k, v in post.items() if k in M.STORE_COLUMNS}
        m.info["events_samples"] = f"redrawn (whole set at {ns_max}, then subset at {ns})"
        return M.load_store(self.ds, sub, attrs, n_sub, ns)

    def evaluate(self, coords):
        rs = {cls: m.evaluate(coords) for cls, m in self.subs.items()}
        out = {"lnL": 0.0, "pe_var": 0.0, "sel_corr": 0.0, "sigma2_sel": 0.0}
        inv = 0.0
        lmu = 0.0
        for cls in ("agn", "gal"):
            r = rs.get(cls)
            for k in ("lnL", "pe_var", "n_eff", "log_mu"):
                out[f"{k}_{cls}"] = r[k] if r else float("nan")
            if r is None:
                continue
            nc = self.subs[cls].n_events
            out["lnL"] += r["lnL"]
            out["pe_var"] += r["pe_var"]
            out["sel_corr"] += r["sel_corr"]
            out["sigma2_sel"] += nc ** 2 / r["n_eff"] if r["n_eff"] > 0 else np.inf
            inv += nc / r["n_eff"] if r["n_eff"] > 0 else np.inf
            lmu += nc / self.n_events * r["log_mu"]
        out["n_eff"] = self.n_events / inv if inv > 0 else float("nan")
        out["log_mu"] = lmu          # N_c-weighted mean of the subsets' ln mu (reference only)
        lnL = out["lnL"]
        out["lnL_corrected"] = (sum(r["lnL_corrected"] for r in rs.values()) if np.isfinite(lnL) else lnL)
        out["sigma2_total"] = out["pe_var"] + out["sigma2_sel"]
        return out

    def check_diag_equals_bound(self, points):
        rows = []
        for cls, m in self.subs.items():
            for r in m.check_diag_equals_bound(points):
                r["subset"] = cls
                rows.append(r)
        return rows


class SpectralMixtureModel:
    """mass_spectrum_alone where the two classes' populations differ: no catalog; every merger is
    from the galaxy-class population with probability 1 - f_AGN and from the AGN-class population with
    probability f_AGN, hosts uniform in comoving volume for both:

        ln L(H0, f) = sum_i ln[ (1-f) L_i,gal + f L_i,agn ] - N ln[ (1-f) mu_gal + f mu_agn ].

    Evaluated with core's own reduction (likelihood.hierarchical._ordinary_hierarchical_likelihood
    with branch_populations): the per-sample weight of a PE sample or an injection is
    logsumexp_k [ ln w_k + ln p_pop(theta | Lambda_k) + ln p_vol(z) ] - ln pi, with core's
    log_comoving_volume_prior (the catalog-free branch's prior), so the per-event Monte Carlo
    variance and the selection N_eff are those of the mixture itself and the variance guard and the
    half-variance correction apply unchanged. At f = 0 (f = 1) it is core's spectral likelihood with
    the galaxy-class (AGN-class) population; every run checks both against core's bound call."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.info = {}
        self.labels = None

    def import_core(self):
        self.ds = Model(self.cfg).import_core()
        return self.ds

    def build(self):
        self.base = {}
        for cls in ("gal", "agn"):
            c = dict(self.cfg)
            c["catalogs"] = "none"
            c["tracers"] = []
            c["surveys"] = {}
            c["free"] = ["H0"]
            c["population_values"] = dict(self.cfg["class_populations"][cls])
            c["fixed_offsets"] = None
            m = Model(c)
            m.ds = self.ds
            m.build()
            self.base[cls] = m
        self.info["core_plan_labels"] = self.base["gal"].info["core_plan_labels"]

    def bind(self):
        mg = self.base["gal"]
        mg.bind()
        self.base["agn"].bind(events_store=mg.events_store, inj_store=mg.inj_store)
        self.n_events, self.nsamp, self.n_draw = mg.n_events, mg.nsamp, mg.n_draw
        self.labels = ["H0", "fcat_2"]
        self.info.update({"n_events": self.n_events, "nsamp": self.nsamp, "n_draw": self.n_draw,
                          "gal_model": mg.info, "agn_model": self.base["agn"].info})
        self._make_diag()

    def _pop_vector(self, m, h0):
        from darksirens.runtime_binding import _decode_theta
        import jax.numpy as jnp
        _, pop, _, _ = _decode_theta(m.an, jnp.asarray([h0], dtype=jnp.float64), z_depth=None)
        return pop

    def _make_diag(self):
        import jax
        import jax.numpy as jnp
        from darksirens.cosmology.distances import threads_distance_table
        from darksirens.cosmology.volume import log_comoving_volume_prior
        from darksirens.likelihood.hierarchical import _ordinary_hierarchical_likelihood
        from darksirens.runtime_binding import _decode_theta
        cfg = self.cfg
        mg, ma = self.base["gal"], self.base["agn"]
        an_g, an_a = mg.an, ma.an
        pop = an_g.population
        common = dict(pop_model=pop.model_name, shared_beta=pop.shared_beta, shared_spin=pop.shared_spin,
                      shared_gamma=pop.shared_gamma, sel_batch_size=int(cfg["sel_batch"]),
                      selection_neff_soft_guard=False, max_likelihood_variance=float(cfg["max_var"]),
                      pe_event_block=int(cfg["pe_event_block"]), angular_model=an_g.angular_model,
                      return_diagnostics=True)
        n_events, nsamp, n_draw = self.n_events, self.nsamp, self.n_draw

        @threads_distance_table()
        def diag(h0f, gw_pe, gw_sel, distance_table=None):
            th = h0f[:1]
            f = h0f[1]
            cosmology, pop_g, _, angular = _decode_theta(an_g, th, z_depth=None)
            _, pop_a, _, _ = _decode_theta(an_a, th, z_depth=None)
            lw = (jnp.log1p(-f), jnp.log(f))

            def prior(z, _pix, _catalog):
                lv = log_comoving_volume_prior(z, cosmology)
                return [lw[0] + lv, lw[1] + lv]

            d = _ordinary_hierarchical_likelihood(
                cosmology, pop_g, gw_pe, None, gw_sel, None, n_events, nsamp, n_draw,
                log_prior_pe=prior, log_prior_sel=prior, angular_params=angular,
                branch_populations=(pop_g, pop_a), **common)
            return (d.log_likelihood, jnp.sum(d.event_mc_variance), d.n_eff, d.log_mu,
                    d.selection_log_correction, d.event_log_evidence, d.event_mc_variance)

        self.diag = jax.jit(diag)
        self.ops = (mg.b.gw_pe, mg.b.gw_selection)

    def theta(self, coords):
        import jax.numpy as jnp
        return jnp.asarray([float(coords["H0"]), float(coords["f_agn"])], dtype=jnp.float64)

    def evaluate(self, coords):
        r = self.diag(self.theta(coords), *self.ops)
        return Model._pack(self, r)

    def evaluate_events(self, coords):
        r = self.diag(self.theta(coords), *self.ops)
        out = Model._pack(self, r)
        out["event_log_evidence"] = np.asarray(r[5], dtype=float)
        out["event_mc_variance"] = np.asarray(r[6], dtype=float)
        return out

    def check_diag_equals_bound(self, points):
        """At f = 0 and f = 1 the mixture is core's bound spectral likelihood of the galaxy-class and
        AGN-class population (exactly the branch with weight 1)."""
        rows = []
        for c in points:
            for f, cls in ((0.0, "gal"), (1.0, "agn")):
                vb = self.base[cls].bound_call({"H0": c["H0"]})
                vd = self.evaluate({"H0": c["H0"], "f_agn": f})["lnL"]
                ok = (vb == vd) if not np.isfinite(vb) else abs(vd - vb) <= 1e-8 * max(1.0, abs(vb))
                log(f"mixture at f={f:g} vs core's bound spectral ({cls} population) at H0 {c['H0']}: "
                    f"{vd:.10f} vs {vb:.10f}")
                rows.append({"coords": {"H0": c["H0"], "f_agn": f}, "bound": vb, "diag": vd,
                             "abs_diff": abs(vb - vd) if np.isfinite(vb) else None, "population": cls})
                if not ok:
                    sys.exit(f"[fatal] the mixture at f={f} differs from core's bound {cls} call: {vd} vs {vb}")
        return rows

    extra_keys = ()


def make_model(cfg):
    """The model object of a run: Model, or a composite for hosts_known / mass_spectrum_alone with
    two populations."""
    if cfg["analysis"] == "hosts_known":
        return HostsKnownModel(cfg)
    if cfg["analysis"] == "mass_spectrum_alone" and "f_agn" in cfg["free"]:
        return SpectralMixtureModel(cfg)
    return Model(cfg)
