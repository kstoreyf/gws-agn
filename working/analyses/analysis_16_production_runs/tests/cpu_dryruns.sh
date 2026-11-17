#!/bin/bash
# CPU dry runs of every configuration class (20 events, 20,000 injections, 3 cells per axis or a few
# nested calls). Each checks: options resolve, core builds and binds, the diagnostics program equals
# the bound call at the truth and at the prior centre, and the products are written.
#   bash tests/cpu_dryruns.sh            (on a CPU node; ~15 min; logs in logs/dryrun/)
#   bash tests/cpu_dryruns.sh NAME ...   only the named dry runs (e.g. the stage-5 ones: s5_*)
set -u
A16=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs
cd $A16
export JAX_PLATFORMS=cpu PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
OUT="--out-root tests/dryrun_runs --dry-run"
ONLY=" $* "
mkdir -p logs/dryrun
run() {  # name, args...
  local name=$1; shift
  if [ "$ONLY" != "  " ] && [[ "$ONLY" != *" $name "* ]]; then return; fi
  $PY scripts/prod_driver.py "$@" $OUT > logs/dryrun/$name.log 2>&1
  echo "$name exit $?"
}
run closure_a14_convention --mode grid --analysis known_population --catalogs both --events-kind equal_populations \
    --realization 100 --events-file /hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100/events/events.h5 \
    --redraw-key 141002026,2,0 --nsamp 8000 --complete-model count_ratio --density given --log10n-gal -24 \
    --log10n-agn -24 --density-units physical --zmax 5.0 --max-var 1e6 --grid H0=64,74,0.25 --grid f_agn=0.05,0.55,0.025 \
    --coarse none --run-name closure/a14_convention
run known_both_complete --mode grid --analysis known_population --catalogs both --events-kind equal_populations --realization 101
run known_gal_galhosted --mode grid --analysis known_population --catalogs galaxy --events-kind galaxy_hosted --realization 102
run known_agn_agnhosted --mode grid --analysis known_population --catalogs agn --events-kind agn_hosted --realization 103
run known_m18_counts --mode grid --analysis known_population --catalogs both --flux 18 --events-kind equal_populations --realization 100
run known_m20_agnhalf --mode grid --analysis known_population --catalogs both --flux 20 --agn-density-factor 0.5 --events-kind equal_populations --realization 105
run free_densities_m19 --mode nested --analysis free_densities --catalogs both --flux 19 --events-kind equal_populations --realization 100 --nlive 10 --maxcall-total 30 --chunk-calls 15 --nsamp 2000
run single_both --mode nested --analysis single_population --catalogs both --events-kind equal_populations --realization 100 --nlive 14 --maxcall-total 40 --chunk-calls 20 --nsamp 2000
run single_none_hostdep --mode nested --analysis single_population --catalogs none --events-kind host_dependent --realization 101 --nlive 12 --maxcall-total 40 --chunk-calls 20 --nsamp 2000
# stage 5 (observed-like family, host-dependent events): the run list's arguments for realization 100
S5="--flux complete --events-kind host_dependent --events-suffix _obslike --family-name obslike --population-family obslike --realization 100 --injections-lane injections/injections_targeted_obslike.h5"
run s5_single_both --mode nested --analysis single_population --catalogs both $S5 --nlive 14 --maxcall-total 40 --chunk-calls 20 --nsamp 2000
run s5_hostdep_both --mode nested --analysis host_dependent_populations --catalogs both $S5 --nlive 12 --maxcall-total 40 --chunk-calls 20 --nsamp 2000
run s5_single_none --mode nested --analysis single_population --catalogs none $S5 --nlive 12 --maxcall-total 40 --chunk-calls 20 --nsamp 2000
