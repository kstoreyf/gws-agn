#!/usr/bin/env bash
# Regression of the --mass_spectrum change: the default (legacy) path and the core path
# of the edited generator must reproduce the files on disk bitwise (seed 100), and the
# new spectrum must run.  Everything is written under a scratch outroot whose catalogs/
# is a symlink to the signed-off catalogs; nothing in the dataset is written.
#   ./regress_generator.sh run | compare
set -uo pipefail
SCR=/hildafs/projects/phy230054p/magana/gws-agn-data-v3-scratch-obslike/regress
DATA=/hildafs/projects/phy230054p/magana/gws-agn-data-v3
GEN=/hildafs/projects/phy230014p/magana/gws-agn/working/data/generate_dataset.py
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
BITCHECK=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_0_pure_tracer_H0/scripts/bitcheck_events.py
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$PIN JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
S=100
case ${1:?run|compare} in
run)
  mkdir -p $SCR/seed$S/events
  ln -sfn $DATA/seed$S/catalogs $SCR/seed$S/catalogs
  ( $PY -u $GEN --seed $S --stage events --outroot $SCR --darksirens $PIN > $HERE/logs/regress_legacy_s$S.log 2>&1 ) &
  ( $PY -u $GEN --seed $S --stage events --outroot $SCR --darksirens $PIN --population_code core \
      --events_suffix _corepop > $HERE/logs/regress_core_s$S.log 2>&1 ) &
  ( $PY -u $GEN --seed $S --stage events --outroot $SCR --darksirens $PIN --population_code core \
      --dmu_chi_agn 0.10 --dmu_G_agn 5 --events_suffix _marked_dmu0p10_dmuG5_corepop \
      > $HERE/logs/regress_core_marked_s$S.log 2>&1 ) &
  ( $PY -u $GEN --seed $S --stage events --outroot $SCR --darksirens $PIN --population_code core \
      --mass_spectrum gwtc3_pp --seed_events $((S*1000+11)) --n_events 200 --nsamp 200 \
      --events_suffix _smoke_obslike > $HERE/logs/regress_smoke_obslike_s$S.log 2>&1 ) &
  ( $PY -u $GEN --seed $S --stage events --outroot $SCR --darksirens $PIN \
      --mass_spectrum gwtc3_pp --events_suffix _smoke_refuse > $HERE/logs/regress_refuse_legacy_s$S.log 2>&1 ) &
  wait; echo run done
  ;;
compare)
  for F in events.h5 events_corepop.h5 events_marked_dmu0p10_dmuG5_corepop.h5; do
    $PY $BITCHECK $DATA/seed$S/events/$F $SCR/seed$S/events/$F \
        --out $HERE/results/regress_bitcheck_${F%.h5}_s$S.json > /dev/null && r=PASS || r=FAIL
    echo "$F: disk vs edited generator: $r"
  done
  grep -h "needs --population_code core" $HERE/logs/regress_refuse_legacy_s$S.log && echo "legacy + gwtc3_pp refused: PASS"
  ;;
esac
