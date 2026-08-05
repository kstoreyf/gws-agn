#!/usr/bin/env bash
# Locate the guard-admissible event-count ceiling for the K=2 mixture on this mock.
#
# The default budget admits a cell only if sigma^2_total = Sum_i sigma^2_i +
# N^2/Neff <= 1.  Measured at the fagn0.3 truth coord: N=1000 -> 21.3,
# N=50 -> 1.247 (fails), N=25 -> 0.371 (passes).  The per-event mean is not
# constant across subsamples (0.0163 at N=1000 vs 0.0247 at N=50), so the ceiling
# has to be measured rather than extrapolated.  This builds a ladder of stratified
# subsamples and probes each at its own truth.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

DER=../data_derived
mkdir -p $DER ../results/guard_audit ../logs/guard_audit

for N in 30 35 40 45; do
  OUT=$DER/gw_fagn0.3_n${N}.h5
  [ -f "$OUT" ] || python build_event_subsample.py --in_path ../data/gw_fagn0.3.h5 \
      --out_path "$OUT" --n_events $N
  # Truth of the subsample (n_agn/N) is written into its attrs by the builder.
  FT=$(python -c "
import h5py,sys
with h5py.File('$OUT') as f: print(f'{f.attrs[\"subsample_truth_alpha_agn\"]:.6f}')")
  echo "=== $(date +%H:%M:%S) ceiling probe N=$N (truth $FT) ==="
  python diag_variance_guard.py --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path "$OUT" --gwselection_path $INJ --log10n0 -12 --log10n0_c2 -12 \
    --f_at "$FT" --out_json ../results/guard_audit/k2_dsf_n${N}_fagn0.3.json \
    > ../logs/guard_audit/k2_dsf_n${N}_fagn0.3.log 2>&1 || echo "FAILED: N=$N"
  grep -h "^\[guard\]" ../logs/guard_audit/k2_dsf_n${N}_fagn0.3.log | tail -1
done

echo "=== $(date +%H:%M:%S) CEILING SCAN DONE ==="
python aggregate_guard_audit.py --audit_dir ../results/guard_audit \
  --out_json ../results/guard_audit_summary.json --out_md ../GUARD_AUDIT.md
