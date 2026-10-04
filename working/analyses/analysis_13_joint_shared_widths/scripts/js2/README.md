# A13 on the Jetstream2 A100 (js2a100)

Owner request 2026-10-03: rita was busy, so A13 runs on the Jetstream2 A100 VM
(`ssh js2a100` = exouser@149.165.174.18). Everything on the VM lives under
`/media/volume/tbs/gws-agn-data-js2` (`/media/volume/h0tbs/gws-agn-data-js2` is the same layout if
tbs runs short). Results come back to hilda under
`/hildafs/projects/phy230054p/magana/gws-agn-data-js2`. The small result files (json/npz) are also
copied into this analysis's `results/`.

The VM cannot reach hilda, so every transfer runs **from hilda**.

| step | where | command |
|---|---|---|
| ship core bf58aa6 + driver + inputs, build the venv | hilda | `scripts/js2/push_to_js2.sh` |
| start (or resume) seed S, detached | VM | `cd $JS2_A13/scripts && setsid nohup bash run_a13_js2.sh S >/dev/null 2>&1 &` |
| copy results, checkpoints and logs back | hilda | `scripts/js2/pull_from_js2.sh` |

What's on the VM:

- `src/darksirens-core-bf58aa6/`: `git archive` of core bf58aa6. It is imported through `PYTHONPATH`
  and never pip-installed, so the driver's provenance check (`darksirens-core-bf58aa6` in
  `darksirens.__file__`) holds.
- `env/core-bf58aa6/`: python 3.11 venv with core's pins (jax/jaxlib 0.4.34, numpy 1.26.4, scipy 1.12.0,
  h5py 3.12.1) and dynesty 2.1.4. These match the hilda venv. The CUDA-12 plugin and nvidia wheels are
  pinned from `~/qchieffenv`.
- `data/seed100/`: only the four A13 inputs, checked against `SHA256SUMS` written on hilda.
- `analysis_13_joint_shared_widths/{scripts,results,queue,logs,ref}`: `ref/` holds the pre-flight
  reference cells measured on rita. The driver stops if the VM does not reproduce them to 1e-8.

**A13 does not fit on js2a100.** Its GPU is a 20 GB vGPU slice (A100X-20C). The pinned field-kernel
build (`build_pinned_catalog_kernel` → `_state(catalog)`) ran out of device memory with the default
JAX cap, with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, and with the galaxy-list/gather layouts. Under
the platform allocator it asked for a single 27.4 GiB buffer. Host RAM was fine (~14 GB of 58).
Use an 80 GB card: point `JS2`/`JS2_ROOT` in `config.sh` at it. `run_a13_js2.sh` hardcodes its
`ROOT`, so change that too. `run_a13_js2.sh` logs host RSS and GPU memory every 60 s to
`logs/<tag>.mem`.
