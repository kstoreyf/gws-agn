#!/usr/bin/env python3
"""
Standalone GPU timing test script.
Use to compare GPU performance across clusters (yours vs colleague's).
Run: python gpu_timing_test.py
Requires: pip install jax[cuda12]   (or jax[cuda11] depending on your CUDA version)
"""

import sys
import time

def main():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("JAX not found. Install with: pip install jax[cuda12]")
        sys.exit(1)

    # --- GPU detection ---
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if not gpu_devices:
        print("No GPU found by JAX. Check that:")
        print("  - You have a GPU allocated")
        print("  - JAX was installed with CUDA support (pip install jax[cuda12])")
        print(f"  - Available devices: {devices}")
        sys.exit(1)

    gpu = gpu_devices[0]
    print(f"JAX version: {jax.__version__}")
    print(f"GPU device: {gpu}")
    print(f"All devices: {devices}")
    print()

    # --- Simple timed task: repeated large matrix multiply (good GPU workout) ---
    # Adjust n (matrix size) or n_repeats if you run out of memory or want heavier load
    n = 8192
    n_repeats = 400  # multiply this many times per timed run
    key = jax.random.PRNGKey(0)
    a = jax.device_put(jax.random.normal(key, (n, n), dtype=jnp.float32), gpu)
    b = jax.device_put(jax.random.normal(key, (n, n), dtype=jnp.float32), gpu)

    # JIT-compile a loop of matmuls so the repeated work hits the GPU hard
    @jax.jit
    def matmul_loop(a, b):
        c = a
        for _ in range(n_repeats):
            c = jnp.matmul(c, b)
        return c

    # Warmup: triggers JIT compilation and caches kernels
    for _ in range(3):
        c = matmul_loop(a, b)
        c.block_until_ready()

    # Timed runs
    n_runs = 5
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        c = matmul_loop(a, b)
        c.block_until_ready()  # ensures GPU is actually done before stopping timer
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_s = sum(times) / len(times)
    std_s = (sum((t - mean_s) ** 2 for t in times) / len(times)) ** 0.5
    # 2*n^3 FLOPS per matmul, n_repeats matmuls per run
    flops = 2 * (n ** 3) * n_repeats
    gflops = (flops / mean_s) / 1e9

    print("Task: {} x matrix multiply {} x {} (float32)".format(n_repeats, n, n))
    print("Runs: {}".format(n_runs))
    print("Time: {:.4f} ± {:.4f} s".format(mean_s, std_s))
    print("Throughput: {:.1f} GFLOPS".format(gflops))
    print()
    print("Use this script on both clusters and compare Time and GFLOPS.")

if __name__ == "__main__":
    main()
