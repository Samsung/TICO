# Performance Benchmarks

This directory contains benchmark scripts that verify the performance requirements
specified in **docs/system_test.md**:

* **RNF‑1 – Conversion Speed**
* **RNF‑2 – Model Size**

Both requirements can be tested with:

```bash
python3 -m test.performance.benchmark_perf
```

The test uses baseline models (`Llama-3.2-1B` and `Llama-3.2-3B`).

Feel free to adjust thresholds or models as needed.
