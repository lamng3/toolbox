# Profiling

Profiling is done with callbacks, so you add it to any run with a config change:

```yaml
callbacks:
  - {name: console}
  - {name: jsonl}
  - {name: throughput}        # step_time_s, tokens_per_s
  - {name: memory_monitor}    # cuda_max_mem_mb, host_max_rss_mb
  - {name: torch_profiler, wait: 1, warmup: 1, active: 3}
```

`torch_profiler` writes Chrome/Perfetto traces to `<output_dir>/profiler/`. Open them at
[ui.perfetto.dev](https://ui.perfetto.dev) or in TensorBoard's profiler plugin. With
`profile_memory=true` they include allocation timelines.

For agents, every traced action records `latency_ms`, so `toolbox traces <dir>` doubles as a
per-agent latency profile. Wrap any function with `@tracer.traced(...)` to add it.

`@timed` (from `toolbox.core`) logs the duration of a function at DEBUG level. Set
`TOOLBOX_LOG_LEVEL=DEBUG` to see it.
