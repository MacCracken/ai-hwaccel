# Migration Guide

## v1.x (Rust) → v2.0.0 (Cyrius)

v2.0.0 is a complete rewrite from Rust to Cyrius. The detection capabilities
are equivalent but the API has changed from Rust method syntax to Cyrius
function calls.

### API mapping

| Rust (v1.2.0) | Cyrius (v2.0.0) |
|----------------|-----------------|
| `AcceleratorRegistry::detect()` | `registry_detect()` |
| `AcceleratorRegistry::detect_async()` | `registry_detect_threaded()` |
| `DetectBuilder::new()` | `builder_all()` |
| `DetectBuilder::none()` | `builder_none()` |
| `builder.with_cuda()` | `builder_with(mask, BACKEND_CUDA)` |
| `registry.best_available()` | `reg_best_available(r)` |
| `registry.total_memory()` | `reg_total_memory(r)` |
| `registry.has_accelerator()` | `reg_has_accelerator(r)` |
| `registry.plan_sharding(params, quant)` | `reg_plan_sharding(r, params, quant)` |
| `registry.suggest_quantization(params)` | `reg_suggest_quant(r, params)` |
| `CachedRegistry::new(ttl)` | `cached_registry_new(ttl_secs)` |
| `cached.get()` | `cached_get(c)` |
| `LazyRegistry::new()` | `lazy_new()` |
| `lazy.by_family(Family::Gpu)` | `lazy_by_family(lr, FAMILY_GPU)` |
| `AcceleratorProfile::cuda(id, mem)` | `profile_cuda(id, mem)` |
| `profile.memory_bytes` | `profile_memory_bytes(p)` |
| `profile.accelerator.is_gpu()` | `accel_is_gpu(profile_accel_type(p))` |
| `can_run(model, quant, mem)` | `model_can_run(m, quant, mem)` |
| `detect_format(path)` | `detect_model_format(path)` |
| `detect_format_from_bytes(bytes)` | `detect_format_from_bytes(buf, len)` |

### Binary

| | Rust | Cyrius |
|--|------|--------|
| Format | ELF (via LLVM) | ELF (direct x86_64) |
| Size | 708 KB | 217 KB |
| Dependencies | 131 crates | 0 |

### JSON output

The JSON schema is unchanged. v1.x and v2.0.0 produce identical JSON
structures. `schema_version` remains `2`.

### What's new in v2.0.0

- `model_format.cyr` — SafeTensors/GGUF/ONNX/PyTorch header detection
- `requirement.cyr` — accelerator requirement matching for scheduling
- `async_detect.cyr` — threaded detection (CLI backends in parallel)
- `cache.cyr` — TTL-based caching (memory + disk)
- `lazy.cyr` — per-family lazy detection

### What's removed

- **Rust crate** — no longer published to crates.io
- **C FFI** (`ffi.rs`) — Cyrius is native, no wrapper needed
- **Windows detection** — Cyrius doesn't target Windows yet (v4.0.0)
- **serde/tokio/tracing** — replaced by manual JSON, thread.cyr, stderr
