//! Python bindings for ai-hwaccel.
//!
//! Provides `detect()`, `suggest_quantization()`, `plan_sharding()`,
//! `system_io()`, and `estimate_training_memory()` as top-level functions,
//! plus `PyRegistry`, `PyProfile`, and `PyShardingPlan` wrapper classes.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ai_hwaccel_core::{
    AcceleratorProfile, AcceleratorRegistry, MemoryEstimate, QuantizationLevel, ShardingPlan,
    ShardingStrategy, TrainingMethod, TrainingTarget,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a quantization level string into the Rust enum.
fn parse_quantization(s: &str) -> PyResult<QuantizationLevel> {
    match s.to_ascii_uppercase().as_str() {
        "FP32" | "NONE" | "F32" => Ok(QuantizationLevel::None),
        "FP16" | "FLOAT16" | "F16" => Ok(QuantizationLevel::Float16),
        "BF16" | "BFLOAT16" => Ok(QuantizationLevel::BFloat16),
        "INT8" | "I8" => Ok(QuantizationLevel::Int8),
        "INT4" | "I4" => Ok(QuantizationLevel::Int4),
        other => Err(PyValueError::new_err(format!(
            "unknown quantization level: '{}' (expected FP32, FP16, BF16, INT8, or INT4)",
            other,
        ))),
    }
}

/// Parse a training method string into the Rust enum.
fn parse_training_method(s: &str) -> PyResult<TrainingMethod> {
    match s.to_ascii_lowercase().as_str() {
        "full" | "full_fine_tune" | "fullfinetune" => Ok(TrainingMethod::FullFineTune),
        "lora" => Ok(TrainingMethod::LoRA),
        "qlora" | "qlora-4bit" | "qlora_4bit" => Ok(TrainingMethod::QLoRA { bits: 4 }),
        "qlora-8bit" | "qlora_8bit" => Ok(TrainingMethod::QLoRA { bits: 8 }),
        "prefix" => Ok(TrainingMethod::Prefix),
        "dpo" => Ok(TrainingMethod::DPO),
        "rlhf" => Ok(TrainingMethod::RLHF),
        "distillation" => Ok(TrainingMethod::Distillation),
        other => Err(PyValueError::new_err(format!(
            "unknown training method: '{}' (expected full, lora, qlora, prefix, dpo, rlhf, or distillation)",
            other,
        ))),
    }
}

/// Parse a training target string into the Rust enum.
fn parse_training_target(s: &str) -> PyResult<TrainingTarget> {
    match s.to_ascii_lowercase().as_str() {
        "gpu" => Ok(TrainingTarget::Gpu),
        "tpu" => Ok(TrainingTarget::Tpu),
        "gaudi" => Ok(TrainingTarget::Gaudi),
        "cpu" => Ok(TrainingTarget::Cpu),
        other => Err(PyValueError::new_err(format!(
            "unknown training target: '{}' (expected gpu, tpu, gaudi, or cpu)",
            other,
        ))),
    }
}

/// Convert an `AcceleratorProfile` to a Python dict.
fn profile_to_dict<'py>(py: Python<'py>, p: &AcceleratorProfile) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    // Serialize via serde_json for the accelerator type (complex enum)
    let accel_json: serde_json::Value =
        serde_json::to_value(&p.accelerator).map_err(|e| PyValueError::new_err(e.to_string()))?;
    d.set_item("accelerator", json_to_py(py, &accel_json)?)?;
    d.set_item("accelerator_str", format!("{}", p.accelerator))?;
    d.set_item("family", format!("{}", p.accelerator.family()))?;
    d.set_item("available", p.available)?;
    d.set_item("memory_bytes", p.memory_bytes)?;
    d.set_item(
        "memory_gb",
        p.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    )?;
    d.set_item("compute_capability", p.compute_capability.as_deref())?;
    d.set_item("driver_version", p.driver_version.as_deref())?;
    d.set_item("memory_bandwidth_gbps", p.memory_bandwidth_gbps)?;
    d.set_item("memory_used_bytes", p.memory_used_bytes)?;
    d.set_item("memory_free_bytes", p.memory_free_bytes)?;
    d.set_item("pcie_bandwidth_gbps", p.pcie_bandwidth_gbps)?;
    d.set_item("numa_node", p.numa_node)?;
    d.set_item("temperature_c", p.temperature_c)?;
    d.set_item("power_watts", p.power_watts)?;
    d.set_item("gpu_utilization_percent", p.gpu_utilization_percent)?;
    Ok(d)
}

/// Convert a serde_json::Value to a PyObject.
fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<PyObject> {
    match v {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, val) in map {
                dict.set_item(k, json_to_py(py, val)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Convert a `ShardingPlan` to a Python dict.
fn sharding_plan_to_dict<'py>(
    py: Python<'py>,
    plan: &ShardingPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("strategy", format!("{}", plan.strategy))?;
    d.set_item(
        "strategy_type",
        match &plan.strategy {
            ShardingStrategy::None => "none",
            ShardingStrategy::PipelineParallel { .. } => "pipeline_parallel",
            ShardingStrategy::TensorParallel { .. } => "tensor_parallel",
            ShardingStrategy::DataParallel { .. } => "data_parallel",
        },
    )?;
    d.set_item("total_memory_bytes", plan.total_memory_bytes)?;
    d.set_item(
        "total_memory_gb",
        plan.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    )?;
    d.set_item("estimated_tokens_per_sec", plan.estimated_tokens_per_sec)?;
    d.set_item("num_shards", plan.shards().len())?;

    let shards_list = PyList::empty(py);
    for shard in plan.shards() {
        let sd = PyDict::new(py);
        sd.set_item("shard_id", shard.shard_id)?;
        sd.set_item("layer_range", (shard.layer_range.0, shard.layer_range.1))?;
        sd.set_item("device", format!("{}", shard.device))?;
        sd.set_item("memory_bytes", shard.memory_bytes)?;
        sd.set_item(
            "memory_gb",
            shard.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        )?;
        shards_list.append(sd)?;
    }
    d.set_item("shards", shards_list)?;
    Ok(d)
}

/// Convert a `MemoryEstimate` to a Python dict.
fn memory_estimate_to_dict<'py>(
    py: Python<'py>,
    est: &MemoryEstimate,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("model_gb", est.model_gb)?;
    d.set_item("optimizer_gb", est.optimizer_gb)?;
    d.set_item("activation_gb", est.activation_gb)?;
    d.set_item("total_gb", est.total_gb)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// PyRegistry wrapper
// ---------------------------------------------------------------------------

/// Registry of detected hardware accelerators.
///
/// Use `detect()` to create a new registry, or `PyRegistry.from_json()` to
/// deserialize one from JSON.
#[pyclass(name = "Registry")]
#[derive(Clone)]
struct PyRegistry {
    inner: AcceleratorRegistry,
}

#[pymethods]
impl PyRegistry {
    /// Return all accelerator profiles as a list of dicts.
    fn all_profiles<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for p in self.inner.all_profiles() {
            list.append(profile_to_dict(py, p)?)?;
        }
        Ok(list)
    }

    /// Return only available accelerator profiles.
    fn available<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for p in self.inner.available() {
            list.append(profile_to_dict(py, p)?)?;
        }
        Ok(list)
    }

    /// Return the best available accelerator profile, or None.
    fn best_available<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        match self.inner.best_available() {
            Some(p) => Ok(Some(profile_to_dict(py, p)?.into_any().unbind())),
            None => Ok(None),
        }
    }

    /// Total memory in bytes across all available devices.
    fn total_memory(&self) -> u64 {
        self.inner.total_memory()
    }

    /// Total accelerator (non-CPU) memory in bytes.
    fn total_accelerator_memory(&self) -> u64 {
        self.inner.total_accelerator_memory()
    }

    /// Whether any non-CPU accelerator is available.
    fn has_accelerator(&self) -> bool {
        self.inner.has_accelerator()
    }

    /// Suggest a quantization level for the given model parameter count.
    fn suggest_quantization(&self, model_params: u64) -> String {
        format!("{}", self.inner.suggest_quantization(model_params))
    }

    /// Generate a sharding plan for a model.
    fn plan_sharding<'py>(
        &self,
        py: Python<'py>,
        model_params: u64,
        quantization: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let quant = parse_quantization(quantization)?;
        let plan = self.inner.plan_sharding(model_params, &quant);
        sharding_plan_to_dict(py, &plan)
    }

    /// Return system I/O information as a dict.
    fn system_io<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let sio = self.inner.system_io();
        let d = PyDict::new(py);

        let interconnects = PyList::empty(py);
        for ic in &sio.interconnects {
            let icd = PyDict::new(py);
            icd.set_item("kind", format!("{}", ic.kind))?;
            icd.set_item("name", &ic.name)?;
            icd.set_item("bandwidth_gbps", ic.bandwidth_gbps)?;
            icd.set_item("state", ic.state.as_deref())?;
            interconnects.append(icd)?;
        }
        d.set_item("interconnects", interconnects)?;

        let storage = PyList::empty(py);
        for dev in &sio.storage {
            let sd = PyDict::new(py);
            sd.set_item("name", &dev.name)?;
            sd.set_item("kind", format!("{}", dev.kind))?;
            sd.set_item("bandwidth_gbps", dev.bandwidth_gbps)?;
            storage.append(sd)?;
        }
        d.set_item("storage", storage)?;

        d.set_item("has_interconnect", sio.has_interconnect())?;
        d.set_item(
            "total_interconnect_bandwidth_gbps",
            sio.total_interconnect_bandwidth_gbps(),
        )?;

        Ok(d)
    }

    /// Serialize the registry to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Deserialize a registry from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner = AcceleratorRegistry::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRegistry { inner })
    }

    /// Return detection warnings as a list of strings.
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings().iter().map(|w| w.to_string()).collect()
    }

    /// Schema version of this registry.
    fn schema_version(&self) -> u32 {
        self.inner.schema_version()
    }

    fn __repr__(&self) -> String {
        let profiles = self.inner.all_profiles().len();
        let avail = self.inner.available().len();
        let mem_gb = self.inner.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        format!(
            "Registry(profiles={}, available={}, total_memory={:.1} GB)",
            profiles, avail, mem_gb,
        )
    }

    fn __len__(&self) -> usize {
        self.inner.all_profiles().len()
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Detect all available AI hardware accelerators on this system.
///
/// Returns a `Registry` containing profiles for every detected device.
#[pyfunction]
fn detect() -> PyRegistry {
    PyRegistry {
        inner: AcceleratorRegistry::detect(),
    }
}

/// Suggest a quantization level for the given model and hardware.
///
/// Args:
///     model_params: Total number of model parameters (e.g. 7_000_000_000 for 7B).
///     registry: Optional Registry. If not provided, detects hardware automatically.
///
/// Returns:
///     A quantization level string: "FP32", "FP16", "BF16", "INT8", or "INT4".
#[pyfunction]
#[pyo3(signature = (model_params, registry=None))]
fn suggest_quantization(model_params: u64, registry: Option<&PyRegistry>) -> String {
    let reg = match registry {
        Some(r) => r.inner.clone(),
        None => AcceleratorRegistry::detect(),
    };
    format!("{}", reg.suggest_quantization(model_params))
}

/// Generate a sharding plan for a model.
///
/// Args:
///     model_params: Total number of model parameters.
///     quantization: Quantization level string ("FP32", "FP16", "BF16", "INT8", "INT4").
///     registry: Optional Registry. If not provided, detects hardware automatically.
///
/// Returns:
///     A dict with strategy, shards, memory, and throughput estimates.
#[pyfunction]
#[pyo3(signature = (model_params, quantization, registry=None))]
fn plan_sharding<'py>(
    py: Python<'py>,
    model_params: u64,
    quantization: &str,
    registry: Option<&PyRegistry>,
) -> PyResult<Bound<'py, PyDict>> {
    let reg = match registry {
        Some(r) => r.inner.clone(),
        None => AcceleratorRegistry::detect(),
    };
    let quant = parse_quantization(quantization)?;
    let plan = reg.plan_sharding(model_params, &quant);
    sharding_plan_to_dict(py, &plan)
}

/// Return system I/O information (interconnects, storage) as a dict.
///
/// Args:
///     registry: Optional Registry. If not provided, detects hardware automatically.
#[pyfunction]
#[pyo3(signature = (registry=None))]
fn system_io<'py>(
    py: Python<'py>,
    registry: Option<&PyRegistry>,
) -> PyResult<Bound<'py, PyDict>> {
    let reg = match registry {
        Some(r) => r.clone(),
        None => PyRegistry {
            inner: AcceleratorRegistry::detect(),
        },
    };
    reg.system_io(py)
}

/// Estimate training/fine-tuning memory requirements.
///
/// Args:
///     model_params_millions: Model parameters in millions (e.g. 7000 for 7B).
///     method: Training method string ("full", "lora", "qlora", "prefix", "dpo", "rlhf", "distillation").
///     target: Target device string ("gpu", "tpu", "gaudi", "cpu").
///
/// Returns:
///     A dict with model_gb, optimizer_gb, activation_gb, total_gb.
#[pyfunction]
fn estimate_training_memory<'py>(
    py: Python<'py>,
    model_params_millions: u64,
    method: &str,
    target: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let m = parse_training_method(method)?;
    let t = parse_training_target(target)?;
    let est = ai_hwaccel_core::estimate_training_memory(model_params_millions, m, t);
    memory_estimate_to_dict(py, &est)
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Universal AI hardware accelerator detection for Python.
///
/// Detects GPUs, TPUs, NPUs, and cloud AI ASICs, then provides
/// quantization suggestions, sharding plans, and training memory estimates.
#[pymodule]
fn ai_hwaccel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRegistry>()?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(suggest_quantization, m)?)?;
    m.add_function(wrap_pyfunction!(plan_sharding, m)?)?;
    m.add_function(wrap_pyfunction!(system_io, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_training_memory, m)?)?;
    Ok(())
}
