//! Serialize a registry to JSON and deserialize it back.
//!
//! ```sh
//! cargo run --example json_output
//! ```

use ai_hwaccel::{AcceleratorProfile, AcceleratorRegistry, TpuVersion};

fn main() {
    // Build a registry manually.
    let registry = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 80 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(1, 80 * 1024 * 1024 * 1024),
        AcceleratorProfile::tpu(0, 4, TpuVersion::V5p),
    ]);

    // Serialize to pretty JSON.
    let json = serde_json::to_string_pretty(&registry).unwrap();
    println!("Serialized registry:\n{}\n", json);

    // Deserialize back.
    let restored: AcceleratorRegistry = serde_json::from_str(&json).unwrap();
    println!(
        "Restored {} profiles, best: {}",
        restored.all_profiles().len(),
        restored.best_available().unwrap()
    );
}
