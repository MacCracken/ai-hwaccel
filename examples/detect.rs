//! Detect and print all available hardware accelerators.
//!
//! ```sh
//! cargo run --example detect
//! ```

use ai_hwaccel::AcceleratorRegistry;

fn main() {
    let registry = AcceleratorRegistry::detect();

    println!("Detected {} device(s):", registry.all_profiles().len());
    for profile in registry.all_profiles() {
        println!("  {}", profile);
    }

    if let Some(best) = registry.best_available() {
        println!("\nBest available: {}", best);
    }

    println!(
        "\nTotal memory: {:.1} GB",
        registry.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "Accelerator memory: {:.1} GB",
        registry.total_accelerator_memory() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    if !registry.warnings().is_empty() {
        println!("\nWarnings:");
        for w in registry.warnings() {
            println!("  {}", w);
        }
    }
}
