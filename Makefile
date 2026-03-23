.PHONY: check fmt clippy test audit deny vet bench build doc clean

# Run all CI checks locally
check: fmt clippy test audit

# Format check
fmt:
	cargo fmt --all -- --check

# Lint (zero warnings)
clippy:
	cargo clippy --all-targets -- -D warnings

# Run test suite
test:
	cargo test

# Security audit
audit:
	cargo audit

# Supply-chain checks (license + advisory + source)
deny:
	cargo deny check

# Supply-chain audit (cargo-vet)
vet:
	cargo vet

# Run benchmarks with history
bench:
	./scripts/bench-history.sh

# Build release
build:
	cargo build --release

# Generate documentation
doc:
	cargo doc --no-deps

# Clean build artifacts
clean:
	cargo clean
