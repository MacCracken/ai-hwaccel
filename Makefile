.PHONY: check fmt clippy test audit deny vet bench coverage fuzz msrv build doc clean

# Run all CI checks locally
check: fmt clippy test audit

# Format check
fmt:
	cargo fmt --all -- --check

# Lint (zero warnings)
clippy:
	cargo clippy --all-features --all-targets -- -D warnings

# Run test suite
test:
	cargo test --all-features

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

# Generate coverage report
coverage:
	cargo llvm-cov --all-features --html --output-dir coverage/
	@echo "Coverage report: coverage/html/index.html"

# Run fuzz targets (15s each)
fuzz:
	@for target in $$(cargo +nightly fuzz list 2>/dev/null); do \
		echo "Fuzzing $$target..."; \
		cargo +nightly fuzz run $$target -- -max_total_time=15 || true; \
	done

# Check minimum supported Rust version
msrv:
	cargo +1.89 check --all-features

# Build release
build:
	cargo build --release --all-features

# Generate documentation (warnings are errors)
doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# Clean build artifacts
clean:
	cargo clean
	rm -rf coverage/
