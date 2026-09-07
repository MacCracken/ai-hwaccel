# `src/cache.cyr` raw `syscall(83)` / `syscall(87)` are not mkdir/unlink on ELF-aarch64 — the disk cache silently does nothing there

> **RESOLVED in 2.3.22 (2026-09-07).** Both call sites now use the per-target
> `sys_mkdir` / `sys_unlink` peers behind the `read_symlink` `#ifdef` pattern,
> with the AGNOS pathlen arity handled separately. Re-measured under
> `qemu-aarch64`: `old syscall(83) = -9` -> `new sys_mkdir = 0`, directory
> created. The qemu-aarch64 disk-cache CI test is **still open**.

**Filed:** 2026-09-07 (found auditing the cyrius `6.5.36 → 6.6.0` bump for 2.3.21)
**Severity:** Medium — `DiskCachedRegistry` never creates its directory and never
removes its cache file on aarch64. Silent: no crash, no diagnostic, no warning.
**Repos:** ai-hwaccel `2.3.21` · cyrius `6.6.0`
**Status:** RESOLVED in 2.3.22 — follow-ups noted above. **Pre-existing** — identical under the 6.5.36 pin, not caused by
the bump. Kept out of a toolchain-bump release deliberately.

## Summary

`src/cache.cyr` issues two raw **x86_64** syscall numbers with no target guard:

```
src/cache.cyr:178:    syscall(83, str_data(dir_s), 493);  # SYS_MKDIR, 0o755
src/cache.cyr:201:    syscall(87, path);                  # SYS_UNLINK
```

ELF-aarch64 has **no `mkdir` or `unlink` syscall at all** — only `mkdirat` (34)
and `unlinkat` (35). `lib/syscalls_aarch64_linux.cyr:67-68`:

```
SYS_MKDIRAT = 34;       # → sys_mkdir (AT_FDCWD)
SYS_UNLINKAT = 35;      # → sys_rmdir (AT_REMOVEDIR) and sys_unlink
```

There is no `SYS_MKDIR` / `SYS_UNLINK` on that table at all, and cyrius's
x86-compat translation does **not** remap 83/87 (it does remap `1` write, `60`
exit and `228` clock_gettime, which is why only these two are affected).

## Measured

Probe replicating both call sites verbatim, built by `cycc_aarch64` from the
6.6.0 release tarball and run under `qemu-aarch64`:

```
x86_64 native :  mkdir(83)=0    unlink(87)=-21    dir created? YES
aarch64 (qemu):  mkdir(83)=-9   unlink(87)=-14    dir created? NO
```

`-9` is `EBADF` and `-14` is `EFAULT` — the path pointer is being read as a file
descriptor by whatever 83 and 87 actually are on that ABI.

## Blast radius

`src/main.cyr` never calls the disk-cache API, so it is eliminated from the CLI
binary by `CYRIUS_DCE=1` and **the shipped `ai-hwaccel` executable is
unaffected on every platform**. The exposure is **library consumers of
`dist/ai-hwaccel.cyr` running on aarch64** who call `disk_cached_get_or_detect`
/ `disk_cached_invalidate`: the cache directory is never created, so every call
re-detects, and invalidation never removes anything.

Note that 6.5.51's new ELF-aarch64 raw-syscall diagnostic does **not** catch
this — 83 and 87 are in its exclusion set. The `cyrius build --aarch64` log
stays quiet, which is exactly why this needs a source fix rather than waiting
for the compiler to point at it.

## Proposed fix

Replace both with the stdlib peers behind the same `#ifdef CYRIUS_TARGET_*`
pattern `src/detect/platform.cyr`'s `read_symlink` already uses. `sys_mkdir` /
`sys_unlink` exist for linux / aarch64 / macos / windows, but **the AGNOS peers
take an extra pathlen argument** (`lib/syscalls_x86_64_agnos.cyr:785,791`), so a
single unguarded call is wrong on that target — guard it the same way.

Then add a disk-cache test that runs under `qemu-aarch64` in CI, since the
current suite proves nothing here: it only ever runs x86_64.

## Related

Same class as
[`_monotonic_secs` unguarded on macOS/Windows](2026-09-07-monotonic-secs-unguarded-on-macos-windows.md)
— `src/cache.cyr` assumes the x86_64-Linux syscall ABI in three places, and
`_monotonic_secs` is the third. Worth fixing together.
