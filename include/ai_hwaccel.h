/* ai-hwaccel C FFI header
 *
 * Link against the ai_hwaccel shared library built with:
 *   cargo build --release --lib
 *
 * SPDX-License-Identifier: AGPL-3.0-only
 */

#ifndef AI_HWACCEL_H
#define AI_HWACCEL_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque registry handle. */
typedef struct HwAccelRegistry HwAccelRegistry;

/* Detect all hardware accelerators. Caller must free with ai_hwaccel_free(). */
HwAccelRegistry *ai_hwaccel_detect(void);

/* Free a registry. Safe to call with NULL. */
void ai_hwaccel_free(HwAccelRegistry *ptr);

/* Number of detected device profiles (including CPU). */
uint32_t ai_hwaccel_device_count(const HwAccelRegistry *ptr);

/* Whether any non-CPU accelerator is available. */
bool ai_hwaccel_has_accelerator(const HwAccelRegistry *ptr);

/* Total accelerator memory in bytes (excluding CPU). */
uint64_t ai_hwaccel_accelerator_memory(const HwAccelRegistry *ptr);

/* Serialize to JSON. Caller must free with ai_hwaccel_free_string(). */
char *ai_hwaccel_json(const HwAccelRegistry *ptr);

/* Free a JSON string. Safe to call with NULL. */
void ai_hwaccel_free_string(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* AI_HWACCEL_H */
