//! Model file format detection.
//!
//! Parses `.safetensors`, `.gguf`, and `.onnx` file headers to extract model
//! metadata (parameter count, data type, tensor names) without loading the
//! full model into memory. Only the first few kilobytes are read.
//!
//! # Examples
//!
//! ```rust,no_run
//! use ai_hwaccel::model_format::{detect_format, ModelFormat};
//!
//! let metadata = detect_format(std::path::Path::new("model.safetensors")).unwrap();
//! println!("Format: {}", metadata.format);
//! println!("Parameters: {}", metadata.param_count.unwrap_or(0));
//! ```

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Maximum bytes to read from a file header.
const MAX_HEADER_BYTES: usize = 16 * 1024; // 16 KB

/// GGUF magic number: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"

/// ONNX protobuf file starts with field tag 0x08 (varint field 1).
const ONNX_IR_VERSION_TAG: u8 = 0x08;

/// Detected model file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelFormat {
    /// HuggingFace SafeTensors format.
    SafeTensors,
    /// GGML/GGUF format (llama.cpp).
    GGUF,
    /// ONNX (Open Neural Network Exchange).
    ONNX,
    /// PyTorch serialized format (pickle-based).
    PyTorch,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::GGUF => write!(f, "GGUF"),
            Self::ONNX => write!(f, "ONNX"),
            Self::PyTorch => write!(f, "PyTorch"),
        }
    }
}

/// Metadata extracted from a model file header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Detected format.
    pub format: ModelFormat,
    /// Estimated total parameter count (if extractable from header).
    pub param_count: Option<u64>,
    /// Weight data type (e.g. "F16", "BF16", "F32", "Q4_0").
    pub dtype: Option<String>,
    /// Number of tensors found in header.
    pub tensor_count: Option<u32>,
    /// GGUF version (if applicable).
    pub format_version: Option<u32>,
}

/// Detect the model format from a file path.
///
/// Reads only the first few kilobytes to identify the format and extract
/// metadata. Returns `None` if the format is unrecognised.
#[must_use]
pub fn detect_format(path: &Path) -> Option<ModelMetadata> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; MAX_HEADER_BYTES];
    let n = file.read(&mut buf).ok()?;
    buf.truncate(n);
    detect_format_from_bytes(&buf)
}

/// Detect the model format from raw bytes (typically the first 16 KB).
///
/// This is the WASM-compatible entry point — no file I/O required.
#[must_use]
pub fn detect_format_from_bytes(bytes: &[u8]) -> Option<ModelMetadata> {
    // Try each format in order of specificity.
    if let Some(meta) = parse_safetensors_header(bytes) {
        return Some(meta);
    }
    if let Some(meta) = parse_gguf_header(bytes) {
        return Some(meta);
    }
    if let Some(meta) = parse_onnx_header(bytes) {
        return Some(meta);
    }
    if is_pytorch_format(bytes) {
        return Some(ModelMetadata {
            format: ModelFormat::PyTorch,
            param_count: None,
            dtype: None,
            tensor_count: None,
            format_version: None,
        });
    }
    None
}

// ---------------------------------------------------------------------------
// SafeTensors parser
// ---------------------------------------------------------------------------

/// Parse SafeTensors file header.
///
/// Format: 8-byte LE header_size, then JSON metadata of that size.
/// JSON contains tensor names as keys with `{dtype, shape, data_offsets}`.
fn parse_safetensors_header(bytes: &[u8]) -> Option<ModelMetadata> {
    if bytes.len() < 8 {
        return None;
    }

    let header_size = u64::from_le_bytes(bytes[..8].try_into().ok()?) as usize;

    // Sanity: header should be reasonable (< 100 MB) and start with '{'.
    if header_size == 0 || header_size > 100 * 1024 * 1024 {
        return None;
    }

    // We may not have the full header in our buffer, but we can still
    // identify the format and parse what we have.
    let json_end = (8 + header_size).min(bytes.len());
    let json_bytes = &bytes[8..json_end];

    // Must start with '{' to be valid JSON object.
    let first_non_ws = json_bytes.iter().find(|b| !b.is_ascii_whitespace())?;
    if *first_non_ws != b'{' {
        return None;
    }

    // Try to parse as complete JSON if we have enough bytes.
    let json_str = std::str::from_utf8(json_bytes).ok()?;

    // If we have the complete header, parse it fully.
    if json_end - 8 >= header_size
        && let Ok(header) = serde_json::from_str::<serde_json::Value>(json_str)
    {
        return Some(extract_safetensors_metadata(&header));
    }

    // Partial header — we can still identify format.
    Some(ModelMetadata {
        format: ModelFormat::SafeTensors,
        param_count: None,
        dtype: None,
        tensor_count: None,
        format_version: None,
    })
}

/// Extract metadata from a parsed SafeTensors JSON header.
fn extract_safetensors_metadata(header: &serde_json::Value) -> ModelMetadata {
    let obj = match header.as_object() {
        Some(o) => o,
        None => {
            return ModelMetadata {
                format: ModelFormat::SafeTensors,
                param_count: None,
                dtype: None,
                tensor_count: None,
                format_version: None,
            };
        }
    };

    let mut total_params: u64 = 0;
    let mut tensor_count: u32 = 0;
    let mut dtype = None;

    for (key, value) in obj {
        // Skip metadata key "__metadata__".
        if key == "__metadata__" {
            continue;
        }

        tensor_count = tensor_count.saturating_add(1);

        if let Some(tensor_obj) = value.as_object() {
            // Extract dtype from first tensor.
            if dtype.is_none()
                && let Some(dt) = tensor_obj.get("dtype").and_then(|v| v.as_str())
            {
                dtype = Some(dt.to_string());
            }

            // Count parameters from shape (skip empty shapes — scalars).
            if let Some(shape) = tensor_obj.get("shape").and_then(|v| v.as_array())
                && !shape.is_empty()
            {
                let params: u64 = shape.iter().filter_map(|d| d.as_u64()).product();
                total_params = total_params.saturating_add(params);
            }
        }
    }

    ModelMetadata {
        format: ModelFormat::SafeTensors,
        param_count: if total_params > 0 {
            Some(total_params)
        } else {
            None
        },
        dtype,
        tensor_count: Some(tensor_count),
        format_version: None,
    }
}

// ---------------------------------------------------------------------------
// GGUF parser
// ---------------------------------------------------------------------------

/// Parse GGUF file header.
///
/// Format: 4-byte magic "GGUF", 4-byte version, 8-byte tensor count,
/// 8-byte metadata KV count, then metadata pairs.
fn parse_gguf_header(bytes: &[u8]) -> Option<ModelMetadata> {
    if bytes.len() < 20 {
        return None;
    }

    let magic = u32::from_le_bytes(bytes[..4].try_into().ok()?);
    if magic != GGUF_MAGIC {
        return None;
    }

    let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let _kv_count = u64::from_le_bytes(bytes[16..24].try_into().ok()?);

    // Try to extract dtype from metadata KV pairs.
    // GGUF metadata is complex to parse fully; extract what we can.
    let dtype = extract_gguf_dtype(bytes, 24);

    Some(ModelMetadata {
        format: ModelFormat::GGUF,
        param_count: None, // GGUF doesn't store param count directly.
        dtype,
        tensor_count: if tensor_count <= u32::MAX as u64 {
            Some(tensor_count as u32)
        } else {
            None
        },
        format_version: Some(version),
    })
}

/// Try to extract the general.file_type from GGUF metadata.
///
/// This is a best-effort parser — GGUF KV format requires walking
/// variable-length keys and values. We scan for known patterns.
fn extract_gguf_dtype(bytes: &[u8], offset: usize) -> Option<String> {
    // Scan for "general.file_type" key followed by a u32 value.
    let needle = b"general.file_type";
    let pos = bytes
        .get(offset..)?
        .windows(needle.len())
        .position(|w| w == needle)?;

    // The value type tag and value follow the key.
    // Skip: key_len(8) + key + value_type(4) + value(4)
    let value_offset = offset + pos + needle.len();
    if value_offset + 8 > bytes.len() {
        return None;
    }

    // Value type 4 = UINT32 in GGUF spec.
    let value_type = u32::from_le_bytes(bytes[value_offset..value_offset + 4].try_into().ok()?);
    if value_type != 4 {
        return None;
    }

    let file_type = u32::from_le_bytes(bytes[value_offset + 4..value_offset + 8].try_into().ok()?);

    // Map GGUF file types to human-readable names.
    let name = match file_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        _ => return Some(format!("GGUF_TYPE_{file_type}")),
    };
    Some(name.to_string())
}

// ---------------------------------------------------------------------------
// ONNX parser
// ---------------------------------------------------------------------------

/// Parse ONNX file header.
///
/// ONNX uses Protocol Buffers. The file starts with the ModelProto message.
/// Field 1 (ir_version) is a varint, field 5 (model_version) is a varint.
fn parse_onnx_header(bytes: &[u8]) -> Option<ModelMetadata> {
    if bytes.len() < 4 {
        return None;
    }

    // ONNX starts with protobuf field 1 (ir_version), tag = 0x08.
    if bytes[0] != ONNX_IR_VERSION_TAG {
        return None;
    }

    // Parse ir_version as varint.
    let (ir_version, consumed) = parse_varint(&bytes[1..])?;

    // Sanity: ir_version should be 1-10 (current range).
    if ir_version == 0 || ir_version > 20 {
        return None;
    }

    // Strengthen detection: require a valid second protobuf field after ir_version.
    // ONNX ModelProto field 2 (opset_import) has tag 0x3A (field 7, wire type 2)
    // or field 8 (metadata_props) 0x42, or producer_name (field 2) 0x12.
    // We accept any valid protobuf field tag (wire type 0-2, field 1-15).
    let next_offset = 1 + consumed;
    if next_offset < bytes.len() {
        let next_tag = bytes[next_offset];
        let wire_type = next_tag & 0x07;
        let field_num = next_tag >> 3;
        // Valid protobuf: wire type 0 (varint), 1 (64-bit), 2 (length-delimited)
        // and field number 1-15 (single-byte tags).
        if wire_type > 2 || field_num == 0 {
            return None;
        }
    } else {
        // Only ir_version and nothing else — too short to be a real ONNX model.
        return None;
    }

    Some(ModelMetadata {
        format: ModelFormat::ONNX,
        param_count: None,
        dtype: None,
        tensor_count: None,
        format_version: Some(ir_version as u32),
    })
}

/// Parse a protobuf varint. Returns (value, bytes_consumed).
fn parse_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    for (i, &byte) in bytes.iter().enumerate() {
        if shift >= 64 {
            return None;
        }
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
    }
    None
}

// ---------------------------------------------------------------------------
// PyTorch detector
// ---------------------------------------------------------------------------

/// Check if bytes look like a PyTorch serialized file.
///
/// PyTorch files are ZIP archives containing pickle data.
/// ZIP magic: PK\x03\x04 (0x04034b50 LE).
fn is_pytorch_format(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[..4] == [0x50, 0x4B, 0x03, 0x04]
}

#[cfg(test)]
mod tests {
    use super::*;

    // SafeTensors tests
    #[test]
    fn safetensors_valid_header() {
        let json = r#"{"weight":{"dtype":"F16","shape":[768,768],"data_offsets":[0,1179648]}}"#;
        let header_size = json.len() as u64;
        let mut bytes = header_size.to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());

        let meta = detect_format_from_bytes(&bytes).unwrap();
        assert_eq!(meta.format, ModelFormat::SafeTensors);
        assert_eq!(meta.param_count, Some(768 * 768));
        assert_eq!(meta.dtype.as_deref(), Some("F16"));
        assert_eq!(meta.tensor_count, Some(1));
    }

    #[test]
    fn safetensors_multi_tensor() {
        let json = r#"{"w1":{"dtype":"BF16","shape":[1024,512],"data_offsets":[0,1]},"w2":{"dtype":"BF16","shape":[512,256],"data_offsets":[1,2]},"__metadata__":{"format":"pt"}}"#;
        let header_size = json.len() as u64;
        let mut bytes = header_size.to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());

        let meta = detect_format_from_bytes(&bytes).unwrap();
        assert_eq!(meta.format, ModelFormat::SafeTensors);
        assert_eq!(meta.param_count, Some(1024 * 512 + 512 * 256));
        assert_eq!(meta.dtype.as_deref(), Some("BF16"));
        assert_eq!(meta.tensor_count, Some(2)); // __metadata__ excluded
    }

    #[test]
    fn safetensors_too_small() {
        assert!(detect_format_from_bytes(&[0u8; 4]).is_none());
    }

    #[test]
    fn safetensors_bad_header_size() {
        // Header size says 1 GB — too large to be real.
        let bytes = (1_000_000_000u64).to_le_bytes();
        assert!(parse_safetensors_header(&bytes).is_none());
    }

    // GGUF tests
    #[test]
    fn gguf_valid_header() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // magic
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version 3
        bytes.extend_from_slice(&42u64.to_le_bytes()); // 42 tensors
        bytes.extend_from_slice(&5u64.to_le_bytes()); // 5 KV pairs

        let meta = detect_format_from_bytes(&bytes).unwrap();
        assert_eq!(meta.format, ModelFormat::GGUF);
        assert_eq!(meta.tensor_count, Some(42));
        assert_eq!(meta.format_version, Some(3));
    }

    #[test]
    fn gguf_wrong_magic() {
        let bytes = [0u8; 24];
        assert!(parse_gguf_header(&bytes).is_none());
    }

    #[test]
    fn gguf_too_small() {
        assert!(parse_gguf_header(&[0u8; 10]).is_none());
    }

    // ONNX tests
    #[test]
    fn onnx_valid_header() {
        // ir_version = 9 encoded as varint: 0x08, 0x09
        let bytes = [0x08, 0x09, 0x12, 0x00];
        let meta = detect_format_from_bytes(&bytes).unwrap();
        assert_eq!(meta.format, ModelFormat::ONNX);
        assert_eq!(meta.format_version, Some(9));
    }

    #[test]
    fn onnx_bad_ir_version() {
        // ir_version = 0 — invalid.
        let bytes = [0x08, 0x00];
        assert!(parse_onnx_header(&bytes).is_none());
    }

    // PyTorch tests
    #[test]
    fn pytorch_zip_magic() {
        let bytes = [0x50, 0x4B, 0x03, 0x04, 0x00, 0x00];
        let meta = detect_format_from_bytes(&bytes).unwrap();
        assert_eq!(meta.format, ModelFormat::PyTorch);
    }

    #[test]
    fn pytorch_not_zip() {
        let bytes = [0x00, 0x00, 0x00, 0x00];
        assert!(!is_pytorch_format(&bytes));
    }

    // Format detection priority
    #[test]
    fn unknown_format_returns_none() {
        let bytes = [0xFF, 0xFE, 0xFD, 0xFC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(detect_format_from_bytes(&bytes).is_none());
    }

    // Display
    #[test]
    fn format_display() {
        assert_eq!(ModelFormat::SafeTensors.to_string(), "SafeTensors");
        assert_eq!(ModelFormat::GGUF.to_string(), "GGUF");
        assert_eq!(ModelFormat::ONNX.to_string(), "ONNX");
        assert_eq!(ModelFormat::PyTorch.to_string(), "PyTorch");
    }

    // Serde roundtrip
    #[test]
    fn format_serde_roundtrip() {
        for fmt in [
            ModelFormat::SafeTensors,
            ModelFormat::GGUF,
            ModelFormat::ONNX,
            ModelFormat::PyTorch,
        ] {
            let json = serde_json::to_string(&fmt).unwrap();
            let back: ModelFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(fmt, back);
        }
    }

    // Varint parser
    #[test]
    fn varint_single_byte() {
        assert_eq!(parse_varint(&[0x09]), Some((9, 1)));
    }

    #[test]
    fn varint_multi_byte() {
        // 300 = 0b100101100 → [0xAC, 0x02]
        assert_eq!(parse_varint(&[0xAC, 0x02]), Some((300, 2)));
    }

    #[test]
    fn varint_empty() {
        assert_eq!(parse_varint(&[]), None);
    }

    #[test]
    fn varint_unterminated() {
        // All continuation bits set, never terminates.
        assert_eq!(parse_varint(&[0x80, 0x80, 0x80]), None);
    }

    // Audit edge cases
    #[test]
    fn safetensors_empty_shape_not_counted() {
        // Scalar tensor with empty shape should not inflate param count.
        let json = r#"{"bias":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#;
        let header_size = json.len() as u64;
        let mut bytes = header_size.to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());

        let meta = detect_format_from_bytes(&bytes).unwrap();
        assert_eq!(meta.format, ModelFormat::SafeTensors);
        assert_eq!(meta.param_count, None); // Empty shape → no params counted.
        assert_eq!(meta.tensor_count, Some(1));
    }

    #[test]
    fn onnx_too_short_after_ir_version() {
        // Only ir_version, no second field — should not match.
        let bytes = [0x08, 0x09];
        assert!(parse_onnx_header(&bytes).is_none());
    }

    #[test]
    fn onnx_invalid_second_field() {
        // ir_version=9, then wire type 7 (invalid) at field 0.
        let bytes = [0x08, 0x09, 0x07];
        assert!(parse_onnx_header(&bytes).is_none());
    }

    #[test]
    fn onnx_valid_second_field() {
        // ir_version=9, then field 2 wire type 2 (producer_name, length-delimited).
        let bytes = [0x08, 0x09, 0x12, 0x05, b'o', b'n', b'n', b'x', b'!'];
        let meta = detect_format_from_bytes(&bytes).unwrap();
        assert_eq!(meta.format, ModelFormat::ONNX);
        assert_eq!(meta.format_version, Some(9));
    }

    #[test]
    fn random_0x08_not_onnx() {
        // Arbitrary binary starting with 0x08 should not be detected as ONNX.
        let bytes = [0x08, 0x05, 0xFF, 0xFF]; // wire type 7 = invalid
        assert!(parse_onnx_header(&bytes).is_none());
    }
}
