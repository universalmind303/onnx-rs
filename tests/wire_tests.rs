use onnx_rs::error::Error;

// We need access to wire internals for testing.
// Since wire is private, we'll test it through a test helper in lib.
// Actually, let's make wire pub(crate) and add unit tests inside wire.rs itself.
// For now, these integration tests focus on the error module and public API.

#[test]
fn test_error_display_unexpected_eof() {
    let err = Error::UnexpectedEof {
        offset: 42,
        context: "reading varint",
    };
    let msg = format!("{err}");
    assert!(msg.contains("42"), "should mention offset");
    assert!(msg.contains("varint"), "should mention context");
}

#[test]
fn test_error_display_invalid_varint() {
    let err = Error::InvalidVarint { offset: 10 };
    let msg = format!("{err}");
    assert!(msg.contains("10"));
}

#[test]
fn test_error_display_invalid_wire_type() {
    let err = Error::InvalidWireType {
        value: 3,
        offset: 5,
    };
    let msg = format!("{err}");
    assert!(msg.contains("3"));
}

#[test]
fn test_error_display_invalid_enum_value() {
    let err = Error::InvalidEnumValue {
        enum_name: "DataType",
        value: 99,
    };
    let msg = format!("{err}");
    assert!(msg.contains("DataType"));
    assert!(msg.contains("99"));
}

#[test]
fn test_error_display_invalid_utf8() {
    let err = Error::InvalidUtf8 { offset: 7 };
    let msg = format!("{err}");
    assert!(msg.contains("UTF-8") || msg.contains("utf8") || msg.contains("Utf8"));
}

#[test]
fn test_error_display_truncated_message() {
    let err = Error::TruncatedMessage {
        offset: 0,
        expected: 100,
        available: 50,
    };
    let msg = format!("{err}");
    assert!(msg.contains("100"));
    assert!(msg.contains("50"));
}

#[test]
fn test_error_implements_std_error() {
    let err = Error::InvalidVarint { offset: 0 };
    // This just needs to compile — proves std::error::Error is implemented
    let _: &dyn std::error::Error = &err;
}
