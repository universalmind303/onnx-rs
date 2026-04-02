use core::fmt;

/// Errors that can occur when parsing ONNX protobuf bytes.
///
/// # Examples
///
/// ```
/// use onnx_rs::{parse, Error};
///
/// // Empty input is a valid (default) model in protobuf
/// let model = parse(&[]).unwrap();
/// assert_eq!(model.ir_version, 0);
///
/// // A truncated varint produces an error
/// let err = parse(&[0x08]).unwrap_err();
/// assert!(err.to_string().contains("unexpected EOF"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    UnexpectedEof {
        offset: usize,
        context: &'static str,
    },
    InvalidVarint {
        offset: usize,
    },
    InvalidWireType {
        value: u8,
        offset: usize,
    },
    InvalidEnumValue {
        enum_name: &'static str,
        value: i32,
    },
    InvalidUtf8 {
        offset: usize,
    },
    TruncatedMessage {
        offset: usize,
        expected: usize,
        available: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnexpectedEof { offset, context } => {
                write!(f, "unexpected EOF at offset {offset} while {context}")
            }
            Error::InvalidVarint { offset } => {
                write!(f, "invalid varint encoding at offset {offset}")
            }
            Error::InvalidWireType { value, offset } => {
                write!(f, "invalid wire type {value} at offset {offset}")
            }
            Error::InvalidEnumValue { enum_name, value } => {
                write!(f, "invalid {enum_name} enum value: {value}")
            }
            Error::InvalidUtf8 { offset } => {
                write!(f, "invalid UTF-8 string at offset {offset}")
            }
            Error::TruncatedMessage {
                offset,
                expected,
                available,
            } => {
                write!(
                    f,
                    "truncated message at offset {offset}: expected {expected} bytes but only {available} available"
                )
            }
        }
    }
}

impl std::error::Error for Error {}
