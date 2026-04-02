//! Zero-dependency ONNX model parser and encoder for Rust.
//!
//! Parses `.onnx` files (protobuf wire format) directly into typed Rust
//! AST structs without relying on `prost`, `protobuf`, or any other crate.
//!
//! # Quick start
//!
//! ```
//! use onnx_rs::ast::*;
//!
//! // Build a minimal ONNX model
//! let model = Model {
//!     ir_version: 9,
//!     producer_name: "example",
//!     opset_import: vec![OperatorSetId { domain: "", version: 19 }],
//!     graph: Some(Graph {
//!         name: "main",
//!         node: vec![Node {
//!             op_type: OpType::Relu,
//!             input: vec!["X"],
//!             output: vec!["Y"],
//!             ..Default::default()
//!         }],
//!         ..Default::default()
//!     }),
//!     ..Default::default()
//! };
//!
//! // Encode to protobuf bytes (suitable for writing to a .onnx file)
//! let bytes = onnx_rs::encode(&model);
//!
//! // Parse the bytes back into a Model
//! let parsed = onnx_rs::parse(&bytes).unwrap();
//! assert_eq!(parsed.ir_version, 9);
//! assert_eq!(parsed.producer_name, "example");
//!
//! let graph = parsed.graph.as_ref().unwrap();
//! assert_eq!(graph.node[0].op_type, OpType::Relu);
//! ```
//!
//! # Reading a .onnx file from disk
//!
//! ```no_run
//! let bytes = std::fs::read("model.onnx").unwrap();
//! let model = onnx_rs::parse(&bytes).unwrap();
//! println!("producer: {}", model.producer_name);
//! ```

pub mod ast;
pub mod encoder;
pub mod error;
pub mod parser;
mod wire;

pub use ast::Model;
pub use encoder::encode;
pub use error::Error;
pub use parser::parse;
