# onnx-rs

Zero-dependency ONNX model parser in Rust. Parses `.onnx` files (protobuf wire format) directly into typed Rust AST structs without relying on `prost`, `protobuf`, or any other crate.

## Architecture

- `src/wire.rs` — Low-level protobuf wire format decoder (`Cursor`). Handles varints, fixed-width types, length-delimited fields, and tag parsing. Internal only (`pub(crate)`).
- `src/parser.rs` — Recursive-descent parser that walks the wire-format bytes and builds AST types. One `parse_*` function per ONNX protobuf message type.
- `src/ast.rs` — Typed Rust structs mirroring the ONNX protobuf schema (`Model`, `Graph`, `Node`, `TensorProto`, etc.). Includes `OpType` enum with all standard ONNX operators.
- `src/error.rs` — Error types for parse failures (EOF, invalid varint, bad wire type, truncated messages, etc.).
- `src/lib.rs` — Public API surface: re-exports `Model`, `Error`, and `parse()`.

## Build & Test

```
cargo build
cargo test
```

## Development Process

- TDD everything. Write tests first, then implement to make them pass.

## Conventions

- Edition 2024.
- No external dependencies — `[dependencies]` must stay empty.
- Field numbers in `parse_*` match functions correspond to the ONNX protobuf field numbers from the `.proto` spec.
- Unknown fields are silently skipped via `skip_field` to stay forward-compatible with newer ONNX versions.
- Packed repeated fields are handled by checking `WireType::LengthDelimited` vs the scalar wire type.
