# onnx-rs

Zero-dependency ONNX model parser in Rust. Parses `.onnx` files (protobuf wire format) directly into typed Rust structs without relying on `prost`, `protobuf`, or any other crate.

## Usage

```rust
let bytes = std::fs::read("model.onnx").unwrap();
let model = onnx_rs::parse(&bytes).unwrap();
```

## License

MIT
