# onnx-rs

Zero-dependency ONNX model parser and encoder in Rust. Parses `.onnx` files (protobuf wire format) directly into typed Rust structs without relying on `prost`, `protobuf`, or any other crate.

## Usage

```rust
use onnx_rs::ast::*;

// Parse
let bytes = std::fs::read("model.onnx").unwrap();
let model = onnx_rs::parse(&bytes).unwrap();

let graph = model.graph.as_ref().unwrap();
for node in &graph.node {
    match &node.op_type {
        OpType::Conv => println!("found conv: {}", node.name),
        OpType::Relu => println!("found relu: {}", node.name),
        op => println!("{}: {}", op, node.name),
    }
}

// Encode back to protobuf
let bytes = onnx_rs::encode(&model);
std::fs::write("output.onnx", &bytes).unwrap();
```

## Features

- **Zero dependencies** — hand-rolled protobuf wire format decoder
- **Typed AST** — `OpType` enum with 170+ standard ONNX operators, `Custom(String)` fallback for vendor ops
- **Full ONNX IR** — `Model`, `Graph`, `Node`, `TensorProto`, `TypeProto` (Tensor/Sequence/Map/Optional/SparseTensor), `Attribute`, `Function`, `TrainingInfo`
- **All 23 data types** — including Float8 (e4m3fn, e5m2, etc.), Int4, Uint4, BFloat16
- **Roundtrip encode/decode** — `parse()` and `encode()` are fully symmetric

## Benchmarks

Measured on Apple Silicon. `onnx-rs` vs the official C++ protobuf implementation (via Python `onnx` package). Parse only — file I/O excluded.

| Model | Size | onnx-rs | C++ protobuf | Speedup |
|-------|------|---------|-------------|---------|
| NLLB Decoder | 1.7 GB | **0.4ms** | 55.3ms | **138x** |
| NLLB Encoder | 1.5 GB | **0.2ms** | 29.8ms | **149x** |
| BGE Large | 1.2 GB | **0.3ms** | 55.7ms | **186x** |
| VGG-19 | 548 MB | 8.2ms | 8.5ms | 1.0x |
| GPT-2 | 523 MB | **0.3ms** | 8.3ms | **28x** |
| Swin Transformer | 422 MB | **2.6ms** | 8.2ms | **3.1x** |
| BERT-SQuAD | 415 MB | **0.1ms** | 12.6ms | **126x** |
| Donut Encoder | 297 MB | **2.2ms** | 6.1ms | **2.8x** |
| YOLOv4 | 245 MB | **0.1ms** | 4.0ms | **40x** |
| ResNet-152 | 230 MB | 3.9ms | 4.0ms | 1.0x |
| Mask R-CNN | 169 MB | **0.9ms** | 3.7ms | **4.1x** |
| Faster R-CNN | 168 MB | **0.7ms** | 3.5ms | **5.0x** |
| MiniLM-L6 | 86 MB | **0.1ms** | 1.5ms | **15x** |
| EfficientNet | 49 MB | **0.1ms** | 0.9ms | **9x** |
| Inception v2 | 42 MB | **0.1ms** | 0.9ms | **9x** |

Models using `raw_data` (the default for modern exporters like PyTorch) are parsed near-instantly via zero-copy borrows. Models using `float_data` (legacy packed floats) use bulk `memcpy` on little-endian platforms.

All 15 models roundtrip cleanly: `onnx_rs::parse` -> `onnx_rs::encode` -> validated by `onnx.checker.check_model()` in the official C++ implementation.

Run benchmarks locally:

```sh
cargo bench
```

## License

MIT
