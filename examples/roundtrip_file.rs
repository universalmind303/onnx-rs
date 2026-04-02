use std::time::Instant;

fn main() {
    let input = std::env::args()
        .nth(1)
        .expect("Usage: roundtrip_file <input.onnx> <output.onnx>");
    let output = std::env::args()
        .nth(2)
        .expect("Usage: roundtrip_file <input.onnx> <output.onnx>");

    let data = std::fs::read(&input).expect("failed to read input");
    eprintln!("Read {} ({:.1} MB)", input, data.len() as f64 / 1_048_576.0);

    let t0 = Instant::now();
    let model = onnx_rs::parse(&data).expect("failed to parse");
    let parse_time = t0.elapsed();

    let g = model.graph.as_ref().unwrap();
    eprintln!(
        "Parsed in {:.1}ms — {} nodes, {} initializers",
        parse_time.as_secs_f64() * 1000.0,
        g.node.len(),
        g.initializer.len()
    );

    let t1 = Instant::now();
    let encoded = onnx_rs::encode(&model);
    let encode_time = t1.elapsed();
    eprintln!(
        "Encoded in {:.1}ms — {:.1} MB",
        encode_time.as_secs_f64() * 1000.0,
        encoded.len() as f64 / 1_048_576.0
    );

    // Verify our own roundtrip first
    let reparsed = onnx_rs::parse(&encoded).expect("failed to re-parse our own output");
    assert_eq!(model, reparsed, "roundtrip mismatch!");
    eprintln!("Self-roundtrip: OK");

    std::fs::write(&output, &encoded).expect("failed to write output");
    eprintln!("Wrote {}", output);
}
