use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: bench <model.onnx>");

    let data = std::fs::read(&path).expect("failed to read file");
    println!("File: {} ({:.1} MB)", path, data.len() as f64 / 1_000_000.0);

    // Warm up
    let _ = onnx_rs::parse(&data).unwrap();

    let mut times = Vec::with_capacity(100);
    for _ in 0..100 {
        let t0 = Instant::now();
        let model = onnx_rs::parse(&data).unwrap();
        let elapsed = t0.elapsed();
        times.push(elapsed);
        std::hint::black_box(model);
    }

    times.sort();
    let avg: f64 = times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / times.len() as f64;
    let best = times[0].as_secs_f64();
    let p50 = times[times.len() / 2].as_secs_f64();
    let p99 = times[times.len() * 99 / 100].as_secs_f64();

    println!("onnx-rs parse (n=100):");
    println!("  avg:  {:.1}ms", avg * 1000.0);
    println!("  best: {:.1}ms", best * 1000.0);
    println!("  p50:  {:.1}ms", p50 * 1000.0);
    println!("  p99:  {:.1}ms", p99 * 1000.0);
}
