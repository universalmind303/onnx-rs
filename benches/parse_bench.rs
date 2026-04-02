use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use onnx_rs::{encode, parse};
use std::path::PathBuf;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
}

fn load_test_model(name: &str) -> Vec<u8> {
    let path = test_data_dir().join(format!("{name}.onnx"));
    std::fs::read(path).unwrap()
}

fn bench_parse_small_models(c: &mut Criterion) {
    let models = [
        "test_relu",
        "test_add",
        "test_conv_with_strides_padding",
        "test_lstm_defaults",
        "test_if",
    ];

    let mut group = c.benchmark_group("parse/small");
    for name in &models {
        let data = load_test_model(name);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(BenchmarkId::new("parse", name), &data, |b, data| {
            b.iter(|| {
                let model = parse(data).unwrap();
                std::hint::black_box(model);
            });
        });
    }
    group.finish();
}

fn bench_roundtrip_small_models(c: &mut Criterion) {
    let models = [
        "test_relu",
        "test_conv_with_strides_padding",
        "test_if",
    ];

    let mut group = c.benchmark_group("roundtrip/small");
    for name in &models {
        let data = load_test_model(name);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(BenchmarkId::new("roundtrip", name), &data, |b, data| {
            b.iter(|| {
                let model = parse(data).unwrap();
                let encoded = encode(&model);
                std::hint::black_box(encoded);
            });
        });
    }
    group.finish();
}

fn bench_parse_all_test_models(c: &mut Criterion) {
    let dir = test_data_dir();
    let mut all_bytes: Vec<Vec<u8>> = Vec::new();
    let mut total_size: usize = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let entry = entry.unwrap();
        if entry.path().extension().is_some_and(|e| e == "onnx") {
            let data = std::fs::read(entry.path()).unwrap();
            total_size += data.len();
            all_bytes.push(data);
        }
    }

    let mut group = c.benchmark_group("parse/all_test_models");
    group.throughput(Throughput::Bytes(total_size as u64));
    group.bench_function(
        BenchmarkId::new("parse", format!("{}_models", all_bytes.len())),
        |b| {
            b.iter(|| {
                for data in &all_bytes {
                    let model = parse(data).unwrap();
                    std::hint::black_box(model);
                }
            });
        },
    );
    group.finish();
}

fn bench_encode(c: &mut Criterion) {
    let models: Vec<(&str, _)> = ["test_relu", "test_conv_with_strides_padding", "test_if"]
        .iter()
        .map(|name| {
            let data = load_test_model(name);
            let model = parse(&data).unwrap();
            (*name, model)
        })
        .collect();

    let mut group = c.benchmark_group("encode");
    for (name, model) in &models {
        group.bench_with_input(BenchmarkId::new("encode", name), model, |b, model| {
            b.iter(|| {
                let encoded = encode(model);
                std::hint::black_box(encoded);
            });
        });
    }
    group.finish();
}

fn bench_parse_large(c: &mut Criterion) {
    let data = load_test_model("squeezenet");

    let mut group = c.benchmark_group("parse/large");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_with_input(BenchmarkId::new("parse", "squeezenet_5mb"), &data, |b, data| {
        b.iter(|| {
            let model = parse(data).unwrap();
            std::hint::black_box(model);
        });
    });
    group.finish();
}

fn bench_roundtrip_large(c: &mut Criterion) {
    let data = load_test_model("squeezenet");

    let mut group = c.benchmark_group("roundtrip/large");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("roundtrip", "squeezenet_5mb"),
        &data,
        |b, data| {
            b.iter(|| {
                let model = parse(data).unwrap();
                let encoded = encode(&model);
                std::hint::black_box(encoded);
            });
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_parse_small_models,
    bench_roundtrip_small_models,
    bench_parse_all_test_models,
    bench_encode,
    bench_parse_large,
    bench_roundtrip_large,
);
criterion_main!(benches);
