use onnx_rs::ast::*;
use onnx_rs::{encode, parse};
use std::path::PathBuf;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
}

fn parse_model_file(name: &str) -> Model {
    let path = test_data_dir().join(format!("{name}.onnx"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    parse(&bytes).unwrap_or_else(|e| panic!("failed to parse {name}: {e}"))
}

fn roundtrip_model_file(name: &str) {
    let path = test_data_dir().join(format!("{name}.onnx"));
    let original_bytes = std::fs::read(&path).unwrap();
    let model = parse(&original_bytes).unwrap_or_else(|e| panic!("failed to parse {name}: {e}"));
    let re_encoded = encode(&model);
    let reparsed = parse(&re_encoded).unwrap_or_else(|e| panic!("failed to re-parse {name}: {e}"));
    assert_eq!(model, reparsed, "roundtrip failed for {name}");
}

// ============================================================
// Parse every model in tests/data/ — bulk test
// ============================================================

#[test]
fn test_parse_all_official_models() {
    let dir = test_data_dir();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "onnx"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    assert!(entries.len() >= 100, "expected at least 100 test models, found {}", entries.len());

    let mut failures = Vec::new();
    for entry in &entries {
        let name = entry.path().file_stem().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(entry.path()).unwrap();
        match parse(&bytes) {
            Ok(_) => {}
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} models failed to parse:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }
}

#[test]
fn test_roundtrip_all_official_models() {
    let dir = test_data_dir();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "onnx"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut failures = Vec::new();
    for entry in &entries {
        let name = entry.path().file_stem().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(entry.path()).unwrap();
        match parse(&bytes) {
            Ok(model) => {
                let re_encoded = encode(&model);
                match parse(&re_encoded) {
                    Ok(reparsed) => {
                        if model != reparsed {
                            failures.push(format!("{name}: roundtrip mismatch"));
                        }
                    }
                    Err(e) => failures.push(format!("{name}: re-parse failed: {e}")),
                }
            }
            Err(e) => failures.push(format!("{name}: parse failed: {e}")),
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} models failed roundtrip:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }
}

// ============================================================
// Detailed tests for specific models — verify parsed content
// ============================================================

#[test]
fn test_official_relu() {
    let model = parse_model_file("test_relu");
    assert!(model.ir_version > 0);
    assert!(!model.opset_import.is_empty());
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node.len(), 1);
    assert_eq!(g.node[0].op_type, OpType::Relu);
    assert_eq!(g.node[0].input.len(), 1);
    assert_eq!(g.node[0].output.len(), 1);
    roundtrip_model_file("test_relu");
}

#[test]
fn test_official_add() {
    let model = parse_model_file("test_add");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node.len(), 1);
    assert_eq!(g.node[0].op_type, OpType::Add);
    assert_eq!(g.node[0].input.len(), 2);
    roundtrip_model_file("test_add");
}

#[test]
fn test_official_conv() {
    let model = parse_model_file("test_basic_conv_without_padding");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Conv);
    roundtrip_model_file("test_basic_conv_without_padding");
}

#[test]
fn test_official_conv_strides() {
    let model = parse_model_file("test_conv_with_strides_padding");
    let g = model.graph.as_ref().unwrap();
    let node = &g.node[0];
    assert_eq!(node.op_type, OpType::Conv);
    let strides = node.attribute.iter().find(|a| a.name == "strides").unwrap();
    assert!(!strides.ints.is_empty());
    let kernel = node.attribute.iter().find(|a| a.name == "kernel_shape").unwrap();
    assert!(!kernel.ints.is_empty());
    roundtrip_model_file("test_conv_with_strides_padding");
}

#[test]
fn test_official_batchnorm() {
    let model = parse_model_file("test_batchnorm_epsilon");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::BatchNormalization);
    let epsilon = g.node[0].attribute.iter().find(|a| a.name == "epsilon").unwrap();
    assert!(epsilon.f > 0.0);
    roundtrip_model_file("test_batchnorm_epsilon");
}

#[test]
fn test_official_concat() {
    let model = parse_model_file("test_concat_1d_axis_0");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Concat);
    let axis = g.node[0].attribute.iter().find(|a| a.name == "axis").unwrap();
    assert_eq!(axis.i, 0);
    roundtrip_model_file("test_concat_1d_axis_0");
}

#[test]
fn test_official_constant() {
    let model = parse_model_file("test_constant");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Constant);
    let value_attr = g.node[0].attribute.iter().find(|a| a.name == "value").unwrap();
    assert!(value_attr.t.is_some());
    roundtrip_model_file("test_constant");
}

#[test]
fn test_official_gemm() {
    let model = parse_model_file("test_gemm_default_no_bias");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Gemm);
    roundtrip_model_file("test_gemm_default_no_bias");
}

#[test]
fn test_official_matmul() {
    let model = parse_model_file("test_matmul_2d");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::MatMul);
    roundtrip_model_file("test_matmul_2d");
}

#[test]
fn test_official_lstm() {
    let model = parse_model_file("test_lstm_defaults");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::LSTM);
    roundtrip_model_file("test_lstm_defaults");
}

#[test]
fn test_official_gru() {
    let model = parse_model_file("test_gru_defaults");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::GRU);
    roundtrip_model_file("test_gru_defaults");
}

#[test]
fn test_official_rnn() {
    let model = parse_model_file("test_rnn_seq_length");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::RNN);
    roundtrip_model_file("test_rnn_seq_length");
}

#[test]
fn test_official_if() {
    let model = parse_model_file("test_if");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::If);
    let then_attr = g.node[0].attribute.iter().find(|a| a.name == "then_branch").unwrap();
    assert!(then_attr.g.is_some());
    let else_attr = g.node[0].attribute.iter().find(|a| a.name == "else_branch").unwrap();
    assert!(else_attr.g.is_some());
    roundtrip_model_file("test_if");
}

#[test]
fn test_official_loop() {
    let model = parse_model_file("test_loop11");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Loop);
    let body_attr = g.node[0].attribute.iter().find(|a| a.name == "body").unwrap();
    assert!(body_attr.g.is_some());
    roundtrip_model_file("test_loop11");
}

#[test]
fn test_official_scan() {
    let model = parse_model_file("test_scan_sum");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Scan);
    let body_attr = g.node[0].attribute.iter().find(|a| a.name == "body").unwrap();
    assert!(body_attr.g.is_some());
    roundtrip_model_file("test_scan_sum");
}

#[test]
fn test_official_reshape() {
    let model = parse_model_file("test_reshape_extended_dims");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Reshape);
    roundtrip_model_file("test_reshape_extended_dims");
}

#[test]
fn test_official_transpose() {
    let model = parse_model_file("test_transpose_default");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Transpose);
    roundtrip_model_file("test_transpose_default");
}

#[test]
fn test_official_softmax() {
    let model = parse_model_file("test_softmax_axis_0");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Softmax);
    roundtrip_model_file("test_softmax_axis_0");
}

#[test]
fn test_official_maxpool() {
    let model = parse_model_file("test_maxpool_2d_default");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::MaxPool);
    roundtrip_model_file("test_maxpool_2d_default");
}

#[test]
fn test_official_averagepool() {
    let model = parse_model_file("test_averagepool_2d_default");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::AveragePool);
    roundtrip_model_file("test_averagepool_2d_default");
}

#[test]
fn test_official_flatten() {
    let model = parse_model_file("test_flatten_axis0");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Flatten);
    roundtrip_model_file("test_flatten_axis0");
}

#[test]
fn test_official_gather() {
    let model = parse_model_file("test_gather_0");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Gather);
    roundtrip_model_file("test_gather_0");
}

#[test]
fn test_official_scatter_elements() {
    let model = parse_model_file("test_scatter_elements_with_axis");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::ScatterElements);
    roundtrip_model_file("test_scatter_elements_with_axis");
}

#[test]
fn test_official_reduce_mean() {
    let model = parse_model_file("test_reduce_mean_default_axes_keepdims_example");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::ReduceMean);
    roundtrip_model_file("test_reduce_mean_default_axes_keepdims_example");
}

#[test]
fn test_official_dropout() {
    let model = parse_model_file("test_dropout_default");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Dropout);
    roundtrip_model_file("test_dropout_default");
}

#[test]
fn test_official_cast() {
    let model = parse_model_file("test_cast_FLOAT_to_DOUBLE");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Cast);
    roundtrip_model_file("test_cast_FLOAT_to_DOUBLE");
}

#[test]
fn test_official_where() {
    let model = parse_model_file("test_where_example");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Where);
    roundtrip_model_file("test_where_example");
}

#[test]
fn test_official_resize() {
    let model = parse_model_file("test_resize_upsample_scales_nearest");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Resize);
    roundtrip_model_file("test_resize_upsample_scales_nearest");
}

#[test]
fn test_official_quantize_linear() {
    let model = parse_model_file("test_quantizelinear");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::QuantizeLinear);
    roundtrip_model_file("test_quantizelinear");
}

#[test]
fn test_official_dequantize_linear() {
    let model = parse_model_file("test_dequantizelinear");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::DequantizeLinear);
    roundtrip_model_file("test_dequantizelinear");
}

#[test]
fn test_official_convtranspose() {
    let model = parse_model_file("test_convtranspose");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::ConvTranspose);
    roundtrip_model_file("test_convtranspose");
}

#[test]
fn test_official_slice() {
    let model = parse_model_file("test_slice");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Slice);
    roundtrip_model_file("test_slice");
}

#[test]
fn test_official_pad() {
    let path = test_data_dir().join("test_pad_constant.onnx");
    if !path.exists() {
        return;
    }
    let model = parse_model_file("test_pad_constant");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Pad);
    roundtrip_model_file("test_pad_constant");
}

#[test]
fn test_official_split() {
    let path = test_data_dir().join("test_split_equal_parts_1d.onnx");
    if !path.exists() {
        return; // model not available
    }
    let model = parse_model_file("test_split_equal_parts_1d");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Split);
    roundtrip_model_file("test_split_equal_parts_1d");
}

#[test]
fn test_official_erf() {
    let model = parse_model_file("test_erf");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Erf);
    roundtrip_model_file("test_erf");
}

#[test]
fn test_official_gelu() {
    let model = parse_model_file("test_gelu_default_1");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::Gelu);
    roundtrip_model_file("test_gelu_default_1");
}

#[test]
fn test_official_layer_norm() {
    let path = test_data_dir().join("test_layernormalization_2d_axis0.onnx");
    if !path.exists() {
        return;
    }
    let model = parse_model_file("test_layernormalization_2d_axis0");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::LayerNormalization);
    roundtrip_model_file("test_layernormalization_2d_axis0");
}

#[test]
fn test_official_optional_has_element() {
    let path = test_data_dir().join("test_optional_has_element.onnx");
    if !path.exists() {
        return;
    }
    let model = parse_model_file("test_optional_has_element");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::OptionalHasElement);
    roundtrip_model_file("test_optional_has_element");
}

#[test]
fn test_official_optional_get_element() {
    let model = parse_model_file("test_optional_get_element_sequence");
    let g = model.graph.as_ref().unwrap();
    assert_eq!(g.node[0].op_type, OpType::OptionalGetElement);
    roundtrip_model_file("test_optional_get_element_sequence");
}

#[test]
fn test_official_sequence() {
    let model = parse_model_file("test_sequence_model1");
    let g = model.graph.as_ref().unwrap();
    assert!(!g.node.is_empty());
    roundtrip_model_file("test_sequence_model1");
}

#[test]
fn test_official_single_relu_model() {
    let model = parse_model_file("test_single_relu_model");
    let g = model.graph.as_ref().unwrap();
    assert!(!g.node.is_empty());
    assert!(!g.input.is_empty());
    assert!(!g.output.is_empty());
    roundtrip_model_file("test_single_relu_model");
}

// Trig ops
#[test]
fn test_official_trig_ops() {
    for (name, expected) in [
        ("test_sin_example", OpType::Sin),
        ("test_cos_example", OpType::Cos),
        ("test_tan_example", OpType::Tan),
        ("test_asin", OpType::Asin),
        ("test_acos", OpType::Acos),
        ("test_atan", OpType::Atan),
        ("test_sinh_example", OpType::Sinh),
        ("test_cosh_example", OpType::Cosh),
        ("test_asinh", OpType::Asinh),
        ("test_acosh", OpType::Acosh),
        ("test_atanh", OpType::Atanh),
    ] {
        let model = parse_model_file(name);
        let g = model.graph.as_ref().unwrap();
        assert_eq!(g.node[0].op_type, expected, "op mismatch for {name}");
        roundtrip_model_file(name);
    }
}

// Math ops
#[test]
fn test_official_math_ops() {
    for (name, expected) in [
        ("test_abs", OpType::Abs),
        ("test_add", OpType::Add),
        ("test_sub", OpType::Sub),
        ("test_mul", OpType::Mul),
        ("test_div", OpType::Div),
        ("test_neg_example", OpType::Neg),
        ("test_sqrt_example", OpType::Sqrt),
        ("test_exp_example", OpType::Exp),
        ("test_log_example", OpType::Log),
        ("test_pow", OpType::Pow),
        ("test_ceil_example", OpType::Ceil),
        ("test_floor_example", OpType::Floor),
        ("test_round", OpType::Round),
        ("test_reciprocal_example", OpType::Reciprocal),
        ("test_sign", OpType::Sign),
    ] {
        let model = parse_model_file(name);
        let g = model.graph.as_ref().unwrap();
        assert_eq!(g.node[0].op_type, expected, "op mismatch for {name}");
        roundtrip_model_file(name);
    }
}

// Activation ops
#[test]
fn test_official_activation_ops() {
    for (name, expected) in [
        ("test_relu", OpType::Relu),
        ("test_sigmoid_example", OpType::Sigmoid),
        ("test_tanh_example", OpType::Tanh),
        ("test_leakyrelu_example", OpType::LeakyRelu),
        ("test_elu_example", OpType::Elu),
        ("test_selu_example", OpType::Selu),
        ("test_hardsigmoid_example", OpType::HardSigmoid),
        ("test_hardswish", OpType::HardSwish),
        ("test_softplus_example", OpType::Softplus),
        ("test_softsign_example", OpType::Softsign),
        ("test_thresholdedrelu_example", OpType::ThresholdedRelu),
        ("test_mish", OpType::Mish),
    ] {
        let model = parse_model_file(name);
        let g = model.graph.as_ref().unwrap();
        assert_eq!(g.node[0].op_type, expected, "op mismatch for {name}");
        roundtrip_model_file(name);
    }
}

// Comparison / logic ops
#[test]
fn test_official_comparison_ops() {
    for (name, expected) in [
        ("test_equal", OpType::Equal),
        ("test_greater", OpType::Greater),
        ("test_greater_equal", OpType::GreaterOrEqual),
        ("test_less", OpType::Less),
        ("test_less_equal", OpType::LessOrEqual),
        ("test_not_2d", OpType::Not),
        ("test_and_bcast3v1d", OpType::And),
        ("test_or_bcast3v1d", OpType::Or),
        ("test_xor_bcast3v1d", OpType::Xor),
        ("test_isinf", OpType::IsInf),
    ] {
        let model = parse_model_file(name);
        let g = model.graph.as_ref().unwrap();
        assert_eq!(g.node[0].op_type, expected, "op mismatch for {name}");
        roundtrip_model_file(name);
    }
}

// Reduction ops
#[test]
fn test_official_reduce_ops() {
    for (name, expected) in [
        ("test_reduce_max_default_axes_keepdim_example", OpType::ReduceMax),
        ("test_reduce_min_default_axes_keepdims_example", OpType::ReduceMin),
        ("test_reduce_mean_default_axes_keepdims_example", OpType::ReduceMean),
        ("test_reduce_sum_default_axes_keepdims_example", OpType::ReduceSum),
        ("test_reduce_prod_default_axes_keepdims_example", OpType::ReduceProd),
        ("test_reduce_l1_default_axes_keepdims_example", OpType::ReduceL1),
        ("test_reduce_l2_default_axes_keepdims_example", OpType::ReduceL2),
        ("test_reduce_log_sum_asc_axes", OpType::ReduceLogSum),
        ("test_reduce_log_sum_exp_default_axes_keepdims_example", OpType::ReduceLogSumExp),
        ("test_reduce_sum_square_default_axes_keepdims_example", OpType::ReduceSumSquare),
        ("test_argmax_default_axis_example", OpType::ArgMax),
        ("test_argmin_default_axis_example", OpType::ArgMin),
    ] {
        let model = parse_model_file(name);
        let g = model.graph.as_ref().unwrap();
        assert_eq!(g.node[0].op_type, expected, "op mismatch for {name}");
        roundtrip_model_file(name);
    }
}
