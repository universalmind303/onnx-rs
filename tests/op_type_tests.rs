use onnx_rs::ast::*;
use onnx_rs::parse;

// === Helper functions (same as parser_tests) ===

fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
    buf
}

fn encode_tag(field_number: u32, wire_type: u8) -> Vec<u8> {
    encode_varint(((field_number as u64) << 3) | wire_type as u64)
}

fn encode_length_delimited(field_number: u32, data: &[u8]) -> Vec<u8> {
    let mut buf = encode_tag(field_number, 2);
    buf.extend(encode_varint(data.len() as u64));
    buf.extend(data);
    buf
}

fn encode_string_field(field_number: u32, s: &str) -> Vec<u8> {
    encode_length_delimited(field_number, s.as_bytes())
}

fn make_model_with_op(op_type: &str) -> Vec<u8> {
    let mut node = Vec::new();
    node.extend(encode_string_field(1, "x"));
    node.extend(encode_string_field(2, "y"));
    node.extend(encode_string_field(4, op_type));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(1, &node));

    encode_length_delimited(7, &graph)
}

macro_rules! assert_parse_op {
    ($op_str:expr, $expected:expr) => {{
        let data = make_model_with_op($op_str);
        let model = parse(&data).unwrap();
        assert_eq!(model.graph.as_ref().unwrap().node[0].op_type, $expected);
    }};
}

// === OpType enum basics ===

#[test]
fn test_op_type_debug_clone_eq() {
    let op = OpType::Relu;
    let cloned = op.clone();
    assert_eq!(op, cloned);
    let _ = format!("{:?}", op);
}

#[test]
fn test_op_type_custom_variant() {
    let op = OpType::Custom("MyCustomOp");
    assert_eq!(op, OpType::Custom("MyCustomOp"));
    assert_ne!(op, OpType::Relu);
}

#[test]
fn test_op_type_display() {
    assert_eq!(OpType::Relu.to_string(), "Relu");
    assert_eq!(OpType::Conv.to_string(), "Conv");
    assert_eq!(OpType::BatchNormalization.to_string(), "BatchNormalization");
    assert_eq!(OpType::Custom("Foo").to_string(), "Foo");
}

#[test]
fn test_op_type_from_str() {
    assert_eq!(OpType::from("Relu"), OpType::Relu);
    assert_eq!(OpType::from("Conv"), OpType::Conv);
    assert_eq!(OpType::from("UnknownThing"), OpType::Custom("UnknownThing"));
}

// === Parsing known ops ===

#[test]
fn test_parse_activations() {
    assert_parse_op!("Relu", OpType::Relu);
    assert_parse_op!("LeakyRelu", OpType::LeakyRelu);
    assert_parse_op!("Sigmoid", OpType::Sigmoid);
    assert_parse_op!("Tanh", OpType::Tanh);
    assert_parse_op!("Softmax", OpType::Softmax);
    assert_parse_op!("LogSoftmax", OpType::LogSoftmax);
    assert_parse_op!("Elu", OpType::Elu);
    assert_parse_op!("Selu", OpType::Selu);
    assert_parse_op!("PRelu", OpType::PRelu);
    assert_parse_op!("Gelu", OpType::Gelu);
    assert_parse_op!("HardSigmoid", OpType::HardSigmoid);
    assert_parse_op!("HardSwish", OpType::HardSwish);
    assert_parse_op!("Softplus", OpType::Softplus);
    assert_parse_op!("Softsign", OpType::Softsign);
    assert_parse_op!("ThresholdedRelu", OpType::ThresholdedRelu);
    assert_parse_op!("Celu", OpType::Celu);
    assert_parse_op!("Mish", OpType::Mish);
}

#[test]
fn test_parse_math_ops() {
    assert_parse_op!("Add", OpType::Add);
    assert_parse_op!("Sub", OpType::Sub);
    assert_parse_op!("Mul", OpType::Mul);
    assert_parse_op!("Div", OpType::Div);
    assert_parse_op!("Neg", OpType::Neg);
    assert_parse_op!("Abs", OpType::Abs);
    assert_parse_op!("Sqrt", OpType::Sqrt);
    assert_parse_op!("Exp", OpType::Exp);
    assert_parse_op!("Log", OpType::Log);
    assert_parse_op!("Pow", OpType::Pow);
    assert_parse_op!("Ceil", OpType::Ceil);
    assert_parse_op!("Floor", OpType::Floor);
    assert_parse_op!("Clip", OpType::Clip);
    assert_parse_op!("Sign", OpType::Sign);
    assert_parse_op!("Reciprocal", OpType::Reciprocal);
    assert_parse_op!("Mod", OpType::Mod);
    assert_parse_op!("BitShift", OpType::BitShift);
    assert_parse_op!("BitwiseAnd", OpType::BitwiseAnd);
    assert_parse_op!("BitwiseOr", OpType::BitwiseOr);
    assert_parse_op!("BitwiseXor", OpType::BitwiseXor);
    assert_parse_op!("BitwiseNot", OpType::BitwiseNot);
}

#[test]
fn test_parse_trig_ops() {
    assert_parse_op!("Sin", OpType::Sin);
    assert_parse_op!("Cos", OpType::Cos);
    assert_parse_op!("Tan", OpType::Tan);
    assert_parse_op!("Asin", OpType::Asin);
    assert_parse_op!("Acos", OpType::Acos);
    assert_parse_op!("Atan", OpType::Atan);
    assert_parse_op!("Sinh", OpType::Sinh);
    assert_parse_op!("Cosh", OpType::Cosh);
    assert_parse_op!("Asinh", OpType::Asinh);
    assert_parse_op!("Acosh", OpType::Acosh);
    assert_parse_op!("Atanh", OpType::Atanh);
}

#[test]
fn test_parse_conv_pool_ops() {
    assert_parse_op!("Conv", OpType::Conv);
    assert_parse_op!("ConvTranspose", OpType::ConvTranspose);
    assert_parse_op!("ConvInteger", OpType::ConvInteger);
    assert_parse_op!("AveragePool", OpType::AveragePool);
    assert_parse_op!("MaxPool", OpType::MaxPool);
    assert_parse_op!("GlobalAveragePool", OpType::GlobalAveragePool);
    assert_parse_op!("GlobalMaxPool", OpType::GlobalMaxPool);
    assert_parse_op!("GlobalLpPool", OpType::GlobalLpPool);
    assert_parse_op!("LpPool", OpType::LpPool);
    assert_parse_op!("MaxRoiPool", OpType::MaxRoiPool);
    assert_parse_op!("MaxUnpool", OpType::MaxUnpool);
}

#[test]
fn test_parse_normalization_ops() {
    assert_parse_op!("BatchNormalization", OpType::BatchNormalization);
    assert_parse_op!("InstanceNormalization", OpType::InstanceNormalization);
    assert_parse_op!("LayerNormalization", OpType::LayerNormalization);
    assert_parse_op!("GroupNormalization", OpType::GroupNormalization);
    assert_parse_op!("LpNormalization", OpType::LpNormalization);
    assert_parse_op!("MeanVarianceNormalization", OpType::MeanVarianceNormalization);
}

#[test]
fn test_parse_linear_ops() {
    assert_parse_op!("MatMul", OpType::MatMul);
    assert_parse_op!("MatMulInteger", OpType::MatMulInteger);
    assert_parse_op!("Gemm", OpType::Gemm);
}

#[test]
fn test_parse_tensor_manipulation_ops() {
    assert_parse_op!("Reshape", OpType::Reshape);
    assert_parse_op!("Transpose", OpType::Transpose);
    assert_parse_op!("Flatten", OpType::Flatten);
    assert_parse_op!("Squeeze", OpType::Squeeze);
    assert_parse_op!("Unsqueeze", OpType::Unsqueeze);
    assert_parse_op!("Expand", OpType::Expand);
    assert_parse_op!("Tile", OpType::Tile);
    assert_parse_op!("Pad", OpType::Pad);
    assert_parse_op!("Slice", OpType::Slice);
    assert_parse_op!("Split", OpType::Split);
    assert_parse_op!("SplitToSequence", OpType::SplitToSequence);
    assert_parse_op!("Concat", OpType::Concat);
    assert_parse_op!("ConcatFromSequence", OpType::ConcatFromSequence);
    assert_parse_op!("DepthToSpace", OpType::DepthToSpace);
    assert_parse_op!("SpaceToDepth", OpType::SpaceToDepth);
    assert_parse_op!("ReverseSequence", OpType::ReverseSequence);
}

#[test]
fn test_parse_gather_scatter_ops() {
    assert_parse_op!("Gather", OpType::Gather);
    assert_parse_op!("GatherElements", OpType::GatherElements);
    assert_parse_op!("GatherND", OpType::GatherND);
    assert_parse_op!("Scatter", OpType::Scatter);
    assert_parse_op!("ScatterElements", OpType::ScatterElements);
    assert_parse_op!("ScatterND", OpType::ScatterND);
}

#[test]
fn test_parse_reduction_ops() {
    assert_parse_op!("ReduceMax", OpType::ReduceMax);
    assert_parse_op!("ReduceMin", OpType::ReduceMin);
    assert_parse_op!("ReduceMean", OpType::ReduceMean);
    assert_parse_op!("ReduceSum", OpType::ReduceSum);
    assert_parse_op!("ReduceProd", OpType::ReduceProd);
    assert_parse_op!("ReduceL1", OpType::ReduceL1);
    assert_parse_op!("ReduceL2", OpType::ReduceL2);
    assert_parse_op!("ReduceLogSum", OpType::ReduceLogSum);
    assert_parse_op!("ReduceLogSumExp", OpType::ReduceLogSumExp);
    assert_parse_op!("ReduceSumSquare", OpType::ReduceSumSquare);
    assert_parse_op!("ArgMax", OpType::ArgMax);
    assert_parse_op!("ArgMin", OpType::ArgMin);
}

#[test]
fn test_parse_comparison_logic_ops() {
    assert_parse_op!("Equal", OpType::Equal);
    assert_parse_op!("Greater", OpType::Greater);
    assert_parse_op!("GreaterOrEqual", OpType::GreaterOrEqual);
    assert_parse_op!("Less", OpType::Less);
    assert_parse_op!("LessOrEqual", OpType::LessOrEqual);
    assert_parse_op!("Not", OpType::Not);
    assert_parse_op!("And", OpType::And);
    assert_parse_op!("Or", OpType::Or);
    assert_parse_op!("Xor", OpType::Xor);
    assert_parse_op!("Where", OpType::Where);
    assert_parse_op!("IsNaN", OpType::IsNaN);
    assert_parse_op!("IsInf", OpType::IsInf);
}

#[test]
fn test_parse_shape_type_ops() {
    assert_parse_op!("Shape", OpType::Shape);
    assert_parse_op!("Size", OpType::Size);
    assert_parse_op!("Cast", OpType::Cast);
    assert_parse_op!("CastLike", OpType::CastLike);
    assert_parse_op!("ConstantOfShape", OpType::ConstantOfShape);
    assert_parse_op!("Range", OpType::Range);
    assert_parse_op!("OneHot", OpType::OneHot);
    assert_parse_op!("NonZero", OpType::NonZero);
    assert_parse_op!("TopK", OpType::TopK);
    assert_parse_op!("Unique", OpType::Unique);
    assert_parse_op!("EyeLike", OpType::EyeLike);
    assert_parse_op!("Compress", OpType::Compress);
    assert_parse_op!("Flatten", OpType::Flatten);
}

#[test]
fn test_parse_rnn_ops() {
    assert_parse_op!("LSTM", OpType::LSTM);
    assert_parse_op!("GRU", OpType::GRU);
    assert_parse_op!("RNN", OpType::RNN);
}

#[test]
fn test_parse_constant_identity_ops() {
    assert_parse_op!("Constant", OpType::Constant);
    assert_parse_op!("Identity", OpType::Identity);
}

#[test]
fn test_parse_control_flow_ops() {
    assert_parse_op!("If", OpType::If);
    assert_parse_op!("Loop", OpType::Loop);
    assert_parse_op!("Scan", OpType::Scan);
}

#[test]
fn test_parse_resize_ops() {
    assert_parse_op!("Resize", OpType::Resize);
    assert_parse_op!("Upsample", OpType::Upsample);
}

#[test]
fn test_parse_regularization_ops() {
    assert_parse_op!("Dropout", OpType::Dropout);
}

#[test]
fn test_parse_quantization_ops() {
    assert_parse_op!("QuantizeLinear", OpType::QuantizeLinear);
    assert_parse_op!("DequantizeLinear", OpType::DequantizeLinear);
    assert_parse_op!("DynamicQuantizeLinear", OpType::DynamicQuantizeLinear);
    assert_parse_op!("QLinearConv", OpType::QLinearConv);
    assert_parse_op!("QLinearMatMul", OpType::QLinearMatMul);
}

#[test]
fn test_parse_attention_transformer_ops() {
    assert_parse_op!("Attention", OpType::Attention);
    assert_parse_op!("Einsum", OpType::Einsum);
}

#[test]
fn test_parse_sequence_ops() {
    assert_parse_op!("SequenceConstruct", OpType::SequenceConstruct);
    assert_parse_op!("SequenceAt", OpType::SequenceAt);
    assert_parse_op!("SequenceEmpty", OpType::SequenceEmpty);
    assert_parse_op!("SequenceInsert", OpType::SequenceInsert);
    assert_parse_op!("SequenceErase", OpType::SequenceErase);
    assert_parse_op!("SequenceLength", OpType::SequenceLength);
    assert_parse_op!("SequenceMap", OpType::SequenceMap);
}

#[test]
fn test_parse_optional_ops() {
    assert_parse_op!("Optional", OpType::Optional);
    assert_parse_op!("OptionalGetElement", OpType::OptionalGetElement);
    assert_parse_op!("OptionalHasElement", OpType::OptionalHasElement);
}

#[test]
fn test_parse_image_ops() {
    assert_parse_op!("ImageDecoder", OpType::ImageDecoder);
    assert_parse_op!("GridSample", OpType::GridSample);
    assert_parse_op!("RoiAlign", OpType::RoiAlign);
    assert_parse_op!("AffineGrid", OpType::AffineGrid);
}

#[test]
fn test_parse_misc_ops() {
    assert_parse_op!("Cumsum", OpType::CumSum);
    assert_parse_op!("Det", OpType::Det);
    assert_parse_op!("Erf", OpType::Erf);
    assert_parse_op!("HardMax", OpType::Hardmax);
    assert_parse_op!("Shrink", OpType::Shrink);
    assert_parse_op!("StringNormalizer", OpType::StringNormalizer);
    assert_parse_op!("TfIdfVectorizer", OpType::TfIdfVectorizer);
    assert_parse_op!("NegativeLogLikelihoodLoss", OpType::NegativeLogLikelihoodLoss);
    assert_parse_op!("SoftmaxCrossEntropyLoss", OpType::SoftmaxCrossEntropyLoss);
    assert_parse_op!("Bernoulli", OpType::Bernoulli);
    assert_parse_op!("RandomNormal", OpType::RandomNormal);
    assert_parse_op!("RandomNormalLike", OpType::RandomNormalLike);
    assert_parse_op!("RandomUniform", OpType::RandomUniform);
    assert_parse_op!("RandomUniformLike", OpType::RandomUniformLike);
    assert_parse_op!("Multinomial", OpType::Multinomial);
    assert_parse_op!("CenterCropPad", OpType::CenterCropPad);
    assert_parse_op!("Col2Im", OpType::Col2Im);
    assert_parse_op!("Trilu", OpType::Trilu);
    assert_parse_op!("BlackmanWindow", OpType::BlackmanWindow);
    assert_parse_op!("HannWindow", OpType::HannWindow);
    assert_parse_op!("HammingWindow", OpType::HammingWindow);
    assert_parse_op!("DFT", OpType::DFT);
    assert_parse_op!("MelWeightMatrix", OpType::MelWeightMatrix);
    assert_parse_op!("STFT", OpType::STFT);
    assert_parse_op!("RegexFullMatch", OpType::RegexFullMatch);
    assert_parse_op!("StringConcat", OpType::StringConcat);
    assert_parse_op!("StringSplit", OpType::StringSplit);
}

// === Unknown ops fall back to Custom ===

#[test]
fn test_parse_unknown_op_becomes_custom() {
    assert_parse_op!("SomeVendorSpecificOp", OpType::Custom("SomeVendorSpecificOp"));
}

#[test]
fn test_parse_empty_op_becomes_custom() {
    assert_parse_op!("", OpType::Custom(""));
}

// === Existing parser tests still work with OpType ===

#[test]
fn test_mini_model_with_typed_op() {
    let data = make_model_with_op("Relu");
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    assert_eq!(g.node[0].op_type, OpType::Relu);

    // Pattern matching works
    match &g.node[0].op_type {
        OpType::Relu => {}
        other => panic!("expected Relu, got {:?}", other),
    }
}

#[test]
fn test_multiple_ops_in_graph() {
    let ops = ["Conv", "BatchNormalization", "Relu", "MaxPool", "Flatten", "Gemm"];
    let mut graph = Vec::new();

    for (i, op) in ops.iter().enumerate() {
        let mut node = Vec::new();
        node.extend(encode_string_field(1, &format!("in_{i}")));
        node.extend(encode_string_field(2, &format!("out_{i}")));
        node.extend(encode_string_field(4, op));
        graph.extend(encode_length_delimited(1, &node));
    }

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    assert_eq!(g.node.len(), 6);
    assert_eq!(g.node[0].op_type, OpType::Conv);
    assert_eq!(g.node[1].op_type, OpType::BatchNormalization);
    assert_eq!(g.node[2].op_type, OpType::Relu);
    assert_eq!(g.node[3].op_type, OpType::MaxPool);
    assert_eq!(g.node[4].op_type, OpType::Flatten);
    assert_eq!(g.node[5].op_type, OpType::Gemm);
}
