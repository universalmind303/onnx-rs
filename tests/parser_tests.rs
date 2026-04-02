use onnx_rs::ast::*;
use onnx_rs::parse;

// Helper: encode a varint into bytes
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

// Helper: encode a protobuf tag
fn encode_tag(field_number: u32, wire_type: u8) -> Vec<u8> {
    encode_varint(((field_number as u64) << 3) | wire_type as u64)
}

// Helper: encode a length-delimited field (string, bytes, or sub-message)
fn encode_length_delimited(field_number: u32, data: &[u8]) -> Vec<u8> {
    let mut buf = encode_tag(field_number, 2);
    buf.extend(encode_varint(data.len() as u64));
    buf.extend(data);
    buf
}

// Helper: encode a varint field
fn encode_varint_field(field_number: u32, value: u64) -> Vec<u8> {
    let mut buf = encode_tag(field_number, 0);
    buf.extend(encode_varint(value));
    buf
}

// Helper: encode a string field
fn encode_string_field(field_number: u32, s: &str) -> Vec<u8> {
    encode_length_delimited(field_number, s.as_bytes())
}

// Helper: encode a fixed32 field
fn encode_fixed32_field(field_number: u32, value: u32) -> Vec<u8> {
    let mut buf = encode_tag(field_number, 5);
    buf.extend(value.to_le_bytes());
    buf
}

// === Empty model ===

#[test]
fn test_parse_empty_bytes() {
    let model = parse(&[]).unwrap();
    assert_eq!(model.ir_version, 0);
    assert!(model.graph.is_none());
    assert!(model.opset_import.is_empty());
}

// === ModelProto fields ===

#[test]
fn test_parse_model_ir_version() {
    let data = encode_varint_field(1, 7);
    let model = parse(&data).unwrap();
    assert_eq!(model.ir_version, 7);
}

#[test]
fn test_parse_model_producer_name() {
    let data = encode_string_field(2, "pytorch");
    let model = parse(&data).unwrap();
    assert_eq!(model.producer_name, "pytorch");
}

#[test]
fn test_parse_model_producer_version() {
    let data = encode_string_field(3, "2.1.0");
    let model = parse(&data).unwrap();
    assert_eq!(model.producer_version, "2.1.0");
}

#[test]
fn test_parse_model_domain() {
    let data = encode_string_field(4, "ai.onnx");
    let model = parse(&data).unwrap();
    assert_eq!(model.domain, "ai.onnx");
}

#[test]
fn test_parse_model_version() {
    let data = encode_varint_field(5, 1);
    let model = parse(&data).unwrap();
    assert_eq!(model.model_version, 1);
}

#[test]
fn test_parse_model_doc_string() {
    let data = encode_string_field(6, "A test model");
    let model = parse(&data).unwrap();
    assert_eq!(model.doc_string, "A test model");
}

#[test]
fn test_parse_model_multiple_fields() {
    let mut data = Vec::new();
    data.extend(encode_varint_field(1, 8)); // ir_version = 8
    data.extend(encode_string_field(2, "onnx-rs")); // producer_name
    data.extend(encode_string_field(3, "0.1.0")); // producer_version
    data.extend(encode_varint_field(5, 1)); // model_version
    let model = parse(&data).unwrap();
    assert_eq!(model.ir_version, 8);
    assert_eq!(model.producer_name, "onnx-rs");
    assert_eq!(model.producer_version, "0.1.0");
    assert_eq!(model.model_version, 1);
}

// === OperatorSetIdProto ===

#[test]
fn test_parse_model_opset_import() {
    // Build an OperatorSetIdProto: domain="" (default ONNX), version=13
    let mut opset = Vec::new();
    opset.extend(encode_varint_field(2, 13)); // version = 13

    let mut data = Vec::new();
    data.extend(encode_length_delimited(8, &opset)); // opset_import

    let model = parse(&data).unwrap();
    assert_eq!(model.opset_import.len(), 1);
    assert_eq!(model.opset_import[0].domain, "");
    assert_eq!(model.opset_import[0].version, 13);
}

#[test]
fn test_parse_model_multiple_opset_imports() {
    let mut opset1 = Vec::new();
    opset1.extend(encode_varint_field(2, 13));

    let mut opset2 = Vec::new();
    opset2.extend(encode_string_field(1, "ai.onnx.ml"));
    opset2.extend(encode_varint_field(2, 3));

    let mut data = Vec::new();
    data.extend(encode_length_delimited(8, &opset1));
    data.extend(encode_length_delimited(8, &opset2));

    let model = parse(&data).unwrap();
    assert_eq!(model.opset_import.len(), 2);
    assert_eq!(model.opset_import[0].version, 13);
    assert_eq!(model.opset_import[1].domain, "ai.onnx.ml");
    assert_eq!(model.opset_import[1].version, 3);
}

// === StringStringEntryProto (metadata_props) ===

#[test]
fn test_parse_model_metadata_props() {
    let mut entry = Vec::new();
    entry.extend(encode_string_field(1, "author"));
    entry.extend(encode_string_field(2, "test"));

    let mut data = Vec::new();
    data.extend(encode_length_delimited(14, &entry));

    let model = parse(&data).unwrap();
    assert_eq!(model.metadata_props.len(), 1);
    assert_eq!(model.metadata_props[0].key, "author");
    assert_eq!(model.metadata_props[0].value, "test");
}

// === GraphProto ===

#[test]
fn test_parse_empty_graph() {
    let graph_bytes = encode_string_field(2, "main_graph"); // name = "main_graph"

    let mut data = Vec::new();
    data.extend(encode_length_delimited(7, &graph_bytes));

    let model = parse(&data).unwrap();
    let graph = model.graph.unwrap();
    assert_eq!(graph.name, "main_graph");
    assert!(graph.node.is_empty());
}

// === NodeProto ===

#[test]
fn test_parse_graph_with_node() {
    // Build a NodeProto: op_type="Relu", input=["x"], output=["y"]
    let mut node = Vec::new();
    node.extend(encode_string_field(1, "x"));      // input
    node.extend(encode_string_field(2, "y"));      // output
    node.extend(encode_string_field(3, "relu0"));  // name
    node.extend(encode_string_field(4, "Relu"));   // op_type

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(1, &node)); // node
    graph.extend(encode_string_field(2, "test"));     // name

    let mut data = Vec::new();
    data.extend(encode_length_delimited(7, &graph));

    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    assert_eq!(g.node.len(), 1);
    assert_eq!(g.node[0].op_type, OpType::Relu);
    assert_eq!(g.node[0].name, "relu0");
    assert_eq!(g.node[0].input, vec!["x"]);
    assert_eq!(g.node[0].output, vec!["y"]);
}

#[test]
fn test_parse_node_multiple_inputs_outputs() {
    let mut node = Vec::new();
    node.extend(encode_string_field(1, "a"));
    node.extend(encode_string_field(1, "b"));
    node.extend(encode_string_field(1, "c"));
    node.extend(encode_string_field(2, "out"));
    node.extend(encode_string_field(4, "Add"));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(1, &node));

    let mut data = Vec::new();
    data.extend(encode_length_delimited(7, &graph));

    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    assert_eq!(g.node[0].input, vec!["a", "b", "c"]);
    assert_eq!(g.node[0].output, vec!["out"]);
    assert_eq!(g.node[0].op_type, OpType::Add);
}

#[test]
fn test_parse_node_with_domain_and_overload() {
    let mut node = Vec::new();
    node.extend(encode_string_field(4, "CustomOp"));
    node.extend(encode_string_field(7, "com.custom"));
    node.extend(encode_string_field(8, "v2"));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(1, &node));

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    assert_eq!(g.node[0].domain, "com.custom");
    assert_eq!(g.node[0].overload, "v2");
}

// === AttributeProto ===

#[test]
fn test_parse_attribute_int() {
    let mut attr = Vec::new();
    attr.extend(encode_string_field(1, "axis")); // name
    attr.extend(encode_varint_field(20, 2));     // type = INT
    attr.extend(encode_varint_field(3, 1));      // i = 1

    let mut node = Vec::new();
    node.extend(encode_string_field(4, "Concat"));
    node.extend(encode_length_delimited(5, &attr));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(1, &node));

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    let a = &g.node[0].attribute[0];
    assert_eq!(a.name, "axis");
    assert_eq!(a.r#type, AttributeType::Int);
    assert_eq!(a.i, 1);
}

#[test]
fn test_parse_attribute_float() {
    let mut attr = Vec::new();
    attr.extend(encode_string_field(1, "alpha"));
    attr.extend(encode_varint_field(20, 1)); // type = FLOAT
    // f is field 2, wire type 5 (fixed32)
    attr.extend(encode_fixed32_field(2, f32::to_bits(0.5)));

    let mut node = Vec::new();
    node.extend(encode_string_field(4, "LeakyRelu"));
    node.extend(encode_length_delimited(5, &attr));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(1, &node));

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    let a = &g.node[0].attribute[0];
    assert_eq!(a.name, "alpha");
    assert_eq!(a.r#type, AttributeType::Float);
    assert_eq!(a.f, 0.5);
}

#[test]
fn test_parse_attribute_string() {
    let mut attr = Vec::new();
    attr.extend(encode_string_field(1, "mode"));
    attr.extend(encode_varint_field(20, 3)); // type = STRING
    attr.extend(encode_length_delimited(4, b"constant")); // s = "constant"

    let mut node = Vec::new();
    node.extend(encode_string_field(4, "Pad"));
    node.extend(encode_length_delimited(5, &attr));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(1, &node));

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    let a = &g.node[0].attribute[0];
    assert_eq!(a.name, "mode");
    assert_eq!(a.s, b"constant");
}

#[test]
fn test_parse_attribute_ints() {
    let mut attr = Vec::new();
    attr.extend(encode_string_field(1, "kernel_shape"));
    attr.extend(encode_varint_field(20, 7)); // type = INTS
    // ints field 8, packed encoding: varint 3, varint 3
    let packed: Vec<u8> = [encode_varint(3), encode_varint(3)].concat();
    attr.extend(encode_length_delimited(8, &packed));

    let mut node = Vec::new();
    node.extend(encode_string_field(4, "Conv"));
    node.extend(encode_length_delimited(5, &attr));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(1, &node));

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    let a = &g.node[0].attribute[0];
    assert_eq!(a.name, "kernel_shape");
    assert_eq!(a.ints, vec![3, 3]);
}

// === ValueInfoProto ===

#[test]
fn test_parse_graph_input_value_info() {
    // ValueInfoProto: name="input", type=Tensor(float, shape=[1, 3, 224, 224])
    let shape_dim = |v: i64| -> Vec<u8> {
        let mut dim = Vec::new();
        dim.extend(encode_varint_field(1, v as u64)); // dim_value
        dim
    };

    let mut shape = Vec::new();
    for &d in &[1i64, 3, 224, 224] {
        shape.extend(encode_length_delimited(1, &shape_dim(d)));
    }

    let mut tensor_type = Vec::new();
    tensor_type.extend(encode_varint_field(1, 1)); // elem_type = FLOAT
    tensor_type.extend(encode_length_delimited(2, &shape));

    let mut type_proto = Vec::new();
    type_proto.extend(encode_length_delimited(1, &tensor_type)); // tensor_type

    let mut vi = Vec::new();
    vi.extend(encode_string_field(1, "input"));
    vi.extend(encode_length_delimited(2, &type_proto));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(11, &vi)); // input

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    assert_eq!(g.input.len(), 1);
    assert_eq!(g.input[0].name, "input");

    let tp = g.input[0].r#type.as_ref().unwrap();
    match tp.value.as_ref().unwrap() {
        TypeValue::Tensor(t) => {
            assert_eq!(t.elem_type, DataType::Float);
            let dims: Vec<_> = t.shape.as_ref().unwrap().dim.iter().map(|d| {
                match &d.value {
                    Dimension::Value(v) => *v,
                    _ => panic!("expected dim value"),
                }
            }).collect();
            assert_eq!(dims, vec![1, 3, 224, 224]);
        }
        _ => panic!("expected tensor type"),
    }
}

#[test]
fn test_parse_symbolic_dimension() {
    let mut dim = Vec::new();
    dim.extend(encode_string_field(2, "batch_size")); // dim_param

    let mut shape = Vec::new();
    shape.extend(encode_length_delimited(1, &dim));

    let mut tensor_type = Vec::new();
    tensor_type.extend(encode_varint_field(1, 1)); // FLOAT
    tensor_type.extend(encode_length_delimited(2, &shape));

    let mut type_proto = Vec::new();
    type_proto.extend(encode_length_delimited(1, &tensor_type));

    let mut vi = Vec::new();
    vi.extend(encode_string_field(1, "x"));
    vi.extend(encode_length_delimited(2, &type_proto));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(11, &vi));

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    let tp = g.input[0].r#type.as_ref().unwrap();
    match tp.value.as_ref().unwrap() {
        TypeValue::Tensor(t) => {
            match &t.shape.as_ref().unwrap().dim[0].value {
                Dimension::Param(p) => assert_eq!(*p, "batch_size"),
                _ => panic!("expected dim param"),
            }
        }
        _ => panic!("expected tensor type"),
    }
}

// === TensorProto (initializers) ===

#[test]
fn test_parse_tensor_proto_float_data() {
    let mut tensor = Vec::new();
    // dims = [2, 2]
    let packed_dims: Vec<u8> = [encode_varint(2), encode_varint(2)].concat();
    tensor.extend(encode_length_delimited(1, &packed_dims));
    // data_type = FLOAT (1)
    tensor.extend(encode_varint_field(2, 1));
    // float_data = [1.0, 2.0, 3.0, 4.0] (packed fixed32)
    let mut packed_floats = Vec::new();
    for &f in &[1.0f32, 2.0, 3.0, 4.0] {
        packed_floats.extend(f.to_le_bytes());
    }
    tensor.extend(encode_length_delimited(4, &packed_floats));
    // name
    tensor.extend(encode_string_field(8, "weight"));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(5, &tensor)); // initializer

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    assert_eq!(g.initializer.len(), 1);
    let t = &g.initializer[0];
    assert_eq!(t.name, "weight");
    assert_eq!(t.dims, vec![2, 2]);
    assert_eq!(t.data_type, DataType::Float);
    assert_eq!(t.float_data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_parse_tensor_proto_raw_data() {
    let mut tensor = Vec::new();
    tensor.extend(encode_varint_field(2, 1)); // data_type = FLOAT
    let raw: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    tensor.extend(encode_length_delimited(9, &raw)); // raw_data

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(5, &tensor));

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    assert_eq!(g.initializer[0].raw_data, raw);
}

#[test]
fn test_parse_tensor_proto_int64_data_packed() {
    let mut tensor = Vec::new();
    tensor.extend(encode_varint_field(2, 7)); // data_type = INT64
    let packed: Vec<u8> = [encode_varint(10), encode_varint(20), encode_varint(30)].concat();
    tensor.extend(encode_length_delimited(7, &packed)); // int64_data

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(5, &tensor));

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    assert_eq!(g.initializer[0].int64_data, vec![10, 20, 30]);
}

// === Unknown fields are skipped ===

#[test]
fn test_unknown_fields_skipped() {
    let mut data = Vec::new();
    data.extend(encode_varint_field(1, 7)); // ir_version (known)
    data.extend(encode_string_field(99, "unknown")); // unknown field
    data.extend(encode_string_field(2, "pytorch")); // producer_name (known)

    let model = parse(&data).unwrap();
    assert_eq!(model.ir_version, 7);
    assert_eq!(model.producer_name, "pytorch");
}

// === Full mini model ===

#[test]
fn test_parse_mini_model() {
    // A minimal but complete model: ir_version=7, opset=13, graph with one Relu node
    let mut opset = Vec::new();
    opset.extend(encode_varint_field(2, 13));

    let mut node = Vec::new();
    node.extend(encode_string_field(1, "x"));
    node.extend(encode_string_field(2, "y"));
    node.extend(encode_string_field(4, "Relu"));

    // input value_info
    let mut in_tensor_type = Vec::new();
    in_tensor_type.extend(encode_varint_field(1, 1)); // FLOAT

    let mut in_type = Vec::new();
    in_type.extend(encode_length_delimited(1, &in_tensor_type));

    let mut input_vi = Vec::new();
    input_vi.extend(encode_string_field(1, "x"));
    input_vi.extend(encode_length_delimited(2, &in_type));

    // output value_info
    let mut output_vi = Vec::new();
    output_vi.extend(encode_string_field(1, "y"));
    output_vi.extend(encode_length_delimited(2, &in_type));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(1, &node));
    graph.extend(encode_string_field(2, "main"));
    graph.extend(encode_length_delimited(11, &input_vi));
    graph.extend(encode_length_delimited(12, &output_vi));

    let mut data = Vec::new();
    data.extend(encode_varint_field(1, 7));
    data.extend(encode_string_field(2, "test"));
    data.extend(encode_length_delimited(7, &graph));
    data.extend(encode_length_delimited(8, &opset));

    let model = parse(&data).unwrap();
    assert_eq!(model.ir_version, 7);
    assert_eq!(model.producer_name, "test");
    assert_eq!(model.opset_import.len(), 1);
    assert_eq!(model.opset_import[0].version, 13);

    let g = model.graph.unwrap();
    assert_eq!(g.name, "main");
    assert_eq!(g.node.len(), 1);
    assert_eq!(g.node[0].op_type, OpType::Relu);
    assert_eq!(g.input.len(), 1);
    assert_eq!(g.input[0].name, "x");
    assert_eq!(g.output.len(), 1);
    assert_eq!(g.output[0].name, "y");
}

// === FunctionProto ===

#[test]
fn test_parse_function_proto() {
    let mut func = Vec::new();
    func.extend(encode_string_field(1, "MyFunc"));
    func.extend(encode_string_field(4, "x"));
    func.extend(encode_string_field(5, "y"));
    func.extend(encode_string_field(10, "com.custom"));

    let mut node = Vec::new();
    node.extend(encode_string_field(1, "x"));
    node.extend(encode_string_field(2, "y"));
    node.extend(encode_string_field(4, "Relu"));
    func.extend(encode_length_delimited(7, &node));

    let mut data = Vec::new();
    data.extend(encode_length_delimited(25, &func));

    let model = parse(&data).unwrap();
    assert_eq!(model.functions.len(), 1);
    assert_eq!(model.functions[0].name, "MyFunc");
    assert_eq!(model.functions[0].domain, "com.custom");
    assert_eq!(model.functions[0].input, vec!["x"]);
    assert_eq!(model.functions[0].output, vec!["y"]);
    assert_eq!(model.functions[0].node.len(), 1);
}

// === TrainingInfoProto ===

#[test]
fn test_parse_training_info() {
    let mut train_graph = Vec::new();
    train_graph.extend(encode_string_field(2, "train_step"));

    let mut training_info = Vec::new();
    training_info.extend(encode_length_delimited(2, &train_graph)); // algorithm

    let mut data = Vec::new();
    data.extend(encode_length_delimited(20, &training_info));

    let model = parse(&data).unwrap();
    assert_eq!(model.training_info.len(), 1);
    let ti = &model.training_info[0];
    assert!(ti.initialization.is_none());
    assert_eq!(ti.algorithm.as_ref().unwrap().name, "train_step");
}

// === SparseTensorProto ===

#[test]
fn test_parse_sparse_tensor() {
    let mut values_tensor = Vec::new();
    values_tensor.extend(encode_varint_field(2, 1)); // FLOAT

    let mut indices_tensor = Vec::new();
    indices_tensor.extend(encode_varint_field(2, 7)); // INT64

    let mut sparse = Vec::new();
    sparse.extend(encode_length_delimited(1, &values_tensor));
    sparse.extend(encode_length_delimited(2, &indices_tensor));
    let packed_dims: Vec<u8> = [encode_varint(3), encode_varint(4)].concat();
    sparse.extend(encode_length_delimited(3, &packed_dims));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(15, &sparse)); // sparse_initializer

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    assert_eq!(g.sparse_initializer.len(), 1);
    assert_eq!(g.sparse_initializer[0].dims, vec![3, 4]);
    assert_eq!(
        g.sparse_initializer[0].values.as_ref().unwrap().data_type,
        DataType::Float
    );
}

// === TypeProto variants ===

#[test]
fn test_parse_sequence_type() {
    let mut inner_tensor_type = Vec::new();
    inner_tensor_type.extend(encode_varint_field(1, 7)); // INT64

    let mut inner_type = Vec::new();
    inner_type.extend(encode_length_delimited(1, &inner_tensor_type));

    let mut seq_type = Vec::new();
    seq_type.extend(encode_length_delimited(1, &inner_type)); // elem_type

    let mut type_proto = Vec::new();
    type_proto.extend(encode_length_delimited(4, &seq_type)); // sequence_type

    let mut vi = Vec::new();
    vi.extend(encode_string_field(1, "seq_input"));
    vi.extend(encode_length_delimited(2, &type_proto));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(11, &vi));

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    let tp = g.input[0].r#type.as_ref().unwrap();
    match tp.value.as_ref().unwrap() {
        TypeValue::Sequence(s) => {
            match s.elem_type.value.as_ref().unwrap() {
                TypeValue::Tensor(t) => assert_eq!(t.elem_type, DataType::Int64),
                _ => panic!("expected tensor inside sequence"),
            }
        }
        _ => panic!("expected sequence type"),
    }
}

#[test]
fn test_parse_map_type() {
    let mut inner_tensor_type = Vec::new();
    inner_tensor_type.extend(encode_varint_field(1, 1)); // FLOAT

    let mut inner_type = Vec::new();
    inner_type.extend(encode_length_delimited(1, &inner_tensor_type));

    let mut map_type = Vec::new();
    map_type.extend(encode_varint_field(1, 8)); // key_type = STRING
    map_type.extend(encode_length_delimited(2, &inner_type)); // value_type

    let mut type_proto = Vec::new();
    type_proto.extend(encode_length_delimited(5, &map_type)); // map_type

    let mut vi = Vec::new();
    vi.extend(encode_string_field(1, "map_input"));
    vi.extend(encode_length_delimited(2, &type_proto));

    let mut graph = Vec::new();
    graph.extend(encode_length_delimited(11, &vi));

    let data = encode_length_delimited(7, &graph);
    let model = parse(&data).unwrap();
    let g = model.graph.unwrap();
    let tp = g.input[0].r#type.as_ref().unwrap();
    match tp.value.as_ref().unwrap() {
        TypeValue::Map(m) => {
            assert_eq!(m.key_type, DataType::String);
            match m.value_type.value.as_ref().unwrap() {
                TypeValue::Tensor(t) => assert_eq!(t.elem_type, DataType::Float),
                _ => panic!("expected tensor value type"),
            }
        }
        _ => panic!("expected map type"),
    }
}
