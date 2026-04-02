use crate::ast::*;

/// Encodes a [`Model`] into ONNX protobuf bytes.
///
/// The returned bytes are a valid serialized `ModelProto` protobuf message
/// that can be written directly to a `.onnx` file.
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::*;
///
/// let model = Model {
///     ir_version: 9,
///     producer_name: "my-tool",
///     producer_version: "1.0",
///     ..Default::default()
/// };
///
/// let bytes = onnx_rs::encode(&model);
/// assert!(!bytes.is_empty());
///
/// // Roundtrips cleanly through parse
/// let roundtrip = onnx_rs::parse(&bytes).unwrap();
/// assert_eq!(roundtrip.ir_version, 9);
/// assert_eq!(roundtrip.producer_name, "my-tool");
/// ```
///
/// Writing to a file:
///
/// ```no_run
/// # use onnx_rs::ast::*;
/// # let model = Model::default();
/// let bytes = onnx_rs::encode(&model);
/// std::fs::write("output.onnx", &bytes).unwrap();
/// ```
pub fn encode(model: &Model<'_>) -> Vec<u8> {
    encode_model(model)
}

fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
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
}

fn write_tag(buf: &mut Vec<u8>, field_number: u32, wire_type: u8) {
    write_varint(buf, ((field_number as u64) << 3) | wire_type as u64);
}

fn write_varint_field(buf: &mut Vec<u8>, field_number: u32, value: u64) {
    if value == 0 {
        return;
    }
    write_tag(buf, field_number, 0);
    write_varint(buf, value);
}

fn write_sint_field(buf: &mut Vec<u8>, field_number: u32, value: i64) {
    write_varint_field(buf, field_number, value as u64);
}

fn write_string_field(buf: &mut Vec<u8>, field_number: u32, value: &str) {
    if value.is_empty() {
        return;
    }
    write_tag(buf, field_number, 2);
    write_varint(buf, value.len() as u64);
    buf.extend_from_slice(value.as_bytes());
}

fn write_string_field_always(buf: &mut Vec<u8>, field_number: u32, value: &str) {
    write_tag(buf, field_number, 2);
    write_varint(buf, value.len() as u64);
    buf.extend_from_slice(value.as_bytes());
}

fn write_bytes_field(buf: &mut Vec<u8>, field_number: u32, value: &[u8]) {
    if value.is_empty() {
        return;
    }
    write_tag(buf, field_number, 2);
    write_varint(buf, value.len() as u64);
    buf.extend_from_slice(value);
}

fn write_message_field(buf: &mut Vec<u8>, field_number: u32, content: &[u8]) {
    if content.is_empty() {
        return;
    }
    write_tag(buf, field_number, 2);
    write_varint(buf, content.len() as u64);
    buf.extend_from_slice(content);
}

fn write_message_field_always(buf: &mut Vec<u8>, field_number: u32, content: &[u8]) {
    write_tag(buf, field_number, 2);
    write_varint(buf, content.len() as u64);
    buf.extend_from_slice(content);
}

fn write_fixed32_field(buf: &mut Vec<u8>, field_number: u32, value: u32) {
    if value == 0 {
        return;
    }
    write_tag(buf, field_number, 5);
    buf.extend_from_slice(&value.to_le_bytes());
}

fn encode_string_string_entry(entry: &StringStringEntry<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_string_field(&mut buf, 1, &entry.key);
    write_string_field(&mut buf, 2, &entry.value);
    buf
}

fn encode_opset_id(opset: &OperatorSetId<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_string_field(&mut buf, 1, &opset.domain);
    write_varint_field(&mut buf, 2, opset.version as u64);
    buf
}

fn encode_tensor_shape_dim(dim: &TensorShapeDimension<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    match &dim.value {
        Dimension::Value(v) => write_sint_field(&mut buf, 1, *v),
        Dimension::Param(p) => write_string_field(&mut buf, 2, p),
    }
    write_string_field(&mut buf, 3, &dim.denotation);
    buf
}

fn encode_tensor_shape(shape: &TensorShape<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    for dim in &shape.dim {
        let inner = encode_tensor_shape_dim(dim);
        write_message_field_always(&mut buf, 1, &inner);
    }
    buf
}

fn encode_tensor_type(tt: &TensorTypeProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_varint_field(&mut buf, 1, tt.elem_type as u64);
    if let Some(shape) = &tt.shape {
        let inner = encode_tensor_shape(shape);
        write_message_field_always(&mut buf, 2, &inner);
    }
    buf
}

fn encode_type_proto(tp: &TypeProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    if let Some(value) = &tp.value {
        match value {
            TypeValue::Tensor(t) => {
                let inner = encode_tensor_type(t);
                write_message_field(&mut buf, 1, &inner);
            }
            TypeValue::Sequence(s) => {
                let inner = encode_sequence_type(s);
                write_message_field(&mut buf, 4, &inner);
            }
            TypeValue::Map(m) => {
                let inner = encode_map_type(m);
                write_message_field(&mut buf, 5, &inner);
            }
            TypeValue::SparseTensor(st) => {
                let inner = encode_sparse_tensor_type(st);
                write_message_field(&mut buf, 8, &inner);
            }
            TypeValue::Optional(o) => {
                let inner = encode_optional_type(o);
                write_message_field(&mut buf, 9, &inner);
            }
        }
    }
    write_string_field(&mut buf, 6, &tp.denotation);
    buf
}

fn encode_sequence_type(s: &SequenceTypeProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    let inner = encode_type_proto(&s.elem_type);
    write_message_field(&mut buf, 1, &inner);
    buf
}

fn encode_map_type(m: &MapTypeProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_varint_field(&mut buf, 1, m.key_type as u64);
    let inner = encode_type_proto(&m.value_type);
    write_message_field(&mut buf, 2, &inner);
    buf
}

fn encode_optional_type(o: &OptionalTypeProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    let inner = encode_type_proto(&o.elem_type);
    write_message_field(&mut buf, 1, &inner);
    buf
}

fn encode_sparse_tensor_type(st: &SparseTensorTypeProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_varint_field(&mut buf, 1, st.elem_type as u64);
    if let Some(shape) = &st.shape {
        let inner = encode_tensor_shape(shape);
        write_message_field_always(&mut buf, 2, &inner);
    }
    buf
}

fn encode_value_info(vi: &ValueInfo<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_string_field(&mut buf, 1, &vi.name);
    if let Some(tp) = &vi.r#type {
        let inner = encode_type_proto(tp);
        write_message_field(&mut buf, 2, &inner);
    }
    write_string_field(&mut buf, 3, &vi.doc_string);
    for entry in &vi.metadata_props {
        let inner = encode_string_string_entry(entry);
        write_message_field(&mut buf, 4, &inner);
    }
    buf
}

fn encode_tensor_segment(seg: &TensorSegment) -> Vec<u8> {
    let mut buf = Vec::new();
    write_sint_field(&mut buf, 1, seg.begin);
    write_sint_field(&mut buf, 2, seg.end);
    buf
}

fn encode_tensor(t: &TensorProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();

    if !t.dims.is_empty() {
        let mut packed = Vec::new();
        for &d in &t.dims {
            write_varint(&mut packed, d as u64);
        }
        write_tag(&mut buf, 1, 2);
        write_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }

    write_varint_field(&mut buf, 2, t.data_type as u64);

    if let Some(seg) = &t.segment {
        let inner = encode_tensor_segment(seg);
        write_message_field(&mut buf, 3, &inner);
    }

    if !t.float_data.is_empty() {
        let mut packed = Vec::new();
        for &f in &t.float_data {
            packed.extend_from_slice(&f.to_le_bytes());
        }
        write_tag(&mut buf, 4, 2);
        write_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }

    if !t.int32_data.is_empty() {
        let mut packed = Vec::new();
        for &v in &t.int32_data {
            write_varint(&mut packed, v as u64);
        }
        write_tag(&mut buf, 5, 2);
        write_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }

    for s in &t.string_data {
        write_bytes_field(&mut buf, 6, s);
    }

    if !t.int64_data.is_empty() {
        let mut packed = Vec::new();
        for &v in &t.int64_data {
            write_varint(&mut packed, v as u64);
        }
        write_tag(&mut buf, 7, 2);
        write_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }

    write_string_field(&mut buf, 8, &t.name);
    write_bytes_field(&mut buf, 9, &t.raw_data);

    if !t.double_data.is_empty() {
        let mut packed = Vec::new();
        for &d in &t.double_data {
            packed.extend_from_slice(&d.to_le_bytes());
        }
        write_tag(&mut buf, 10, 2);
        write_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }

    if !t.uint64_data.is_empty() {
        let mut packed = Vec::new();
        for &v in &t.uint64_data {
            write_varint(&mut packed, v);
        }
        write_tag(&mut buf, 11, 2);
        write_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }

    write_string_field(&mut buf, 12, &t.doc_string);

    for entry in &t.external_data {
        let inner = encode_string_string_entry(entry);
        write_message_field(&mut buf, 13, &inner);
    }

    if t.data_location != DataLocation::Default {
        write_varint_field(&mut buf, 14, t.data_location as u64);
    }

    for entry in &t.metadata_props {
        let inner = encode_string_string_entry(entry);
        write_message_field(&mut buf, 16, &inner);
    }

    buf
}

fn encode_sparse_tensor(st: &SparseTensor<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    if let Some(values) = &st.values {
        let inner = encode_tensor(values);
        write_message_field(&mut buf, 1, &inner);
    }
    if let Some(indices) = &st.indices {
        let inner = encode_tensor(indices);
        write_message_field(&mut buf, 2, &inner);
    }
    if !st.dims.is_empty() {
        let mut packed = Vec::new();
        for &d in &st.dims {
            write_varint(&mut packed, d as u64);
        }
        write_tag(&mut buf, 3, 2);
        write_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }
    buf
}

fn encode_tensor_annotation(ta: &TensorAnnotation<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_string_field(&mut buf, 1, &ta.tensor_name);
    for entry in &ta.quant_parameter_tensor_names {
        let inner = encode_string_string_entry(entry);
        write_message_field(&mut buf, 2, &inner);
    }
    buf
}

fn encode_attribute(attr: &Attribute<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_string_field(&mut buf, 1, &attr.name);
    write_fixed32_field(&mut buf, 2, f32::to_bits(attr.f));
    write_sint_field(&mut buf, 3, attr.i);
    write_bytes_field(&mut buf, 4, &attr.s);

    if let Some(t) = &attr.t {
        let inner = encode_tensor(t);
        write_message_field(&mut buf, 5, &inner);
    }
    if let Some(g) = &attr.g {
        let inner = encode_graph(g);
        write_message_field(&mut buf, 6, &inner);
    }

    if !attr.floats.is_empty() {
        let mut packed = Vec::new();
        for &f in &attr.floats {
            packed.extend_from_slice(&f.to_le_bytes());
        }
        write_tag(&mut buf, 7, 2);
        write_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }

    if !attr.ints.is_empty() {
        let mut packed = Vec::new();
        for &v in &attr.ints {
            write_varint(&mut packed, v as u64);
        }
        write_tag(&mut buf, 8, 2);
        write_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }

    for s in &attr.strings {
        write_bytes_field(&mut buf, 9, s);
    }

    for t in &attr.tensors {
        let inner = encode_tensor(t);
        write_message_field(&mut buf, 10, &inner);
    }

    for g in &attr.graphs {
        let inner = encode_graph(g);
        write_message_field(&mut buf, 11, &inner);
    }

    write_string_field(&mut buf, 13, &attr.doc_string);

    if let Some(tp) = &attr.tp {
        let inner = encode_type_proto(tp);
        write_message_field(&mut buf, 14, &inner);
    }

    for tp in &attr.type_protos {
        let inner = encode_type_proto(tp);
        write_message_field(&mut buf, 15, &inner);
    }

    if attr.r#type != AttributeType::Undefined {
        write_varint_field(&mut buf, 20, attr.r#type as u64);
    }

    write_string_field(&mut buf, 21, &attr.ref_attr_name);

    if let Some(st) = &attr.sparse_tensor {
        let inner = encode_sparse_tensor(st);
        write_message_field(&mut buf, 22, &inner);
    }

    for st in &attr.sparse_tensors {
        let inner = encode_sparse_tensor(st);
        write_message_field(&mut buf, 23, &inner);
    }

    buf
}

fn encode_node(node: &Node<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    for input in &node.input {
        write_string_field_always(&mut buf, 1, input);
    }
    for output in &node.output {
        write_string_field_always(&mut buf, 2, output);
    }
    write_string_field(&mut buf, 3, &node.name);
    write_string_field(&mut buf, 4, node.op_type.as_str());
    for attr in &node.attribute {
        let inner = encode_attribute(attr);
        write_message_field(&mut buf, 5, &inner);
    }
    write_string_field(&mut buf, 6, &node.doc_string);
    write_string_field(&mut buf, 7, &node.domain);
    write_string_field(&mut buf, 8, &node.overload);
    for entry in &node.metadata_props {
        let inner = encode_string_string_entry(entry);
        write_message_field(&mut buf, 9, &inner);
    }
    buf
}

fn encode_graph(graph: &Graph<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    for node in &graph.node {
        let inner = encode_node(node);
        write_message_field(&mut buf, 1, &inner);
    }
    write_string_field(&mut buf, 2, &graph.name);
    for init in &graph.initializer {
        let inner = encode_tensor(init);
        write_message_field(&mut buf, 5, &inner);
    }
    write_string_field(&mut buf, 10, &graph.doc_string);
    for vi in &graph.input {
        let inner = encode_value_info(vi);
        write_message_field(&mut buf, 11, &inner);
    }
    for vi in &graph.output {
        let inner = encode_value_info(vi);
        write_message_field(&mut buf, 12, &inner);
    }
    for vi in &graph.value_info {
        let inner = encode_value_info(vi);
        write_message_field(&mut buf, 13, &inner);
    }
    for ta in &graph.quantization_annotation {
        let inner = encode_tensor_annotation(ta);
        write_message_field(&mut buf, 14, &inner);
    }
    for st in &graph.sparse_initializer {
        let inner = encode_sparse_tensor(st);
        write_message_field(&mut buf, 15, &inner);
    }
    for entry in &graph.metadata_props {
        let inner = encode_string_string_entry(entry);
        write_message_field(&mut buf, 16, &inner);
    }
    buf
}

fn encode_training_info(ti: &TrainingInfo<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    if let Some(init) = &ti.initialization {
        let inner = encode_graph(init);
        write_message_field(&mut buf, 1, &inner);
    }
    if let Some(algo) = &ti.algorithm {
        let inner = encode_graph(algo);
        write_message_field(&mut buf, 2, &inner);
    }
    for entry in &ti.initialization_binding {
        let inner = encode_string_string_entry(entry);
        write_message_field(&mut buf, 3, &inner);
    }
    for entry in &ti.update_binding {
        let inner = encode_string_string_entry(entry);
        write_message_field(&mut buf, 4, &inner);
    }
    buf
}

fn encode_function(func: &Function<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_string_field(&mut buf, 1, &func.name);
    for input in &func.input {
        write_string_field_always(&mut buf, 4, input);
    }
    for output in &func.output {
        write_string_field_always(&mut buf, 5, output);
    }
    for attr in &func.attribute {
        write_string_field_always(&mut buf, 6, attr);
    }
    for node in &func.node {
        let inner = encode_node(node);
        write_message_field(&mut buf, 7, &inner);
    }
    write_string_field(&mut buf, 8, &func.doc_string);
    for opset in &func.opset_import {
        let inner = encode_opset_id(opset);
        write_message_field(&mut buf, 9, &inner);
    }
    write_string_field(&mut buf, 10, &func.domain);
    for attr in &func.attribute_proto {
        let inner = encode_attribute(attr);
        write_message_field(&mut buf, 11, &inner);
    }
    for vi in &func.value_info {
        let inner = encode_value_info(vi);
        write_message_field(&mut buf, 12, &inner);
    }
    write_string_field(&mut buf, 13, &func.overload);
    for entry in &func.metadata_props {
        let inner = encode_string_string_entry(entry);
        write_message_field(&mut buf, 14, &inner);
    }
    buf
}

fn encode_model(model: &Model<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_varint_field(&mut buf, 1, model.ir_version as u64);
    write_string_field(&mut buf, 2, &model.producer_name);
    write_string_field(&mut buf, 3, &model.producer_version);
    write_string_field(&mut buf, 4, &model.domain);
    write_varint_field(&mut buf, 5, model.model_version as u64);
    write_string_field(&mut buf, 6, &model.doc_string);
    if let Some(graph) = &model.graph {
        let inner = encode_graph(graph);
        write_message_field(&mut buf, 7, &inner);
    }
    for opset in &model.opset_import {
        let inner = encode_opset_id(opset);
        write_message_field(&mut buf, 8, &inner);
    }
    for entry in &model.metadata_props {
        let inner = encode_string_string_entry(entry);
        write_message_field(&mut buf, 14, &inner);
    }
    for ti in &model.training_info {
        let inner = encode_training_info(ti);
        write_message_field(&mut buf, 20, &inner);
    }
    for func in &model.functions {
        let inner = encode_function(func);
        write_message_field(&mut buf, 25, &inner);
    }
    buf
}
