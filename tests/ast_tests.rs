use onnx_rs::ast::*;

// === Enum conversions ===

#[test]
fn test_data_type_try_from_valid() {
    assert_eq!(DataType::try_from(0i32).unwrap(), DataType::Undefined);
    assert_eq!(DataType::try_from(1i32).unwrap(), DataType::Float);
    assert_eq!(DataType::try_from(7i32).unwrap(), DataType::Int64);
    assert_eq!(DataType::try_from(9i32).unwrap(), DataType::Bool);
    assert_eq!(DataType::try_from(11i32).unwrap(), DataType::Double);
    assert_eq!(DataType::try_from(16i32).unwrap(), DataType::Bfloat16);
}

#[test]
fn test_data_type_try_from_invalid() {
    assert!(DataType::try_from(99i32).is_err());
    assert!(DataType::try_from(-1i32).is_err());
}

#[test]
fn test_data_type_default() {
    assert_eq!(DataType::default(), DataType::Undefined);
}

#[test]
fn test_attribute_type_try_from_valid() {
    assert_eq!(AttributeType::try_from(0i32).unwrap(), AttributeType::Undefined);
    assert_eq!(AttributeType::try_from(1i32).unwrap(), AttributeType::Float);
    assert_eq!(AttributeType::try_from(2i32).unwrap(), AttributeType::Int);
    assert_eq!(AttributeType::try_from(5i32).unwrap(), AttributeType::Graph);
    assert_eq!(AttributeType::try_from(14i32).unwrap(), AttributeType::TypeProtos);
}

#[test]
fn test_attribute_type_try_from_invalid() {
    assert!(AttributeType::try_from(99i32).is_err());
}

#[test]
fn test_attribute_type_default() {
    assert_eq!(AttributeType::default(), AttributeType::Undefined);
}

#[test]
fn test_data_location_try_from() {
    assert_eq!(DataLocation::try_from(0i32).unwrap(), DataLocation::Default);
    assert_eq!(DataLocation::try_from(1i32).unwrap(), DataLocation::External);
    assert!(DataLocation::try_from(2i32).is_err());
}

#[test]
fn test_data_location_default() {
    assert_eq!(DataLocation::default(), DataLocation::Default);
}

// === Default values ===

#[test]
fn test_model_default() {
    let model = Model::default();
    assert_eq!(model.ir_version, 0);
    assert!(model.opset_import.is_empty());
    assert!(model.producer_name.is_empty());
    assert!(model.graph.is_none());
    assert!(model.functions.is_empty());
}

#[test]
fn test_graph_default() {
    let graph = Graph::default();
    assert!(graph.node.is_empty());
    assert!(graph.name.is_empty());
    assert!(graph.initializer.is_empty());
    assert!(graph.input.is_empty());
    assert!(graph.output.is_empty());
}

#[test]
fn test_node_default() {
    let node = Node::default();
    assert!(node.input.is_empty());
    assert!(node.output.is_empty());
    assert_eq!(node.op_type, OpType::Custom(String::new()));
    assert!(node.attribute.is_empty());
}

#[test]
fn test_tensor_proto_default() {
    let tensor = TensorProto::default();
    assert!(tensor.dims.is_empty());
    assert_eq!(tensor.data_type, DataType::Undefined);
    assert!(tensor.float_data.is_empty());
    assert!(tensor.raw_data.is_empty());
    assert_eq!(tensor.data_location, DataLocation::Default);
}

#[test]
fn test_attribute_default() {
    let attr = Attribute::default();
    assert!(attr.name.is_empty());
    assert_eq!(attr.r#type, AttributeType::Undefined);
    assert_eq!(attr.f, 0.0);
    assert_eq!(attr.i, 0);
    assert!(attr.t.is_none());
    assert!(attr.g.is_none());
}

// === Derive traits ===

#[test]
fn test_model_debug_clone() {
    let model = Model::default();
    let cloned = model.clone();
    assert_eq!(model, cloned);
    let _ = format!("{:?}", model);
}

#[test]
fn test_dimension_variants() {
    let v = Dimension::Value(42);
    let p = Dimension::Param("batch".to_string());
    assert_ne!(v, p);
    let v2 = v.clone();
    assert_eq!(v, v2);
}

#[test]
fn test_type_value_variants() {
    let tv = TypeValue::Tensor(TensorTypeProto {
        elem_type: DataType::Float,
        shape: None,
    });
    let _ = format!("{:?}", tv);

    let sv = TypeValue::Sequence(SequenceTypeProto {
        elem_type: Box::new(TypeProto::default()),
    });
    assert_ne!(tv, sv);
}
