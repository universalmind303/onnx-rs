use onnx_rs::ast::*;
use onnx_rs::{encode, parse};

#[test]
fn test_roundtrip_empty_model() {
    let model = Model::default();
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_model_scalars() {
    let model = Model {
        ir_version: 8,
        producer_name: "onnx-rs".to_string(),
        producer_version: "0.1.0".to_string(),
        domain: "ai.onnx".to_string(),
        model_version: 1,
        doc_string: "a test model".to_string(),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_opset_imports() {
    let model = Model {
        ir_version: 7,
        opset_import: vec![
            OperatorSetId {
                domain: String::new(),
                version: 13,
            },
            OperatorSetId {
                domain: "ai.onnx.ml".to_string(),
                version: 3,
            },
        ],
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_metadata_props() {
    let model = Model {
        metadata_props: vec![
            StringStringEntry {
                key: "author".to_string(),
                value: "test".to_string(),
            },
            StringStringEntry {
                key: "version".to_string(),
                value: "1.0".to_string(),
            },
        ],
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_simple_graph() {
    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(Graph {
            name: "main".to_string(),
            node: vec![Node {
                name: "relu0".to_string(),
                op_type: OpType::Relu,
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            }],
            input: vec![ValueInfo {
                name: "x".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfo {
                name: "y".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_node_with_attributes() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Conv,
                input: vec!["x".into(), "w".into(), "b".into()],
                output: vec!["y".into()],
                attribute: vec![
                    Attribute {
                        name: "kernel_shape".to_string(),
                        r#type: AttributeType::Ints,
                        ints: vec![3, 3],
                        ..Default::default()
                    },
                    Attribute {
                        name: "strides".to_string(),
                        r#type: AttributeType::Ints,
                        ints: vec![1, 1],
                        ..Default::default()
                    },
                    Attribute {
                        name: "alpha".to_string(),
                        r#type: AttributeType::Float,
                        f: 0.25,
                        ..Default::default()
                    },
                    Attribute {
                        name: "axis".to_string(),
                        r#type: AttributeType::Int,
                        i: 1,
                        ..Default::default()
                    },
                    Attribute {
                        name: "mode".to_string(),
                        r#type: AttributeType::String,
                        s: b"constant".to_vec(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_value_info_with_tensor_type() {
    let model = Model {
        graph: Some(Graph {
            input: vec![ValueInfo {
                name: "input".to_string(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Tensor(TensorTypeProto {
                        elem_type: DataType::Float,
                        shape: Some(TensorShape {
                            dim: vec![
                                TensorShapeDimension {
                                    value: Dimension::Param("batch".into()),
                                    denotation: "DATA_BATCH".into(),
                                },
                                TensorShapeDimension {
                                    value: Dimension::Value(3),
                                    ..Default::default()
                                },
                                TensorShapeDimension {
                                    value: Dimension::Value(224),
                                    ..Default::default()
                                },
                                TensorShapeDimension {
                                    value: Dimension::Value(224),
                                    ..Default::default()
                                },
                            ],
                        }),
                    })),
                    denotation: "IMAGE".into(),
                }),
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_sequence_type() {
    let model = Model {
        graph: Some(Graph {
            input: vec![ValueInfo {
                name: "seq".to_string(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Sequence(SequenceTypeProto {
                        elem_type: Box::new(TypeProto {
                            value: Some(TypeValue::Tensor(TensorTypeProto {
                                elem_type: DataType::Int64,
                                shape: None,
                            })),
                            ..Default::default()
                        }),
                    })),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_map_type() {
    let model = Model {
        graph: Some(Graph {
            input: vec![ValueInfo {
                name: "m".to_string(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Map(MapTypeProto {
                        key_type: DataType::String,
                        value_type: Box::new(TypeProto {
                            value: Some(TypeValue::Tensor(TensorTypeProto {
                                elem_type: DataType::Float,
                                shape: None,
                            })),
                            ..Default::default()
                        }),
                    })),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_tensor_proto_float_data() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "weight".to_string(),
                dims: vec![2, 3],
                data_type: DataType::Float,
                float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_tensor_proto_raw_data() {
    let raw: Vec<u8> = [1.0f32, 2.0, 3.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "w".to_string(),
                dims: vec![3],
                data_type: DataType::Float,
                raw_data: raw,
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_tensor_proto_int64_data() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "indices".to_string(),
                dims: vec![4],
                data_type: DataType::Int64,
                int64_data: vec![10, 20, 30, 40],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_tensor_proto_double_data() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "d".to_string(),
                data_type: DataType::Double,
                double_data: vec![3.14, 2.718],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_tensor_proto_string_data() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "labels".to_string(),
                data_type: DataType::String,
                string_data: vec![b"cat".to_vec(), b"dog".to_vec()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_sparse_tensor() {
    let model = Model {
        graph: Some(Graph {
            sparse_initializer: vec![SparseTensor {
                dims: vec![3, 4],
                values: Some(TensorProto {
                    data_type: DataType::Float,
                    float_data: vec![1.0, 2.0],
                    ..Default::default()
                }),
                indices: Some(TensorProto {
                    data_type: DataType::Int64,
                    int64_data: vec![0, 5],
                    ..Default::default()
                }),
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_training_info() {
    let model = Model {
        training_info: vec![TrainingInfo {
            initialization: Some(Graph {
                name: "init".to_string(),
                ..Default::default()
            }),
            algorithm: Some(Graph {
                name: "train_step".to_string(),
                ..Default::default()
            }),
            initialization_binding: vec![StringStringEntry {
                key: "a".to_string(),
                value: "b".to_string(),
            }],
            update_binding: vec![StringStringEntry {
                key: "c".to_string(),
                value: "d".to_string(),
            }],
        }],
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_function() {
    let model = Model {
        functions: vec![Function {
            name: "MyFunc".to_string(),
            domain: "com.test".to_string(),
            input: vec!["x".into()],
            output: vec!["y".into()],
            attribute: vec!["alpha".into()],
            node: vec![Node {
                op_type: OpType::Relu,
                input: vec!["x".into()],
                output: vec!["y".into()],
                ..Default::default()
            }],
            opset_import: vec![OperatorSetId {
                domain: String::new(),
                version: 17,
            }],
            ..Default::default()
        }],
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_node_domain_overload() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Custom("MyOp".to_string()),
                domain: "com.vendor".to_string(),
                overload: "v2".to_string(),
                doc_string: "a custom op".to_string(),
                input: vec!["in".into()],
                output: vec!["out".into()],
                metadata_props: vec![StringStringEntry {
                    key: "k".into(),
                    value: "v".into(),
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_attribute_floats() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Custom("Op".into()),
                attribute: vec![Attribute {
                    name: "scales".into(),
                    r#type: AttributeType::Floats,
                    floats: vec![0.5, 1.0, 1.5],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_attribute_strings() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Custom("Op".into()),
                attribute: vec![Attribute {
                    name: "labels".into(),
                    r#type: AttributeType::Strings,
                    strings: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_attribute_tensor() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Constant,
                attribute: vec![Attribute {
                    name: "value".into(),
                    r#type: AttributeType::Tensor,
                    t: Some(TensorProto {
                        dims: vec![2, 2],
                        data_type: DataType::Float,
                        float_data: vec![1.0, 2.0, 3.0, 4.0],
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_attribute_graph() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::If,
                attribute: vec![Attribute {
                    name: "then_branch".into(),
                    r#type: AttributeType::Graph,
                    g: Some(Box::new(Graph {
                        name: "then".into(),
                        node: vec![Node {
                            op_type: OpType::Identity,
                            input: vec!["x".into()],
                            output: vec!["y".into()],
                            ..Default::default()
                        }],
                        ..Default::default()
                    })),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_tensor_external_data() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "ext".to_string(),
                data_type: DataType::Float,
                data_location: DataLocation::External,
                external_data: vec![
                    StringStringEntry {
                        key: "location".into(),
                        value: "weights.bin".into(),
                    },
                    StringStringEntry {
                        key: "offset".into(),
                        value: "0".into(),
                    },
                    StringStringEntry {
                        key: "length".into(),
                        value: "1024".into(),
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_roundtrip_real_squeezenet() {
    let path = "/tmp/squeezenet.onnx";
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping squeezenet roundtrip test (file not found at {path})");
        return;
    }
    let original_bytes = std::fs::read(path).unwrap();
    let model = parse(&original_bytes).unwrap();
    let re_encoded = encode(&model);
    let reparsed = parse(&re_encoded).unwrap();
    assert_eq!(model, reparsed);
}
