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
        producer_name: "onnx-rs",
        producer_version: "0.1.0",
        domain: "ai.onnx",
        model_version: 1,
        doc_string: "a test model",
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
                domain: "",
                version: 13,
            },
            OperatorSetId {
                domain: "ai.onnx.ml",
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
                key: "author",
                value: "test",
            },
            StringStringEntry {
                key: "version",
                value: "1.0",
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
            domain: "",
            version: 17,
        }],
        graph: Some(Graph {
            name: "main",
            node: vec![Node {
                name: "relu0",
                op_type: OpType::Relu,
                input: vec!["x"],
                output: vec!["y"],
                ..Default::default()
            }],
            input: vec![ValueInfo {
                name: "x",
                ..Default::default()
            }],
            output: vec![ValueInfo {
                name: "y",
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
                input: vec!["x", "w", "b"],
                output: vec!["y"],
                attribute: vec![
                    Attribute {
                        name: "kernel_shape",
                        r#type: AttributeType::Ints,
                        ints: vec![3, 3],
                        ..Default::default()
                    },
                    Attribute {
                        name: "strides",
                        r#type: AttributeType::Ints,
                        ints: vec![1, 1],
                        ..Default::default()
                    },
                    Attribute {
                        name: "alpha",
                        r#type: AttributeType::Float,
                        f: 0.25,
                        ..Default::default()
                    },
                    Attribute {
                        name: "axis",
                        r#type: AttributeType::Int,
                        i: 1,
                        ..Default::default()
                    },
                    Attribute {
                        name: "mode",
                        r#type: AttributeType::String,
                        s: b"constant",
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
                name: "input",
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Tensor(TensorTypeProto {
                        elem_type: DataType::Float,
                        shape: Some(TensorShape {
                            dim: vec![
                                TensorShapeDimension {
                                    value: Dimension::Param("batch"),
                                    denotation: "DATA_BATCH",
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
                    denotation: "IMAGE",
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
                name: "seq",
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
                name: "m",
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
            initializer: vec![TensorProto::from_f32("weight", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
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
            initializer: vec![TensorProto::from_raw("w", vec![3], DataType::Float, &raw)],
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
            initializer: vec![TensorProto::from_i64("indices", vec![4], vec![10, 20, 30, 40])],
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
            initializer: vec![TensorProto::from_f64("d", vec![], vec![3.14, 2.718])],
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
            initializer: vec![TensorProto::from_strings("labels", vec![], vec![b"cat", b"dog"])],
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
                values: Some(TensorProto::from_f32("", vec![], vec![1.0, 2.0])),
                indices: Some(TensorProto::from_i64("", vec![], vec![0, 5])),
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
                name: "init",
                ..Default::default()
            }),
            algorithm: Some(Graph {
                name: "train_step",
                ..Default::default()
            }),
            initialization_binding: vec![StringStringEntry {
                key: "a",
                value: "b",
            }],
            update_binding: vec![StringStringEntry {
                key: "c",
                value: "d",
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
            name: "MyFunc",
            domain: "com.test",
            input: vec!["x"],
            output: vec!["y"],
            attribute: vec!["alpha"],
            node: vec![Node {
                op_type: OpType::Relu,
                input: vec!["x"],
                output: vec!["y"],
                ..Default::default()
            }],
            opset_import: vec![OperatorSetId {
                domain: "",
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
                op_type: OpType::Custom("MyOp"),
                domain: "com.vendor",
                overload: "v2",
                doc_string: "a custom op",
                input: vec!["in"],
                output: vec!["out"],
                metadata_props: vec![StringStringEntry {
                    key: "k",
                    value: "v",
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
                op_type: OpType::Custom("Op"),
                attribute: vec![Attribute {
                    name: "scales",
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
                op_type: OpType::Custom("Op"),
                attribute: vec![Attribute {
                    name: "labels",
                    r#type: AttributeType::Strings,
                    strings: vec![b"a", b"b", b"c"],
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
                    name: "value",
                    r#type: AttributeType::Tensor,
                    t: Some(TensorProto::from_f32("", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])),
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
                    name: "then_branch",
                    r#type: AttributeType::Graph,
                    g: Some(Box::new(Graph {
                        name: "then",
                        node: vec![Node {
                            op_type: OpType::Identity,
                            input: vec!["x"],
                            output: vec!["y"],
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
            initializer: vec![
                TensorProto::from_raw("ext", vec![], DataType::Float, &[])
                    .with_data_location(DataLocation::External)
                    .with_external_data(vec![
                        StringStringEntry {
                            key: "location",
                            value: "weights.bin",
                        },
                        StringStringEntry {
                            key: "offset",
                            value: "0",
                        },
                        StringStringEntry {
                            key: "length",
                            value: "1024",
                        },
                    ])
            ],
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
