use onnx_rs::ast::*;
use onnx_rs::{encode, parse};

// ============================================================
// GAP 1: Missing DataType variants (Float8, Int4, Uint4, etc.)
// ============================================================

#[test]
fn test_data_type_float8e4m3fn() {
    assert_eq!(DataType::try_from(17i32).unwrap(), DataType::Float8e4m3fn);
}

#[test]
fn test_data_type_float8e4m3fnuz() {
    assert_eq!(DataType::try_from(18i32).unwrap(), DataType::Float8e4m3fnuz);
}

#[test]
fn test_data_type_float8e5m2() {
    assert_eq!(DataType::try_from(19i32).unwrap(), DataType::Float8e5m2);
}

#[test]
fn test_data_type_float8e5m2fnuz() {
    assert_eq!(DataType::try_from(20i32).unwrap(), DataType::Float8e5m2fnuz);
}

#[test]
fn test_data_type_uint4() {
    assert_eq!(DataType::try_from(21i32).unwrap(), DataType::Uint4);
}

#[test]
fn test_data_type_int4() {
    assert_eq!(DataType::try_from(22i32).unwrap(), DataType::Int4);
}

#[test]
fn test_data_type_float4e2m1() {
    assert_eq!(DataType::try_from(23i32).unwrap(), DataType::Float4e2m1);
}

#[test]
fn test_roundtrip_tensor_with_float8_type() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "f8_weight".to_string(),
                dims: vec![4],
                data_type: DataType::Float8e4m3fn,
                raw_data: vec![0x10, 0x20, 0x30, 0x40],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(parsed.graph.unwrap().initializer[0].data_type, DataType::Float8e4m3fn);
}

#[test]
fn test_roundtrip_tensor_with_int4_type() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "i4".to_string(),
                dims: vec![8],
                data_type: DataType::Int4,
                raw_data: vec![0x12, 0x34, 0x56, 0x78],
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

// ============================================================
// GAP 2: Default/zero value handling
// ============================================================

#[test]
fn test_zero_ir_version_roundtrip() {
    // ir_version=0 is valid and should survive a roundtrip
    // In proto3, 0 is default and won't be serialized — that's correct behavior
    let model = Model {
        ir_version: 0,
        producer_name: "test".to_string(),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(parsed.ir_version, 0);
    assert_eq!(parsed.producer_name, "test");
}

#[test]
fn test_model_with_no_graph() {
    let model = Model {
        ir_version: 7,
        opset_import: vec![OperatorSetId {
            domain: String::new(),
            version: 13,
        }],
        graph: None,
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert!(parsed.graph.is_none());
    assert_eq!(parsed.ir_version, 7);
}

#[test]
fn test_default_attribute_values_roundtrip() {
    // Attribute with f=0.0 — zero float should roundtrip correctly
    // In proto3, f=0.0 won't be serialized, so on parse it comes back as 0.0 default
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Relu,
                attribute: vec![Attribute {
                    name: "test_attr".to_string(),
                    r#type: AttributeType::Int,
                    i: 0,
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
    let attr = &parsed.graph.unwrap().node[0].attribute[0];
    assert_eq!(attr.name, "test_attr");
    assert_eq!(attr.i, 0);
    assert_eq!(attr.f, 0.0);
}

// ============================================================
// GAP 3: Negative integer values
// ============================================================

#[test]
fn test_negative_int64_in_tensor() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "neg".to_string(),
                data_type: DataType::Int64,
                int64_data: vec![-1, -100, -9999, i64::MIN],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(
        parsed.graph.unwrap().initializer[0].int64_data,
        vec![-1, -100, -9999, i64::MIN]
    );
}

#[test]
fn test_negative_int32_in_tensor() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "neg32".to_string(),
                data_type: DataType::Int32,
                int32_data: vec![-1, -42, i32::MIN],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(
        parsed.graph.unwrap().initializer[0].int32_data,
        vec![-1, -42, i32::MIN]
    );
}

#[test]
fn test_negative_attribute_int() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Concat,
                attribute: vec![Attribute {
                    name: "axis".to_string(),
                    r#type: AttributeType::Int,
                    i: -1,
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
    assert_eq!(parsed.graph.unwrap().node[0].attribute[0].i, -1);
}

#[test]
fn test_negative_ints_attribute() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Unsqueeze,
                attribute: vec![Attribute {
                    name: "axes".to_string(),
                    r#type: AttributeType::Ints,
                    ints: vec![-1, -2, 0, 1],
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
    assert_eq!(
        parsed.graph.unwrap().node[0].attribute[0].ints,
        vec![-1, -2, 0, 1]
    );
}

// ============================================================
// GAP 4: Unicode strings
// ============================================================

#[test]
fn test_unicode_doc_string() {
    let model = Model {
        doc_string: "模型描述 — réseau de neurones 🧠".to_string(),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(parsed.doc_string, "模型描述 — réseau de neurones 🧠");
}

#[test]
fn test_unicode_node_name() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                name: "层_1".to_string(),
                op_type: OpType::Relu,
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(parsed.graph.unwrap().node[0].name, "层_1");
}

// ============================================================
// GAP 5: Scalar tensors (empty dims)
// ============================================================

#[test]
fn test_scalar_tensor_roundtrip() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "scalar".to_string(),
                dims: vec![],
                data_type: DataType::Float,
                float_data: vec![3.14],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    let t = &parsed.graph.unwrap().initializer[0];
    assert!(t.dims.is_empty());
    assert_eq!(t.float_data, vec![3.14f32]);
}

// ============================================================
// GAP 6: ref_attr_name in attributes (function contexts)
// ============================================================

#[test]
fn test_ref_attr_name_roundtrip() {
    let model = Model {
        functions: vec![Function {
            name: "MyFunc".to_string(),
            attribute: vec!["alpha".to_string()],
            node: vec![Node {
                op_type: OpType::LeakyRelu,
                input: vec!["x".into()],
                output: vec!["y".into()],
                attribute: vec![Attribute {
                    name: "alpha".to_string(),
                    r#type: AttributeType::Float,
                    ref_attr_name: "alpha".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            input: vec!["x".into()],
            output: vec!["y".into()],
            ..Default::default()
        }],
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(
        parsed.functions[0].node[0].attribute[0].ref_attr_name,
        "alpha"
    );
}

// ============================================================
// GAP 7: TypeProto attributes
// ============================================================

#[test]
fn test_attribute_type_proto_roundtrip() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Custom("TypedOp".into()),
                attribute: vec![Attribute {
                    name: "output_type".to_string(),
                    r#type: AttributeType::TypeProto,
                    tp: Some(TypeProto {
                        value: Some(TypeValue::Tensor(TensorTypeProto {
                            elem_type: DataType::Float,
                            shape: Some(TensorShape {
                                dim: vec![TensorShapeDimension {
                                    value: Dimension::Value(10),
                                    ..Default::default()
                                }],
                            }),
                        })),
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
    let attr = &parsed.graph.unwrap().node[0].attribute[0];
    assert!(attr.tp.is_some());
    match attr.tp.as_ref().unwrap().value.as_ref().unwrap() {
        TypeValue::Tensor(t) => assert_eq!(t.elem_type, DataType::Float),
        _ => panic!("expected tensor type"),
    }
}

#[test]
fn test_attribute_type_protos_repeated_roundtrip() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Custom("MultiTypedOp".into()),
                attribute: vec![Attribute {
                    name: "types".to_string(),
                    r#type: AttributeType::TypeProtos,
                    type_protos: vec![
                        TypeProto {
                            value: Some(TypeValue::Tensor(TensorTypeProto {
                                elem_type: DataType::Float,
                                shape: None,
                            })),
                            ..Default::default()
                        },
                        TypeProto {
                            value: Some(TypeValue::Tensor(TensorTypeProto {
                                elem_type: DataType::Int64,
                                shape: None,
                            })),
                            ..Default::default()
                        },
                    ],
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
    let attr = &parsed.graph.unwrap().node[0].attribute[0];
    assert_eq!(attr.type_protos.len(), 2);
}

// ============================================================
// GAP 8: Multiple graph attributes (If with then/else)
// ============================================================

#[test]
fn test_if_node_with_then_else_branches() {
    let then_graph = Graph {
        name: "then_branch".to_string(),
        node: vec![Node {
            op_type: OpType::Constant,
            output: vec!["then_out".into()],
            attribute: vec![Attribute {
                name: "value".into(),
                r#type: AttributeType::Tensor,
                t: Some(TensorProto {
                    data_type: DataType::Float,
                    float_data: vec![1.0],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }],
        output: vec![ValueInfo {
            name: "then_out".into(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let else_graph = Graph {
        name: "else_branch".to_string(),
        node: vec![Node {
            op_type: OpType::Constant,
            output: vec!["else_out".into()],
            attribute: vec![Attribute {
                name: "value".into(),
                r#type: AttributeType::Tensor,
                t: Some(TensorProto {
                    data_type: DataType::Float,
                    float_data: vec![0.0],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }],
        output: vec![ValueInfo {
            name: "else_out".into(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::If,
                input: vec!["cond".into()],
                output: vec!["result".into()],
                attribute: vec![
                    Attribute {
                        name: "then_branch".to_string(),
                        r#type: AttributeType::Graph,
                        g: Some(Box::new(then_graph)),
                        ..Default::default()
                    },
                    Attribute {
                        name: "else_branch".to_string(),
                        r#type: AttributeType::Graph,
                        g: Some(Box::new(else_graph)),
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
    let node = &parsed.graph.unwrap().node[0];
    assert_eq!(node.op_type, OpType::If);
    assert_eq!(node.attribute.len(), 2);
    assert_eq!(node.attribute[0].name, "then_branch");
    assert_eq!(node.attribute[0].g.as_ref().unwrap().name, "then_branch");
    assert_eq!(node.attribute[1].name, "else_branch");
    assert_eq!(node.attribute[1].g.as_ref().unwrap().name, "else_branch");
}

// ============================================================
// GAP 9: Optional/empty inputs in node
// ============================================================

#[test]
fn test_node_with_empty_optional_inputs() {
    // BatchNormalization has 5 inputs, some models use "" for optional ones
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::BatchNormalization,
                input: vec![
                    "x".into(),
                    "scale".into(),
                    "bias".into(),
                    "".into(), // optional mean
                    "".into(), // optional var
                ],
                output: vec!["y".into()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    let inputs = &parsed.graph.unwrap().node[0].input;
    assert_eq!(inputs.len(), 5);
    assert_eq!(inputs[3], "");
    assert_eq!(inputs[4], "");
}

// ============================================================
// GAP 10: Quantization annotations
// ============================================================

#[test]
fn test_quantization_annotation_roundtrip() {
    let model = Model {
        graph: Some(Graph {
            quantization_annotation: vec![TensorAnnotation {
                tensor_name: "conv1_weight".to_string(),
                quant_parameter_tensor_names: vec![
                    StringStringEntry {
                        key: "SCALE_TENSOR".into(),
                        value: "conv1_weight_scale".into(),
                    },
                    StringStringEntry {
                        key: "ZERO_POINT_TENSOR".into(),
                        value: "conv1_weight_zp".into(),
                    },
                ],
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    let ann = &parsed.graph.unwrap().quantization_annotation;
    assert_eq!(ann.len(), 1);
    assert_eq!(ann[0].tensor_name, "conv1_weight");
    assert_eq!(ann[0].quant_parameter_tensor_names.len(), 2);
    assert_eq!(ann[0].quant_parameter_tensor_names[0].key, "SCALE_TENSOR");
}

// ============================================================
// GAP 11: Symbolic dimensions with special characters
// ============================================================

#[test]
fn test_symbolic_dim_with_operators() {
    let model = Model {
        graph: Some(Graph {
            input: vec![ValueInfo {
                name: "x".to_string(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Tensor(TensorTypeProto {
                        elem_type: DataType::Float,
                        shape: Some(TensorShape {
                            dim: vec![
                                TensorShapeDimension {
                                    value: Dimension::Param("M + N".into()),
                                    ..Default::default()
                                },
                                TensorShapeDimension {
                                    value: Dimension::Param("batch * 2".into()),
                                    ..Default::default()
                                },
                            ],
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
    let g = parsed.graph.unwrap();
    let shape = match &g.input[0].r#type.as_ref().unwrap().value {
        Some(TypeValue::Tensor(t)) => t.shape.as_ref().unwrap(),
        _ => panic!("expected tensor type"),
    };
    assert_eq!(shape.dim[0].value, Dimension::Param("M + N".into()));
    assert_eq!(shape.dim[1].value, Dimension::Param("batch * 2".into()));
}

// ============================================================
// GAP 12: Value info for intermediate tensors
// ============================================================

#[test]
fn test_graph_value_info_intermediates() {
    let model = Model {
        graph: Some(Graph {
            node: vec![
                Node {
                    op_type: OpType::Relu,
                    input: vec!["x".into()],
                    output: vec!["relu_out".into()],
                    ..Default::default()
                },
                Node {
                    op_type: OpType::Sigmoid,
                    input: vec!["relu_out".into()],
                    output: vec!["y".into()],
                    ..Default::default()
                },
            ],
            input: vec![ValueInfo {
                name: "x".into(),
                ..Default::default()
            }],
            output: vec![ValueInfo {
                name: "y".into(),
                ..Default::default()
            }],
            value_info: vec![ValueInfo {
                name: "relu_out".into(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Tensor(TensorTypeProto {
                        elem_type: DataType::Float,
                        shape: Some(TensorShape {
                            dim: vec![TensorShapeDimension {
                                value: Dimension::Param("N".into()),
                                ..Default::default()
                            }],
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
    let g = parsed.graph.unwrap();
    assert_eq!(g.value_info.len(), 1);
    assert_eq!(g.value_info[0].name, "relu_out");
}

// ============================================================
// GAP 13: Optional type in TypeProto
// ============================================================

#[test]
fn test_optional_type_roundtrip() {
    let model = Model {
        graph: Some(Graph {
            input: vec![ValueInfo {
                name: "opt".into(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Optional(OptionalTypeProto {
                        elem_type: Box::new(TypeProto {
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
    match &parsed.graph.unwrap().input[0].r#type.as_ref().unwrap().value {
        Some(TypeValue::Optional(o)) => match &o.elem_type.value {
            Some(TypeValue::Tensor(t)) => assert_eq!(t.elem_type, DataType::Float),
            _ => panic!("expected tensor inside optional"),
        },
        _ => panic!("expected optional type"),
    }
}

// ============================================================
// GAP 14: SparseTensor type in TypeProto
// ============================================================

#[test]
fn test_sparse_tensor_type_roundtrip() {
    let model = Model {
        graph: Some(Graph {
            input: vec![ValueInfo {
                name: "sp".into(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::SparseTensor(SparseTensorTypeProto {
                        elem_type: DataType::Float,
                        shape: Some(TensorShape {
                            dim: vec![
                                TensorShapeDimension {
                                    value: Dimension::Value(100),
                                    ..Default::default()
                                },
                                TensorShapeDimension {
                                    value: Dimension::Value(200),
                                    ..Default::default()
                                },
                            ],
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
    match &parsed.graph.unwrap().input[0].r#type.as_ref().unwrap().value {
        Some(TypeValue::SparseTensor(st)) => {
            assert_eq!(st.elem_type, DataType::Float);
            assert_eq!(st.shape.as_ref().unwrap().dim.len(), 2);
        }
        _ => panic!("expected sparse tensor type"),
    }
}

// ============================================================
// GAP 15: Nested optional-of-sequence type
// ============================================================

#[test]
fn test_nested_optional_sequence_type() {
    let model = Model {
        graph: Some(Graph {
            input: vec![ValueInfo {
                name: "opt_seq".into(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Optional(OptionalTypeProto {
                        elem_type: Box::new(TypeProto {
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

// ============================================================
// GAP 16: Tensor segment field (deprecated but must parse)
// ============================================================

#[test]
fn test_tensor_segment_roundtrip() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "chunked".to_string(),
                data_type: DataType::Float,
                segment: Some(TensorSegment {
                    begin: 0,
                    end: 100,
                }),
                raw_data: vec![0; 400], // 100 floats
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    let g = parsed.graph.unwrap();
    let seg = g.initializer[0].segment.as_ref().unwrap();
    assert_eq!(seg.begin, 0);
    assert_eq!(seg.end, 100);
}

// ============================================================
// GAP 17: Large model with many nodes (stress test)
// ============================================================

#[test]
fn test_roundtrip_large_graph() {
    let nodes: Vec<Node> = (0..200)
        .map(|i| Node {
            name: format!("node_{i}"),
            op_type: if i % 3 == 0 {
                OpType::Conv
            } else if i % 3 == 1 {
                OpType::Relu
            } else {
                OpType::Add
            },
            input: vec![format!("t_{i}")],
            output: vec![format!("t_{}", i + 1)],
            ..Default::default()
        })
        .collect();

    let model = Model {
        ir_version: 8,
        graph: Some(Graph {
            name: "big".to_string(),
            node: nodes,
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(parsed.graph.unwrap().node.len(), 200);
}

// ============================================================
// GAP 18: Attribute with multiple tensors
// ============================================================

#[test]
fn test_attribute_tensors_repeated() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Custom("MultiTensor".into()),
                attribute: vec![Attribute {
                    name: "weights".into(),
                    r#type: AttributeType::Tensors,
                    tensors: vec![
                        TensorProto {
                            name: "w1".into(),
                            data_type: DataType::Float,
                            float_data: vec![1.0, 2.0],
                            ..Default::default()
                        },
                        TensorProto {
                            name: "w2".into(),
                            data_type: DataType::Float,
                            float_data: vec![3.0, 4.0],
                            ..Default::default()
                        },
                    ],
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
    let attr = &parsed.graph.unwrap().node[0].attribute[0];
    assert_eq!(attr.tensors.len(), 2);
    assert_eq!(attr.tensors[0].name, "w1");
    assert_eq!(attr.tensors[1].float_data, vec![3.0, 4.0]);
}

// ============================================================
// GAP 19: Attribute with repeated graphs (Scan body)
// ============================================================

#[test]
fn test_attribute_graphs_repeated() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Custom("MultiGraph".into()),
                attribute: vec![Attribute {
                    name: "bodies".into(),
                    r#type: AttributeType::Graphs,
                    graphs: vec![
                        Graph {
                            name: "body0".into(),
                            ..Default::default()
                        },
                        Graph {
                            name: "body1".into(),
                            ..Default::default()
                        },
                    ],
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
    let attr = &parsed.graph.unwrap().node[0].attribute[0];
    assert_eq!(attr.graphs.len(), 2);
    assert_eq!(attr.graphs[0].name, "body0");
    assert_eq!(attr.graphs[1].name, "body1");
}

// ============================================================
// GAP 20: Attribute with sparse tensor
// ============================================================

#[test]
fn test_attribute_sparse_tensor_roundtrip() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Constant,
                attribute: vec![Attribute {
                    name: "sparse_value".into(),
                    r#type: AttributeType::SparseTensor,
                    sparse_tensor: Some(SparseTensor {
                        dims: vec![10, 10],
                        values: Some(TensorProto {
                            data_type: DataType::Float,
                            float_data: vec![1.0, 2.0, 3.0],
                            ..Default::default()
                        }),
                        indices: Some(TensorProto {
                            data_type: DataType::Int64,
                            int64_data: vec![0, 5, 9],
                            ..Default::default()
                        }),
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
    let attr = &parsed.graph.unwrap().node[0].attribute[0];
    assert!(attr.sparse_tensor.is_some());
    assert_eq!(attr.sparse_tensor.as_ref().unwrap().dims, vec![10, 10]);
}

// ============================================================
// GAP 21: Function with attribute_proto defaults
// ============================================================

#[test]
fn test_function_attribute_proto_defaults() {
    let model = Model {
        functions: vec![Function {
            name: "MyFunc".into(),
            attribute_proto: vec![Attribute {
                name: "epsilon".into(),
                r#type: AttributeType::Float,
                f: 1e-5,
                ..Default::default()
            }],
            node: vec![Node {
                op_type: OpType::Add,
                input: vec!["x".into(), "eps".into()],
                output: vec!["y".into()],
                ..Default::default()
            }],
            input: vec!["x".into()],
            output: vec!["y".into()],
            ..Default::default()
        }],
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(parsed.functions[0].attribute_proto.len(), 1);
    assert_eq!(parsed.functions[0].attribute_proto[0].name, "epsilon");
    assert!((parsed.functions[0].attribute_proto[0].f - 1e-5).abs() < 1e-10);
}

// ============================================================
// GAP 22: uint64 data in tensors
// ============================================================

#[test]
fn test_tensor_uint64_data_roundtrip() {
    let model = Model {
        graph: Some(Graph {
            initializer: vec![TensorProto {
                name: "u64".into(),
                data_type: DataType::Uint64,
                uint64_data: vec![0, 1, u64::MAX, 42],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(
        parsed.graph.unwrap().initializer[0].uint64_data,
        vec![0, 1, u64::MAX, 42]
    );
}

// ============================================================
// GAP 23: Node with empty string op_type (edge case)
// ============================================================

#[test]
fn test_node_empty_op_type() {
    let model = Model {
        graph: Some(Graph {
            node: vec![Node {
                op_type: OpType::Custom(String::new()),
                input: vec!["x".into()],
                output: vec!["y".into()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = encode(&model);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(
        parsed.graph.unwrap().node[0].op_type,
        OpType::Custom(String::new())
    );
}

// ============================================================
// GAP 24: Denotation fields on type and dimensions
// ============================================================

#[test]
fn test_denotation_fields_roundtrip() {
    let model = Model {
        graph: Some(Graph {
            input: vec![ValueInfo {
                name: "image".into(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Tensor(TensorTypeProto {
                        elem_type: DataType::Float,
                        shape: Some(TensorShape {
                            dim: vec![
                                TensorShapeDimension {
                                    value: Dimension::Param("N".into()),
                                    denotation: "DATA_BATCH".into(),
                                },
                                TensorShapeDimension {
                                    value: Dimension::Value(3),
                                    denotation: "DATA_CHANNEL".into(),
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
    let tp = parsed.graph.unwrap().input[0].r#type.as_ref().unwrap().clone();
    assert_eq!(tp.denotation, "IMAGE");
    match &tp.value.unwrap() {
        TypeValue::Tensor(t) => {
            let dims = &t.shape.as_ref().unwrap().dim;
            assert_eq!(dims[0].denotation, "DATA_BATCH");
            assert_eq!(dims[1].denotation, "DATA_CHANNEL");
        }
        _ => panic!("expected tensor"),
    }
}
