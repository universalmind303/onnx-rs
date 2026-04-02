use onnx_rs::ast::*;

fn main() {
    // Build a simple model AST by hand (shows the types without needing a .onnx file)
    let model = Model {
        ir_version: 8,
        producer_name: "example".to_string(),
        opset_import: vec![OperatorSetId {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(Graph {
            name: "main".to_string(),
            node: vec![
                Node {
                    name: "conv1".to_string(),
                    op_type: OpType::Conv,
                    input: vec!["input".into(), "conv1.weight".into(), "conv1.bias".into()],
                    output: vec!["conv1_out".into()],
                    attribute: vec![Attribute {
                        name: "kernel_shape".to_string(),
                        r#type: AttributeType::Ints,
                        ints: vec![3, 3],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Node {
                    name: "bn1".to_string(),
                    op_type: OpType::BatchNormalization,
                    input: vec![
                        "conv1_out".into(),
                        "bn1.scale".into(),
                        "bn1.bias".into(),
                        "bn1.mean".into(),
                        "bn1.var".into(),
                    ],
                    output: vec!["bn1_out".into()],
                    ..Default::default()
                },
                Node {
                    name: "relu1".to_string(),
                    op_type: OpType::Relu,
                    input: vec!["bn1_out".into()],
                    output: vec!["relu1_out".into()],
                    ..Default::default()
                },
                Node {
                    name: "pool1".to_string(),
                    op_type: OpType::GlobalAveragePool,
                    input: vec!["relu1_out".into()],
                    output: vec!["pool1_out".into()],
                    ..Default::default()
                },
                Node {
                    name: "fc".to_string(),
                    op_type: OpType::Gemm,
                    input: vec!["pool1_out".into(), "fc.weight".into(), "fc.bias".into()],
                    output: vec!["output".into()],
                    ..Default::default()
                },
            ],
            input: vec![ValueInfo {
                name: "input".to_string(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Tensor(TensorTypeProto {
                        elem_type: DataType::Float,
                        shape: Some(TensorShape {
                            dim: vec![
                                TensorShapeDimension {
                                    value: Dimension::Param("batch".into()),
                                    ..Default::default()
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
                    ..Default::default()
                }),
                ..Default::default()
            }],
            output: vec![ValueInfo {
                name: "output".to_string(),
                r#type: Some(TypeProto {
                    value: Some(TypeValue::Tensor(TensorTypeProto {
                        elem_type: DataType::Float,
                        shape: Some(TensorShape {
                            dim: vec![
                                TensorShapeDimension {
                                    value: Dimension::Param("batch".into()),
                                    ..Default::default()
                                },
                                TensorShapeDimension {
                                    value: Dimension::Value(1000),
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

    let graph = model.graph.as_ref().unwrap();

    // Pattern match on op types
    println!("Walking the graph:\n");
    for node in &graph.node {
        match &node.op_type {
            OpType::Conv => {
                let kernel = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "kernel_shape")
                    .map(|a| &a.ints);
                println!("  Found Conv '{}' with kernel {:?}", node.name, kernel.unwrap());
            }
            OpType::BatchNormalization => {
                println!("  Found BatchNorm '{}'", node.name);
            }
            OpType::Relu | OpType::LeakyRelu | OpType::Gelu | OpType::Sigmoid => {
                println!("  Found activation '{}' ({})", node.name, node.op_type);
            }
            OpType::Gemm | OpType::MatMul => {
                println!("  Found linear '{}' ({})", node.name, node.op_type);
            }
            op => {
                println!("  Found other op '{}' ({})", node.name, op);
            }
        }
    }

    // Find all ops that consume a given tensor
    let target = "conv1_out";
    println!("\nConsumers of '{target}':");
    for node in &graph.node {
        if node.input.iter().any(|i| i == target) {
            println!("  {} ({})", node.name, node.op_type);
        }
    }

    // Check if graph uses any quantization ops
    let has_quant = graph.node.iter().any(|n| {
        matches!(
            n.op_type,
            OpType::QuantizeLinear | OpType::DequantizeLinear | OpType::QLinearConv
        )
    });
    println!("\nUses quantization: {has_quant}");
}
