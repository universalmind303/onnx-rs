use onnx_rs::ast::*;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: parse_and_walk <model.onnx>");

    let bytes = std::fs::read(&path).expect("failed to read file");
    let model = onnx_rs::parse(&bytes).expect("failed to parse ONNX model");

    println!("=== ONNX Model ===");
    println!("IR version: {}", model.ir_version);
    println!("Producer:   {} {}", model.producer_name, model.producer_version);
    if !model.domain.is_empty() {
        println!("Domain:     {}", model.domain);
    }

    println!("\nOpset imports:");
    for opset in &model.opset_import {
        let domain = if opset.domain.is_empty() {
            "ai.onnx (default)"
        } else {
            &opset.domain
        };
        println!("  {} v{}", domain, opset.version);
    }

    if let Some(graph) = &model.graph {
        println!("\n=== Graph: {} ===", graph.name);

        println!("\nInputs:");
        for input in &graph.input {
            println!("  {} — {}", input.name, format_type(&input.r#type));
        }

        println!("\nOutputs:");
        for output in &graph.output {
            println!("  {} — {}", output.name, format_type(&output.r#type));
        }

        println!("\nInitializers: {}", graph.initializer.len());
        for init in &graph.initializer {
            println!("  {} — {:?} {:?}", init.name, init.data_type, init.dims);
        }

        println!("\nNodes ({}):", graph.node.len());
        for (i, node) in graph.node.iter().enumerate() {
            println!(
                "  [{i}] {} ({})",
                node.op_type,
                if node.name.is_empty() { "<unnamed>" } else { &node.name },
            );
            println!("       inputs:  {:?}", node.input);
            println!("       outputs: {:?}", node.output);

            for attr in &node.attribute {
                print!("       @{}: ", attr.name);
                match attr.r#type {
                    AttributeType::Int => println!("{}", attr.i),
                    AttributeType::Float => println!("{}", attr.f),
                    AttributeType::String => {
                        println!("{}", String::from_utf8_lossy(&attr.s))
                    }
                    AttributeType::Ints => println!("{:?}", attr.ints),
                    AttributeType::Floats => println!("{:?}", attr.floats),
                    AttributeType::Tensor => println!("<tensor>"),
                    AttributeType::Graph => println!("<graph>"),
                    _ => println!("<{:?}>", attr.r#type),
                }
            }
        }

        // Walk the graph and count ops by type
        println!("\n=== Op histogram ===");
        let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        count_ops(graph, &mut counts);
        for (op, count) in &counts {
            println!("  {op}: {count}");
        }
    }

    if !model.functions.is_empty() {
        println!("\n=== Functions ===");
        for func in &model.functions {
            println!("  {} (domain: {})", func.name, func.domain);
            println!("    inputs:  {:?}", func.input);
            println!("    outputs: {:?}", func.output);
            println!("    nodes:   {}", func.node.len());
        }
    }
}

fn count_ops(graph: &Graph, counts: &mut std::collections::BTreeMap<String, usize>) {
    for node in &graph.node {
        *counts.entry(node.op_type.to_string()).or_default() += 1;

        // Recurse into subgraphs (If/Loop/Scan nodes have graph attributes)
        for attr in &node.attribute {
            if let Some(g) = &attr.g {
                count_ops(g, counts);
            }
            for g in &attr.graphs {
                count_ops(g, counts);
            }
        }
    }
}

fn format_type(tp: &Option<TypeProto>) -> String {
    match tp {
        None => "unknown".to_string(),
        Some(tp) => match &tp.value {
            None => "unknown".to_string(),
            Some(TypeValue::Tensor(t)) => {
                let shape = match &t.shape {
                    None => "?".to_string(),
                    Some(s) => {
                        let dims: Vec<String> = s
                            .dim
                            .iter()
                            .map(|d| match &d.value {
                                Dimension::Value(v) => v.to_string(),
                                Dimension::Param(p) => p.clone(),
                            })
                            .collect();
                        format!("[{}]", dims.join(", "))
                    }
                };
                format!("{:?}{}", t.elem_type, shape)
            }
            Some(TypeValue::Sequence(s)) => {
                format!("Sequence({})", format_type(&Some(*s.elem_type.clone())))
            }
            Some(TypeValue::Map(m)) => {
                format!(
                    "Map({:?}, {})",
                    m.key_type,
                    format_type(&Some(*m.value_type.clone()))
                )
            }
            Some(TypeValue::Optional(o)) => {
                format!("Optional({})", format_type(&Some(*o.elem_type.clone())))
            }
            Some(TypeValue::SparseTensor(t)) => format!("SparseTensor({:?})", t.elem_type),
        },
    }
}
