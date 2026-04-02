use crate::ast::*;
use crate::error::Error;
use crate::wire::{Cursor, WireType};

pub fn parse(data: &[u8]) -> Result<Model, Error> {
    let mut cursor = Cursor::new(data);
    parse_model(&mut cursor)
}

fn parse_string_string_entry(cursor: &mut Cursor) -> Result<StringStringEntry, Error> {
    let mut entry = StringStringEntry::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => entry.key = cursor.read_string()?.to_string(),
            2 => entry.value = cursor.read_string()?.to_string(),
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(entry)
}

fn parse_opset_id(cursor: &mut Cursor) -> Result<OperatorSetId, Error> {
    let mut opset = OperatorSetId::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => opset.domain = cursor.read_string()?.to_string(),
            2 => opset.version = cursor.read_varint()? as i64,
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(opset)
}

fn parse_tensor_shape_dim(cursor: &mut Cursor) -> Result<TensorShapeDimension, Error> {
    let mut dim = TensorShapeDimension::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => dim.value = Dimension::Value(cursor.read_varint()? as i64),
            2 => dim.value = Dimension::Param(cursor.read_string()?.to_string()),
            3 => dim.denotation = cursor.read_string()?.to_string(),
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(dim)
}

fn parse_tensor_shape(cursor: &mut Cursor) -> Result<TensorShape, Error> {
    let mut shape = TensorShape::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => {
                let mut sub = cursor.read_sub_cursor()?;
                shape.dim.push(parse_tensor_shape_dim(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(shape)
}

fn parse_tensor_type(cursor: &mut Cursor) -> Result<TensorTypeProto, Error> {
    let mut tt = TensorTypeProto::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => tt.elem_type = DataType::try_from(cursor.read_varint()? as i32)?,
            2 => {
                let mut sub = cursor.read_sub_cursor()?;
                tt.shape = Some(parse_tensor_shape(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(tt)
}

fn parse_sequence_type(cursor: &mut Cursor) -> Result<SequenceTypeProto, Error> {
    let mut elem_type = TypeProto::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => {
                let mut sub = cursor.read_sub_cursor()?;
                elem_type = parse_type_proto(&mut sub)?;
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(SequenceTypeProto {
        elem_type: Box::new(elem_type),
    })
}

fn parse_map_type(cursor: &mut Cursor) -> Result<MapTypeProto, Error> {
    let mut key_type = DataType::default();
    let mut value_type = TypeProto::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => key_type = DataType::try_from(cursor.read_varint()? as i32)?,
            2 => {
                let mut sub = cursor.read_sub_cursor()?;
                value_type = parse_type_proto(&mut sub)?;
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(MapTypeProto {
        key_type,
        value_type: Box::new(value_type),
    })
}

fn parse_optional_type(cursor: &mut Cursor) -> Result<OptionalTypeProto, Error> {
    let mut elem_type = TypeProto::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => {
                let mut sub = cursor.read_sub_cursor()?;
                elem_type = parse_type_proto(&mut sub)?;
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(OptionalTypeProto {
        elem_type: Box::new(elem_type),
    })
}

fn parse_sparse_tensor_type(cursor: &mut Cursor) -> Result<SparseTensorTypeProto, Error> {
    let mut st = SparseTensorTypeProto::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => st.elem_type = DataType::try_from(cursor.read_varint()? as i32)?,
            2 => {
                let mut sub = cursor.read_sub_cursor()?;
                st.shape = Some(parse_tensor_shape(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(st)
}

fn parse_type_proto(cursor: &mut Cursor) -> Result<TypeProto, Error> {
    let mut tp = TypeProto::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => {
                let mut sub = cursor.read_sub_cursor()?;
                tp.value = Some(TypeValue::Tensor(parse_tensor_type(&mut sub)?));
            }
            4 => {
                let mut sub = cursor.read_sub_cursor()?;
                tp.value = Some(TypeValue::Sequence(parse_sequence_type(&mut sub)?));
            }
            5 => {
                let mut sub = cursor.read_sub_cursor()?;
                tp.value = Some(TypeValue::Map(parse_map_type(&mut sub)?));
            }
            6 => tp.denotation = cursor.read_string()?.to_string(),
            8 => {
                let mut sub = cursor.read_sub_cursor()?;
                tp.value = Some(TypeValue::SparseTensor(parse_sparse_tensor_type(&mut sub)?));
            }
            9 => {
                let mut sub = cursor.read_sub_cursor()?;
                tp.value = Some(TypeValue::Optional(parse_optional_type(&mut sub)?));
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(tp)
}

fn parse_value_info(cursor: &mut Cursor) -> Result<ValueInfo, Error> {
    let mut vi = ValueInfo::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => vi.name = cursor.read_string()?.to_string(),
            2 => {
                let mut sub = cursor.read_sub_cursor()?;
                vi.r#type = Some(parse_type_proto(&mut sub)?);
            }
            3 => vi.doc_string = cursor.read_string()?.to_string(),
            4 => {
                let mut sub = cursor.read_sub_cursor()?;
                vi.metadata_props.push(parse_string_string_entry(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(vi)
}

fn parse_tensor_segment(cursor: &mut Cursor) -> Result<TensorSegment, Error> {
    let mut seg = TensorSegment::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => seg.begin = cursor.read_varint()? as i64,
            2 => seg.end = cursor.read_varint()? as i64,
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(seg)
}

fn parse_tensor(cursor: &mut Cursor) -> Result<TensorProto, Error> {
    let mut t = TensorProto::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => match wire_type {
                WireType::LengthDelimited => {
                    let mut sub = cursor.read_sub_cursor()?;
                    t.dims.reserve(sub.remaining());
                    while !sub.is_empty() {
                        t.dims.push(sub.read_varint()? as i64);
                    }
                }
                _ => t.dims.push(cursor.read_varint()? as i64),
            },
            2 => t.data_type = DataType::try_from(cursor.read_varint()? as i32)?,
            3 => {
                let mut sub = cursor.read_sub_cursor()?;
                t.segment = Some(parse_tensor_segment(&mut sub)?);
            }
            4 => match wire_type {
                WireType::LengthDelimited => {
                    let sub = cursor.read_sub_cursor()?;
                    let bytes = sub.remaining_slice();
                    let count = bytes.len() / 4;
                    t.float_data.reserve(count);
                    for chunk in bytes.chunks_exact(4) {
                        t.float_data.push(f32::from_le_bytes(
                            chunk.try_into().unwrap(),
                        ));
                    }
                }
                _ => {
                    t.float_data.push(cursor.read_f32_le()?);
                }
            },
            5 => match wire_type {
                WireType::LengthDelimited => {
                    let mut sub = cursor.read_sub_cursor()?;
                    t.int32_data.reserve(sub.remaining());
                    while !sub.is_empty() {
                        t.int32_data.push(sub.read_varint()? as i32);
                    }
                }
                _ => t.int32_data.push(cursor.read_varint()? as i32),
            },
            6 => t.string_data.push(cursor.read_bytes()?.to_vec()),
            7 => match wire_type {
                WireType::LengthDelimited => {
                    let mut sub = cursor.read_sub_cursor()?;
                    t.int64_data.reserve(sub.remaining());
                    while !sub.is_empty() {
                        t.int64_data.push(sub.read_varint()? as i64);
                    }
                }
                _ => t.int64_data.push(cursor.read_varint()? as i64),
            },
            8 => t.name = cursor.read_string()?.to_string(),
            9 => t.raw_data = cursor.read_bytes()?.to_vec(),
            10 => match wire_type {
                WireType::LengthDelimited => {
                    let sub = cursor.read_sub_cursor()?;
                    let bytes = sub.remaining_slice();
                    let count = bytes.len() / 8;
                    t.double_data.reserve(count);
                    for chunk in bytes.chunks_exact(8) {
                        t.double_data.push(f64::from_le_bytes(
                            chunk.try_into().unwrap(),
                        ));
                    }
                }
                _ => {
                    t.double_data.push(cursor.read_f64_le()?);
                }
            },
            11 => match wire_type {
                WireType::LengthDelimited => {
                    let mut sub = cursor.read_sub_cursor()?;
                    t.uint64_data.reserve(sub.remaining());
                    while !sub.is_empty() {
                        t.uint64_data.push(sub.read_varint()?);
                    }
                }
                _ => t.uint64_data.push(cursor.read_varint()?),
            },
            12 => t.doc_string = cursor.read_string()?.to_string(),
            13 => {
                let mut sub = cursor.read_sub_cursor()?;
                t.external_data.push(parse_string_string_entry(&mut sub)?);
            }
            14 => t.data_location = DataLocation::try_from(cursor.read_varint()? as i32)?,
            16 => {
                let mut sub = cursor.read_sub_cursor()?;
                t.metadata_props.push(parse_string_string_entry(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(t)
}

fn parse_sparse_tensor(cursor: &mut Cursor) -> Result<SparseTensor, Error> {
    let mut st = SparseTensor::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => {
                let mut sub = cursor.read_sub_cursor()?;
                st.values = Some(parse_tensor(&mut sub)?);
            }
            2 => {
                let mut sub = cursor.read_sub_cursor()?;
                st.indices = Some(parse_tensor(&mut sub)?);
            }
            3 => match wire_type {
                WireType::LengthDelimited => {
                    let mut sub = cursor.read_sub_cursor()?;
                    st.dims.reserve(sub.remaining());
                    while !sub.is_empty() {
                        st.dims.push(sub.read_varint()? as i64);
                    }
                }
                _ => st.dims.push(cursor.read_varint()? as i64),
            },
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(st)
}

fn parse_tensor_annotation(cursor: &mut Cursor) -> Result<TensorAnnotation, Error> {
    let mut ta = TensorAnnotation::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => ta.tensor_name = cursor.read_string()?.to_string(),
            2 => {
                let mut sub = cursor.read_sub_cursor()?;
                ta.quant_parameter_tensor_names
                    .push(parse_string_string_entry(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(ta)
}

fn parse_attribute(cursor: &mut Cursor) -> Result<Attribute, Error> {
    let mut attr = Attribute::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => attr.name = cursor.read_string()?.to_string(),
            2 => {
                attr.f = cursor.read_f32_le()?;
            }
            3 => attr.i = cursor.read_varint()? as i64,
            4 => attr.s = cursor.read_bytes()?.to_vec(),
            5 => {
                let mut sub = cursor.read_sub_cursor()?;
                attr.t = Some(parse_tensor(&mut sub)?);
            }
            6 => {
                let mut sub = cursor.read_sub_cursor()?;
                attr.g = Some(Box::new(parse_graph(&mut sub)?));
            }
            7 => match wire_type {
                WireType::LengthDelimited => {
                    let sub = cursor.read_sub_cursor()?;
                    let bytes = sub.remaining_slice();
                    let count = bytes.len() / 4;
                    attr.floats.reserve(count);
                    for chunk in bytes.chunks_exact(4) {
                        attr.floats.push(f32::from_le_bytes(
                            chunk.try_into().unwrap(),
                        ));
                    }
                }
                _ => {
                    attr.floats.push(cursor.read_f32_le()?);
                }
            },
            8 => match wire_type {
                WireType::LengthDelimited => {
                    let mut sub = cursor.read_sub_cursor()?;
                    attr.ints.reserve(sub.remaining());
                    while !sub.is_empty() {
                        attr.ints.push(sub.read_varint()? as i64);
                    }
                }
                _ => attr.ints.push(cursor.read_varint()? as i64),
            },
            9 => attr.strings.push(cursor.read_bytes()?.to_vec()),
            10 => {
                let mut sub = cursor.read_sub_cursor()?;
                attr.tensors.push(parse_tensor(&mut sub)?);
            }
            11 => {
                let mut sub = cursor.read_sub_cursor()?;
                attr.graphs.push(parse_graph(&mut sub)?);
            }
            13 => attr.doc_string = cursor.read_string()?.to_string(),
            14 => {
                let mut sub = cursor.read_sub_cursor()?;
                attr.tp = Some(parse_type_proto(&mut sub)?);
            }
            15 => {
                let mut sub = cursor.read_sub_cursor()?;
                attr.type_protos.push(parse_type_proto(&mut sub)?);
            }
            20 => attr.r#type = AttributeType::try_from(cursor.read_varint()? as i32)?,
            21 => attr.ref_attr_name = cursor.read_string()?.to_string(),
            22 => {
                let mut sub = cursor.read_sub_cursor()?;
                attr.sparse_tensor = Some(parse_sparse_tensor(&mut sub)?);
            }
            23 => {
                let mut sub = cursor.read_sub_cursor()?;
                attr.sparse_tensors.push(parse_sparse_tensor(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(attr)
}

fn parse_node(cursor: &mut Cursor) -> Result<Node, Error> {
    let mut node = Node::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => node.input.push(cursor.read_string()?.to_string()),
            2 => node.output.push(cursor.read_string()?.to_string()),
            3 => node.name = cursor.read_string()?.to_string(),
            4 => node.op_type = OpType::from(cursor.read_string()?),
            5 => {
                let mut sub = cursor.read_sub_cursor()?;
                node.attribute.push(parse_attribute(&mut sub)?);
            }
            6 => node.doc_string = cursor.read_string()?.to_string(),
            7 => node.domain = cursor.read_string()?.to_string(),
            8 => node.overload = cursor.read_string()?.to_string(),
            9 => {
                let mut sub = cursor.read_sub_cursor()?;
                node.metadata_props.push(parse_string_string_entry(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(node)
}

fn parse_graph(cursor: &mut Cursor) -> Result<Graph, Error> {
    let mut graph = Graph::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => {
                let mut sub = cursor.read_sub_cursor()?;
                graph.node.push(parse_node(&mut sub)?);
            }
            2 => graph.name = cursor.read_string()?.to_string(),
            5 => {
                let mut sub = cursor.read_sub_cursor()?;
                graph.initializer.push(parse_tensor(&mut sub)?);
            }
            10 => graph.doc_string = cursor.read_string()?.to_string(),
            11 => {
                let mut sub = cursor.read_sub_cursor()?;
                graph.input.push(parse_value_info(&mut sub)?);
            }
            12 => {
                let mut sub = cursor.read_sub_cursor()?;
                graph.output.push(parse_value_info(&mut sub)?);
            }
            13 => {
                let mut sub = cursor.read_sub_cursor()?;
                graph.value_info.push(parse_value_info(&mut sub)?);
            }
            14 => {
                let mut sub = cursor.read_sub_cursor()?;
                graph.quantization_annotation.push(parse_tensor_annotation(&mut sub)?);
            }
            15 => {
                let mut sub = cursor.read_sub_cursor()?;
                graph.sparse_initializer.push(parse_sparse_tensor(&mut sub)?);
            }
            16 => {
                let mut sub = cursor.read_sub_cursor()?;
                graph.metadata_props.push(parse_string_string_entry(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(graph)
}

fn parse_training_info(cursor: &mut Cursor) -> Result<TrainingInfo, Error> {
    let mut ti = TrainingInfo::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => {
                let mut sub = cursor.read_sub_cursor()?;
                ti.initialization = Some(parse_graph(&mut sub)?);
            }
            2 => {
                let mut sub = cursor.read_sub_cursor()?;
                ti.algorithm = Some(parse_graph(&mut sub)?);
            }
            3 => {
                let mut sub = cursor.read_sub_cursor()?;
                ti.initialization_binding.push(parse_string_string_entry(&mut sub)?);
            }
            4 => {
                let mut sub = cursor.read_sub_cursor()?;
                ti.update_binding.push(parse_string_string_entry(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(ti)
}

fn parse_function(cursor: &mut Cursor) -> Result<Function, Error> {
    let mut func = Function::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => func.name = cursor.read_string()?.to_string(),
            4 => func.input.push(cursor.read_string()?.to_string()),
            5 => func.output.push(cursor.read_string()?.to_string()),
            6 => func.attribute.push(cursor.read_string()?.to_string()),
            7 => {
                let mut sub = cursor.read_sub_cursor()?;
                func.node.push(parse_node(&mut sub)?);
            }
            8 => func.doc_string = cursor.read_string()?.to_string(),
            9 => {
                let mut sub = cursor.read_sub_cursor()?;
                func.opset_import.push(parse_opset_id(&mut sub)?);
            }
            10 => func.domain = cursor.read_string()?.to_string(),
            11 => {
                let mut sub = cursor.read_sub_cursor()?;
                func.attribute_proto.push(parse_attribute(&mut sub)?);
            }
            12 => {
                let mut sub = cursor.read_sub_cursor()?;
                func.value_info.push(parse_value_info(&mut sub)?);
            }
            13 => func.overload = cursor.read_string()?.to_string(),
            14 => {
                let mut sub = cursor.read_sub_cursor()?;
                func.metadata_props.push(parse_string_string_entry(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(func)
}

fn parse_model(cursor: &mut Cursor) -> Result<Model, Error> {
    let mut model = Model::default();
    while !cursor.is_empty() {
        let (field, wire_type) = cursor.read_tag()?;
        match field {
            1 => model.ir_version = cursor.read_varint()? as i64,
            2 => model.producer_name = cursor.read_string()?.to_string(),
            3 => model.producer_version = cursor.read_string()?.to_string(),
            4 => model.domain = cursor.read_string()?.to_string(),
            5 => model.model_version = cursor.read_varint()? as i64,
            6 => model.doc_string = cursor.read_string()?.to_string(),
            7 => {
                let mut sub = cursor.read_sub_cursor()?;
                model.graph = Some(parse_graph(&mut sub)?);
            }
            8 => {
                let mut sub = cursor.read_sub_cursor()?;
                model.opset_import.push(parse_opset_id(&mut sub)?);
            }
            14 => {
                let mut sub = cursor.read_sub_cursor()?;
                model.metadata_props.push(parse_string_string_entry(&mut sub)?);
            }
            20 => {
                let mut sub = cursor.read_sub_cursor()?;
                model.training_info.push(parse_training_info(&mut sub)?);
            }
            25 => {
                let mut sub = cursor.read_sub_cursor()?;
                model.functions.push(parse_function(&mut sub)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(model)
}
