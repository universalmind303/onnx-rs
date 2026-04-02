/// Top-level ONNX model container.
///
/// Corresponds to the `ModelProto` message in the ONNX protobuf schema.
/// Contains model metadata, operator set versions, and the computation
/// [`Graph`].
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::*;
///
/// let model = Model {
///     ir_version: 9,
///     producer_name: "my-framework",
///     producer_version: "1.0",
///     opset_import: vec![OperatorSetId { domain: "", version: 19 }],
///     graph: Some(Graph {
///         name: "main_graph",
///         node: vec![Node {
///             op_type: OpType::Add,
///             input: vec!["A", "B"],
///             output: vec!["C"],
///             ..Default::default()
///         }],
///         ..Default::default()
///     }),
///     ..Default::default()
/// };
///
/// assert_eq!(model.ir_version, 9);
/// assert_eq!(model.graph.as_ref().unwrap().name, "main_graph");
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Model<'a> {
    pub ir_version: i64,
    pub opset_import: Vec<OperatorSetId<'a>>,
    pub producer_name: &'a str,
    pub producer_version: &'a str,
    pub domain: &'a str,
    pub model_version: i64,
    pub doc_string: &'a str,
    pub graph: Option<Graph<'a>>,
    pub metadata_props: Vec<StringStringEntry<'a>>,
    pub training_info: Vec<TrainingInfo<'a>>,
    pub functions: Vec<Function<'a>>,
}

/// An operator set version imported by a model.
///
/// Corresponds to `OperatorSetIdProto` in the ONNX protobuf schema.
/// An empty `domain` refers to the default ONNX operator set.
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::OperatorSetId;
///
/// let opset = OperatorSetId { domain: "", version: 19 };
/// assert_eq!(opset.version, 19);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OperatorSetId<'a> {
    pub domain: &'a str,
    pub version: i64,
}

/// A key-value string pair used for metadata properties.
///
/// Corresponds to `StringStringEntryProto` in the ONNX protobuf schema.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StringStringEntry<'a> {
    pub key: &'a str,
    pub value: &'a str,
}

/// A computation graph containing nodes, inputs, outputs, and initializers.
///
/// Corresponds to `GraphProto` in the ONNX protobuf schema.
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::*;
///
/// let graph = Graph {
///     name: "my_graph",
///     node: vec![
///         Node {
///             op_type: OpType::Relu,
///             input: vec!["X"],
///             output: vec!["Y"],
///             ..Default::default()
///         },
///     ],
///     input: vec![ValueInfo { name: "X", ..Default::default() }],
///     output: vec![ValueInfo { name: "Y", ..Default::default() }],
///     ..Default::default()
/// };
///
/// assert_eq!(graph.node.len(), 1);
/// assert_eq!(graph.node[0].op_type, OpType::Relu);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Graph<'a> {
    pub node: Vec<Node<'a>>,
    pub name: &'a str,
    pub initializer: Vec<TensorProto<'a>>,
    pub sparse_initializer: Vec<SparseTensor<'a>>,
    pub doc_string: &'a str,
    pub input: Vec<ValueInfo<'a>>,
    pub output: Vec<ValueInfo<'a>>,
    pub value_info: Vec<ValueInfo<'a>>,
    pub quantization_annotation: Vec<TensorAnnotation<'a>>,
    pub metadata_props: Vec<StringStringEntry<'a>>,
}

/// A single operation in the computation graph.
///
/// Corresponds to `NodeProto` in the ONNX protobuf schema. Each node has an
/// [`OpType`], consumes named inputs, produces named outputs, and may carry
/// [`Attribute`]s that parameterize the operation.
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::*;
///
/// let node = Node {
///     name: "conv_0",
///     op_type: OpType::Conv,
///     input: vec!["X", "W"],
///     output: vec!["Y"],
///     attribute: vec![Attribute {
///         name: "kernel_shape",
///         r#type: AttributeType::Ints,
///         ints: vec![3, 3],
///         ..Default::default()
///     }],
///     ..Default::default()
/// };
///
/// assert_eq!(node.op_type, OpType::Conv);
/// assert_eq!(node.attribute[0].ints, vec![3, 3]);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Node<'a> {
    pub input: Vec<&'a str>,
    pub output: Vec<&'a str>,
    pub name: &'a str,
    pub op_type: OpType<'a>,
    pub domain: &'a str,
    pub overload: &'a str,
    pub attribute: Vec<Attribute<'a>>,
    pub doc_string: &'a str,
    pub metadata_props: Vec<StringStringEntry<'a>>,
}

/// ONNX operator type.
///
/// Contains variants for all standard ONNX operators. Unrecognized operator
/// names are stored as [`OpType::Custom`].
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::OpType;
///
/// // Convert from a string
/// let op = OpType::from("MatMul");
/// assert_eq!(op, OpType::MatMul);
///
/// // Convert back to a string
/// assert_eq!(op.as_str(), "MatMul");
///
/// // Unknown operators become Custom
/// let custom = OpType::from("MyCustomOp");
/// assert_eq!(custom, OpType::Custom("MyCustomOp"));
/// assert_eq!(custom.as_str(), "MyCustomOp");
///
/// // Display works as expected
/// assert_eq!(format!("{}", OpType::Relu), "Relu");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType<'a> {
    // Activation
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,
    Elu,
    Selu,
    PRelu,
    Gelu,
    HardSigmoid,
    HardSwish,
    Softplus,
    Softsign,
    ThresholdedRelu,
    Celu,
    Mish,

    // Math (element-wise)
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Pow,
    Ceil,
    Floor,
    Round,
    Clip,
    Sign,
    Reciprocal,
    Mod,
    BitShift,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNot,
    Min,
    Max,
    Mean,
    Sum,

    // Trigonometric
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,

    // Convolution and pooling
    Conv,
    ConvTranspose,
    ConvInteger,
    AveragePool,
    MaxPool,
    GlobalAveragePool,
    GlobalMaxPool,
    GlobalLpPool,
    LpPool,
    MaxRoiPool,
    MaxUnpool,

    // Normalization
    BatchNormalization,
    InstanceNormalization,
    LayerNormalization,
    GroupNormalization,
    LpNormalization,
    MeanVarianceNormalization,

    // Linear algebra
    MatMul,
    MatMulInteger,
    Gemm,

    // Tensor manipulation
    Reshape,
    Transpose,
    Flatten,
    Squeeze,
    Unsqueeze,
    Expand,
    Tile,
    Pad,
    Slice,
    Split,
    SplitToSequence,
    Concat,
    ConcatFromSequence,
    DepthToSpace,
    SpaceToDepth,
    ReverseSequence,

    // Gather / Scatter
    Gather,
    GatherElements,
    GatherND,
    Scatter,
    ScatterElements,
    ScatterND,

    // Reduction
    ReduceMax,
    ReduceMin,
    ReduceMean,
    ReduceSum,
    ReduceProd,
    ReduceL1,
    ReduceL2,
    ReduceLogSum,
    ReduceLogSumExp,
    ReduceSumSquare,
    ArgMax,
    ArgMin,

    // Comparison and logic
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    Not,
    And,
    Or,
    Xor,
    Where,
    IsNaN,
    IsInf,

    // Shape and type
    Shape,
    Size,
    Cast,
    CastLike,
    ConstantOfShape,
    Range,
    OneHot,
    NonZero,
    TopK,
    Unique,
    EyeLike,
    Compress,

    // RNN
    LSTM,
    GRU,
    RNN,

    // Constants
    Constant,
    Identity,

    // Control flow
    If,
    Loop,
    Scan,

    // Resize
    Resize,
    Upsample,

    // Regularization
    Dropout,

    // Quantization
    QuantizeLinear,
    DequantizeLinear,
    DynamicQuantizeLinear,
    QLinearConv,
    QLinearMatMul,

    // Attention / Transformer
    Attention,
    Einsum,

    // Sequence
    SequenceConstruct,
    SequenceAt,
    SequenceEmpty,
    SequenceInsert,
    SequenceErase,
    SequenceLength,
    SequenceMap,

    // Optional
    Optional,
    OptionalGetElement,
    OptionalHasElement,

    // Image
    ImageDecoder,
    GridSample,
    RoiAlign,
    AffineGrid,

    // Misc
    CumSum,
    Det,
    Erf,
    Hardmax,
    Shrink,
    StringNormalizer,
    TfIdfVectorizer,
    NegativeLogLikelihoodLoss,
    SoftmaxCrossEntropyLoss,
    Bernoulli,
    RandomNormal,
    RandomNormalLike,
    RandomUniform,
    RandomUniformLike,
    Multinomial,
    CenterCropPad,
    Col2Im,
    Trilu,
    BlackmanWindow,
    HannWindow,
    HammingWindow,
    DFT,
    MelWeightMatrix,
    STFT,
    RegexFullMatch,
    StringConcat,
    StringSplit,

    Custom(&'a str),
}

impl OpType<'_> {
    /// Returns the ONNX string name of this operator.
    ///
    /// For standard operators this returns the canonical name (e.g. `"Relu"`).
    /// For [`OpType::Custom`] it returns the stored string.
    ///
    /// ```
    /// use onnx_rs::ast::OpType;
    ///
    /// assert_eq!(OpType::Softmax.as_str(), "Softmax");
    /// assert_eq!(OpType::Custom("FusedGelu").as_str(), "FusedGelu");
    /// ```
    pub fn as_str(&self) -> &str {
        match self {
            OpType::Custom(s) => s,
            other => op_type_to_str(other),
        }
    }
}

impl Default for OpType<'_> {
    fn default() -> Self {
        OpType::Custom("")
    }
}

impl std::fmt::Display for OpType<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpType::Custom(s) => write!(f, "{s}"),
            other => write!(f, "{}", op_type_to_str(other)),
        }
    }
}

impl<'a> From<&'a str> for OpType<'a> {
    fn from(s: &'a str) -> Self {
        match s {
            "Relu" => OpType::Relu,
            "LeakyRelu" => OpType::LeakyRelu,
            "Sigmoid" => OpType::Sigmoid,
            "Tanh" => OpType::Tanh,
            "Softmax" => OpType::Softmax,
            "LogSoftmax" => OpType::LogSoftmax,
            "Elu" => OpType::Elu,
            "Selu" => OpType::Selu,
            "PRelu" => OpType::PRelu,
            "Gelu" => OpType::Gelu,
            "HardSigmoid" => OpType::HardSigmoid,
            "HardSwish" => OpType::HardSwish,
            "Softplus" => OpType::Softplus,
            "Softsign" => OpType::Softsign,
            "ThresholdedRelu" => OpType::ThresholdedRelu,
            "Celu" => OpType::Celu,
            "Mish" => OpType::Mish,
            "Add" => OpType::Add,
            "Sub" => OpType::Sub,
            "Mul" => OpType::Mul,
            "Div" => OpType::Div,
            "Neg" => OpType::Neg,
            "Abs" => OpType::Abs,
            "Sqrt" => OpType::Sqrt,
            "Exp" => OpType::Exp,
            "Log" => OpType::Log,
            "Pow" => OpType::Pow,
            "Ceil" => OpType::Ceil,
            "Floor" => OpType::Floor,
            "Round" => OpType::Round,
            "Clip" => OpType::Clip,
            "Sign" => OpType::Sign,
            "Reciprocal" => OpType::Reciprocal,
            "Mod" => OpType::Mod,
            "BitShift" => OpType::BitShift,
            "BitwiseAnd" => OpType::BitwiseAnd,
            "BitwiseOr" => OpType::BitwiseOr,
            "BitwiseXor" => OpType::BitwiseXor,
            "BitwiseNot" => OpType::BitwiseNot,
            "Min" => OpType::Min,
            "Max" => OpType::Max,
            "Mean" => OpType::Mean,
            "Sum" => OpType::Sum,
            "Sin" => OpType::Sin,
            "Cos" => OpType::Cos,
            "Tan" => OpType::Tan,
            "Asin" => OpType::Asin,
            "Acos" => OpType::Acos,
            "Atan" => OpType::Atan,
            "Sinh" => OpType::Sinh,
            "Cosh" => OpType::Cosh,
            "Asinh" => OpType::Asinh,
            "Acosh" => OpType::Acosh,
            "Atanh" => OpType::Atanh,
            "Conv" => OpType::Conv,
            "ConvTranspose" => OpType::ConvTranspose,
            "ConvInteger" => OpType::ConvInteger,
            "AveragePool" => OpType::AveragePool,
            "MaxPool" => OpType::MaxPool,
            "GlobalAveragePool" => OpType::GlobalAveragePool,
            "GlobalMaxPool" => OpType::GlobalMaxPool,
            "GlobalLpPool" => OpType::GlobalLpPool,
            "LpPool" => OpType::LpPool,
            "MaxRoiPool" => OpType::MaxRoiPool,
            "MaxUnpool" => OpType::MaxUnpool,
            "BatchNormalization" => OpType::BatchNormalization,
            "InstanceNormalization" => OpType::InstanceNormalization,
            "LayerNormalization" => OpType::LayerNormalization,
            "GroupNormalization" => OpType::GroupNormalization,
            "LpNormalization" => OpType::LpNormalization,
            "MeanVarianceNormalization" => OpType::MeanVarianceNormalization,
            "MatMul" => OpType::MatMul,
            "MatMulInteger" => OpType::MatMulInteger,
            "Gemm" => OpType::Gemm,
            "Reshape" => OpType::Reshape,
            "Transpose" => OpType::Transpose,
            "Flatten" => OpType::Flatten,
            "Squeeze" => OpType::Squeeze,
            "Unsqueeze" => OpType::Unsqueeze,
            "Expand" => OpType::Expand,
            "Tile" => OpType::Tile,
            "Pad" => OpType::Pad,
            "Slice" => OpType::Slice,
            "Split" => OpType::Split,
            "SplitToSequence" => OpType::SplitToSequence,
            "Concat" => OpType::Concat,
            "ConcatFromSequence" => OpType::ConcatFromSequence,
            "DepthToSpace" => OpType::DepthToSpace,
            "SpaceToDepth" => OpType::SpaceToDepth,
            "ReverseSequence" => OpType::ReverseSequence,
            "Gather" => OpType::Gather,
            "GatherElements" => OpType::GatherElements,
            "GatherND" => OpType::GatherND,
            "Scatter" => OpType::Scatter,
            "ScatterElements" => OpType::ScatterElements,
            "ScatterND" => OpType::ScatterND,
            "ReduceMax" => OpType::ReduceMax,
            "ReduceMin" => OpType::ReduceMin,
            "ReduceMean" => OpType::ReduceMean,
            "ReduceSum" => OpType::ReduceSum,
            "ReduceProd" => OpType::ReduceProd,
            "ReduceL1" => OpType::ReduceL1,
            "ReduceL2" => OpType::ReduceL2,
            "ReduceLogSum" => OpType::ReduceLogSum,
            "ReduceLogSumExp" => OpType::ReduceLogSumExp,
            "ReduceSumSquare" => OpType::ReduceSumSquare,
            "ArgMax" => OpType::ArgMax,
            "ArgMin" => OpType::ArgMin,
            "Equal" => OpType::Equal,
            "Greater" => OpType::Greater,
            "GreaterOrEqual" => OpType::GreaterOrEqual,
            "Less" => OpType::Less,
            "LessOrEqual" => OpType::LessOrEqual,
            "Not" => OpType::Not,
            "And" => OpType::And,
            "Or" => OpType::Or,
            "Xor" => OpType::Xor,
            "Where" => OpType::Where,
            "IsNaN" => OpType::IsNaN,
            "IsInf" => OpType::IsInf,
            "Shape" => OpType::Shape,
            "Size" => OpType::Size,
            "Cast" => OpType::Cast,
            "CastLike" => OpType::CastLike,
            "ConstantOfShape" => OpType::ConstantOfShape,
            "Range" => OpType::Range,
            "OneHot" => OpType::OneHot,
            "NonZero" => OpType::NonZero,
            "TopK" => OpType::TopK,
            "Unique" => OpType::Unique,
            "EyeLike" => OpType::EyeLike,
            "Compress" => OpType::Compress,
            "LSTM" => OpType::LSTM,
            "GRU" => OpType::GRU,
            "RNN" => OpType::RNN,
            "Constant" => OpType::Constant,
            "Identity" => OpType::Identity,
            "If" => OpType::If,
            "Loop" => OpType::Loop,
            "Scan" => OpType::Scan,
            "Resize" => OpType::Resize,
            "Upsample" => OpType::Upsample,
            "Dropout" => OpType::Dropout,
            "QuantizeLinear" => OpType::QuantizeLinear,
            "DequantizeLinear" => OpType::DequantizeLinear,
            "DynamicQuantizeLinear" => OpType::DynamicQuantizeLinear,
            "QLinearConv" => OpType::QLinearConv,
            "QLinearMatMul" => OpType::QLinearMatMul,
            "Attention" => OpType::Attention,
            "Einsum" => OpType::Einsum,
            "SequenceConstruct" => OpType::SequenceConstruct,
            "SequenceAt" => OpType::SequenceAt,
            "SequenceEmpty" => OpType::SequenceEmpty,
            "SequenceInsert" => OpType::SequenceInsert,
            "SequenceErase" => OpType::SequenceErase,
            "SequenceLength" => OpType::SequenceLength,
            "SequenceMap" => OpType::SequenceMap,
            "Optional" => OpType::Optional,
            "OptionalGetElement" => OpType::OptionalGetElement,
            "OptionalHasElement" => OpType::OptionalHasElement,
            "ImageDecoder" => OpType::ImageDecoder,
            "GridSample" => OpType::GridSample,
            "RoiAlign" => OpType::RoiAlign,
            "AffineGrid" => OpType::AffineGrid,
            "Cumsum" | "CumSum" => OpType::CumSum,
            "Det" => OpType::Det,
            "Erf" => OpType::Erf,
            "Hardmax" | "HardMax" => OpType::Hardmax,
            "Shrink" => OpType::Shrink,
            "StringNormalizer" => OpType::StringNormalizer,
            "TfIdfVectorizer" => OpType::TfIdfVectorizer,
            "NegativeLogLikelihoodLoss" => OpType::NegativeLogLikelihoodLoss,
            "SoftmaxCrossEntropyLoss" => OpType::SoftmaxCrossEntropyLoss,
            "Bernoulli" => OpType::Bernoulli,
            "RandomNormal" => OpType::RandomNormal,
            "RandomNormalLike" => OpType::RandomNormalLike,
            "RandomUniform" => OpType::RandomUniform,
            "RandomUniformLike" => OpType::RandomUniformLike,
            "Multinomial" => OpType::Multinomial,
            "CenterCropPad" => OpType::CenterCropPad,
            "Col2Im" => OpType::Col2Im,
            "Trilu" => OpType::Trilu,
            "BlackmanWindow" => OpType::BlackmanWindow,
            "HannWindow" => OpType::HannWindow,
            "HammingWindow" => OpType::HammingWindow,
            "DFT" => OpType::DFT,
            "MelWeightMatrix" => OpType::MelWeightMatrix,
            "STFT" => OpType::STFT,
            "RegexFullMatch" => OpType::RegexFullMatch,
            "StringConcat" => OpType::StringConcat,
            "StringSplit" => OpType::StringSplit,
            other => OpType::Custom(other),
        }
    }
}

fn op_type_to_str(op: &OpType<'_>) -> &'static str {
    match op {
        OpType::Relu => "Relu",
        OpType::LeakyRelu => "LeakyRelu",
        OpType::Sigmoid => "Sigmoid",
        OpType::Tanh => "Tanh",
        OpType::Softmax => "Softmax",
        OpType::LogSoftmax => "LogSoftmax",
        OpType::Elu => "Elu",
        OpType::Selu => "Selu",
        OpType::PRelu => "PRelu",
        OpType::Gelu => "Gelu",
        OpType::HardSigmoid => "HardSigmoid",
        OpType::HardSwish => "HardSwish",
        OpType::Softplus => "Softplus",
        OpType::Softsign => "Softsign",
        OpType::ThresholdedRelu => "ThresholdedRelu",
        OpType::Celu => "Celu",
        OpType::Mish => "Mish",
        OpType::Add => "Add",
        OpType::Sub => "Sub",
        OpType::Mul => "Mul",
        OpType::Div => "Div",
        OpType::Neg => "Neg",
        OpType::Abs => "Abs",
        OpType::Sqrt => "Sqrt",
        OpType::Exp => "Exp",
        OpType::Log => "Log",
        OpType::Pow => "Pow",
        OpType::Ceil => "Ceil",
        OpType::Floor => "Floor",
        OpType::Round => "Round",
        OpType::Clip => "Clip",
        OpType::Sign => "Sign",
        OpType::Reciprocal => "Reciprocal",
        OpType::Mod => "Mod",
        OpType::BitShift => "BitShift",
        OpType::BitwiseAnd => "BitwiseAnd",
        OpType::BitwiseOr => "BitwiseOr",
        OpType::BitwiseXor => "BitwiseXor",
        OpType::BitwiseNot => "BitwiseNot",
        OpType::Min => "Min",
        OpType::Max => "Max",
        OpType::Mean => "Mean",
        OpType::Sum => "Sum",
        OpType::Sin => "Sin",
        OpType::Cos => "Cos",
        OpType::Tan => "Tan",
        OpType::Asin => "Asin",
        OpType::Acos => "Acos",
        OpType::Atan => "Atan",
        OpType::Sinh => "Sinh",
        OpType::Cosh => "Cosh",
        OpType::Asinh => "Asinh",
        OpType::Acosh => "Acosh",
        OpType::Atanh => "Atanh",
        OpType::Conv => "Conv",
        OpType::ConvTranspose => "ConvTranspose",
        OpType::ConvInteger => "ConvInteger",
        OpType::AveragePool => "AveragePool",
        OpType::MaxPool => "MaxPool",
        OpType::GlobalAveragePool => "GlobalAveragePool",
        OpType::GlobalMaxPool => "GlobalMaxPool",
        OpType::GlobalLpPool => "GlobalLpPool",
        OpType::LpPool => "LpPool",
        OpType::MaxRoiPool => "MaxRoiPool",
        OpType::MaxUnpool => "MaxUnpool",
        OpType::BatchNormalization => "BatchNormalization",
        OpType::InstanceNormalization => "InstanceNormalization",
        OpType::LayerNormalization => "LayerNormalization",
        OpType::GroupNormalization => "GroupNormalization",
        OpType::LpNormalization => "LpNormalization",
        OpType::MeanVarianceNormalization => "MeanVarianceNormalization",
        OpType::MatMul => "MatMul",
        OpType::MatMulInteger => "MatMulInteger",
        OpType::Gemm => "Gemm",
        OpType::Reshape => "Reshape",
        OpType::Transpose => "Transpose",
        OpType::Flatten => "Flatten",
        OpType::Squeeze => "Squeeze",
        OpType::Unsqueeze => "Unsqueeze",
        OpType::Expand => "Expand",
        OpType::Tile => "Tile",
        OpType::Pad => "Pad",
        OpType::Slice => "Slice",
        OpType::Split => "Split",
        OpType::SplitToSequence => "SplitToSequence",
        OpType::Concat => "Concat",
        OpType::ConcatFromSequence => "ConcatFromSequence",
        OpType::DepthToSpace => "DepthToSpace",
        OpType::SpaceToDepth => "SpaceToDepth",
        OpType::ReverseSequence => "ReverseSequence",
        OpType::Gather => "Gather",
        OpType::GatherElements => "GatherElements",
        OpType::GatherND => "GatherND",
        OpType::Scatter => "Scatter",
        OpType::ScatterElements => "ScatterElements",
        OpType::ScatterND => "ScatterND",
        OpType::ReduceMax => "ReduceMax",
        OpType::ReduceMin => "ReduceMin",
        OpType::ReduceMean => "ReduceMean",
        OpType::ReduceSum => "ReduceSum",
        OpType::ReduceProd => "ReduceProd",
        OpType::ReduceL1 => "ReduceL1",
        OpType::ReduceL2 => "ReduceL2",
        OpType::ReduceLogSum => "ReduceLogSum",
        OpType::ReduceLogSumExp => "ReduceLogSumExp",
        OpType::ReduceSumSquare => "ReduceSumSquare",
        OpType::ArgMax => "ArgMax",
        OpType::ArgMin => "ArgMin",
        OpType::Equal => "Equal",
        OpType::Greater => "Greater",
        OpType::GreaterOrEqual => "GreaterOrEqual",
        OpType::Less => "Less",
        OpType::LessOrEqual => "LessOrEqual",
        OpType::Not => "Not",
        OpType::And => "And",
        OpType::Or => "Or",
        OpType::Xor => "Xor",
        OpType::Where => "Where",
        OpType::IsNaN => "IsNaN",
        OpType::IsInf => "IsInf",
        OpType::Shape => "Shape",
        OpType::Size => "Size",
        OpType::Cast => "Cast",
        OpType::CastLike => "CastLike",
        OpType::ConstantOfShape => "ConstantOfShape",
        OpType::Range => "Range",
        OpType::OneHot => "OneHot",
        OpType::NonZero => "NonZero",
        OpType::TopK => "TopK",
        OpType::Unique => "Unique",
        OpType::EyeLike => "EyeLike",
        OpType::Compress => "Compress",
        OpType::LSTM => "LSTM",
        OpType::GRU => "GRU",
        OpType::RNN => "RNN",
        OpType::Constant => "Constant",
        OpType::Identity => "Identity",
        OpType::If => "If",
        OpType::Loop => "Loop",
        OpType::Scan => "Scan",
        OpType::Resize => "Resize",
        OpType::Upsample => "Upsample",
        OpType::Dropout => "Dropout",
        OpType::QuantizeLinear => "QuantizeLinear",
        OpType::DequantizeLinear => "DequantizeLinear",
        OpType::DynamicQuantizeLinear => "DynamicQuantizeLinear",
        OpType::QLinearConv => "QLinearConv",
        OpType::QLinearMatMul => "QLinearMatMul",
        OpType::Attention => "Attention",
        OpType::Einsum => "Einsum",
        OpType::SequenceConstruct => "SequenceConstruct",
        OpType::SequenceAt => "SequenceAt",
        OpType::SequenceEmpty => "SequenceEmpty",
        OpType::SequenceInsert => "SequenceInsert",
        OpType::SequenceErase => "SequenceErase",
        OpType::SequenceLength => "SequenceLength",
        OpType::SequenceMap => "SequenceMap",
        OpType::Optional => "Optional",
        OpType::OptionalGetElement => "OptionalGetElement",
        OpType::OptionalHasElement => "OptionalHasElement",
        OpType::ImageDecoder => "ImageDecoder",
        OpType::GridSample => "GridSample",
        OpType::RoiAlign => "RoiAlign",
        OpType::AffineGrid => "AffineGrid",
        OpType::CumSum => "CumSum",
        OpType::Det => "Det",
        OpType::Erf => "Erf",
        OpType::Hardmax => "Hardmax",
        OpType::Shrink => "Shrink",
        OpType::StringNormalizer => "StringNormalizer",
        OpType::TfIdfVectorizer => "TfIdfVectorizer",
        OpType::NegativeLogLikelihoodLoss => "NegativeLogLikelihoodLoss",
        OpType::SoftmaxCrossEntropyLoss => "SoftmaxCrossEntropyLoss",
        OpType::Bernoulli => "Bernoulli",
        OpType::RandomNormal => "RandomNormal",
        OpType::RandomNormalLike => "RandomNormalLike",
        OpType::RandomUniform => "RandomUniform",
        OpType::RandomUniformLike => "RandomUniformLike",
        OpType::Multinomial => "Multinomial",
        OpType::CenterCropPad => "CenterCropPad",
        OpType::Col2Im => "Col2Im",
        OpType::Trilu => "Trilu",
        OpType::BlackmanWindow => "BlackmanWindow",
        OpType::HannWindow => "HannWindow",
        OpType::HammingWindow => "HammingWindow",
        OpType::DFT => "DFT",
        OpType::MelWeightMatrix => "MelWeightMatrix",
        OpType::STFT => "STFT",
        OpType::RegexFullMatch => "RegexFullMatch",
        OpType::StringConcat => "StringConcat",
        OpType::StringSplit => "StringSplit",
        OpType::Custom(_) => unreachable!(),
    }
}

/// Element data type for tensors.
///
/// Corresponds to `TensorProto.DataType` in the ONNX protobuf schema.
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::DataType;
///
/// let dt = DataType::Float;
/// assert_ne!(dt, DataType::Int32);
///
/// // Convert from protobuf integer values
/// let dt = DataType::try_from(1).unwrap();
/// assert_eq!(dt, DataType::Float);
///
/// // Invalid values produce an error
/// assert!(DataType::try_from(999).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataType {
    #[default]
    Undefined,
    Float,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Int32,
    Int64,
    String,
    Bool,
    Float16,
    Double,
    Uint32,
    Uint64,
    Complex64,
    Complex128,
    Bfloat16,
    Float8e4m3fn,
    Float8e4m3fnuz,
    Float8e5m2,
    Float8e5m2fnuz,
    Uint4,
    Int4,
    Float4e2m1,
}

impl TryFrom<i32> for DataType {
    type Error = crate::error::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(DataType::Undefined),
            1 => Ok(DataType::Float),
            2 => Ok(DataType::Uint8),
            3 => Ok(DataType::Int8),
            4 => Ok(DataType::Uint16),
            5 => Ok(DataType::Int16),
            6 => Ok(DataType::Int32),
            7 => Ok(DataType::Int64),
            8 => Ok(DataType::String),
            9 => Ok(DataType::Bool),
            10 => Ok(DataType::Float16),
            11 => Ok(DataType::Double),
            12 => Ok(DataType::Uint32),
            13 => Ok(DataType::Uint64),
            14 => Ok(DataType::Complex64),
            15 => Ok(DataType::Complex128),
            16 => Ok(DataType::Bfloat16),
            17 => Ok(DataType::Float8e4m3fn),
            18 => Ok(DataType::Float8e4m3fnuz),
            19 => Ok(DataType::Float8e5m2),
            20 => Ok(DataType::Float8e5m2fnuz),
            21 => Ok(DataType::Uint4),
            22 => Ok(DataType::Int4),
            23 => Ok(DataType::Float4e2m1),
            _ => Err(crate::error::Error::InvalidEnumValue {
                enum_name: "DataType",
                value,
            }),
        }
    }
}

/// Discriminant for the kind of value stored in an [`Attribute`].
///
/// Corresponds to `AttributeProto.AttributeType` in the ONNX protobuf schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttributeType {
    #[default]
    Undefined,
    Float,
    Int,
    String,
    Tensor,
    Graph,
    Floats,
    Ints,
    Strings,
    Tensors,
    Graphs,
    SparseTensor,
    SparseTensors,
    TypeProto,
    TypeProtos,
}

impl TryFrom<i32> for AttributeType {
    type Error = crate::error::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(AttributeType::Undefined),
            1 => Ok(AttributeType::Float),
            2 => Ok(AttributeType::Int),
            3 => Ok(AttributeType::String),
            4 => Ok(AttributeType::Tensor),
            5 => Ok(AttributeType::Graph),
            6 => Ok(AttributeType::Floats),
            7 => Ok(AttributeType::Ints),
            8 => Ok(AttributeType::Strings),
            9 => Ok(AttributeType::Tensors),
            10 => Ok(AttributeType::Graphs),
            11 => Ok(AttributeType::SparseTensor),
            12 => Ok(AttributeType::SparseTensors),
            13 => Ok(AttributeType::TypeProto),
            14 => Ok(AttributeType::TypeProtos),
            _ => Err(crate::error::Error::InvalidEnumValue {
                enum_name: "AttributeType",
                value,
            }),
        }
    }
}

/// Where tensor data is stored.
///
/// `Default` means inline in the protobuf; `External` means in a separate file
/// referenced by [`TensorProto::external_data`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataLocation {
    #[default]
    Default,
    External,
}

impl TryFrom<i32> for DataLocation {
    type Error = crate::error::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(DataLocation::Default),
            1 => Ok(DataLocation::External),
            _ => Err(crate::error::Error::InvalidEnumValue {
                enum_name: "DataLocation",
                value,
            }),
        }
    }
}

/// A single dimension in a tensor shape — either a fixed value or a symbolic parameter.
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::Dimension;
///
/// let fixed = Dimension::Value(224);
/// let dynamic = Dimension::Param("batch_size");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Dimension<'a> {
    Value(i64),
    Param(&'a str),
}

impl Default for Dimension<'_> {
    fn default() -> Self {
        Dimension::Value(0)
    }
}

/// A single dimension in a [`TensorShape`], combining a [`Dimension`] value with
/// an optional denotation string.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorShapeDimension<'a> {
    pub value: Dimension<'a>,
    pub denotation: &'a str,
}

/// The shape of a tensor, as a list of dimensions.
///
/// Corresponds to `TensorShapeProto` in the ONNX protobuf schema.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorShape<'a> {
    pub dim: Vec<TensorShapeDimension<'a>>,
}

/// Describes a tensor type with an element data type and optional shape.
///
/// Corresponds to `TypeProto.Tensor` in the ONNX protobuf schema.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorTypeProto<'a> {
    pub elem_type: DataType,
    pub shape: Option<TensorShape<'a>>,
}

/// Describes a sequence type whose elements have a given [`TypeProto`].
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceTypeProto<'a> {
    pub elem_type: Box<TypeProto<'a>>,
}

/// Describes a map type with a key [`DataType`] and a value [`TypeProto`].
#[derive(Debug, Clone, PartialEq)]
pub struct MapTypeProto<'a> {
    pub key_type: DataType,
    pub value_type: Box<TypeProto<'a>>,
}

/// Describes an optional type whose element has a given [`TypeProto`].
#[derive(Debug, Clone, PartialEq)]
pub struct OptionalTypeProto<'a> {
    pub elem_type: Box<TypeProto<'a>>,
}

/// Describes a sparse tensor type with an element data type and optional shape.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SparseTensorTypeProto<'a> {
    pub elem_type: DataType,
    pub shape: Option<TensorShape<'a>>,
}

/// The concrete type stored inside a [`TypeProto`].
#[derive(Debug, Clone, PartialEq)]
pub enum TypeValue<'a> {
    Tensor(TensorTypeProto<'a>),
    Sequence(SequenceTypeProto<'a>),
    Map(MapTypeProto<'a>),
    Optional(OptionalTypeProto<'a>),
    SparseTensor(SparseTensorTypeProto<'a>),
}

/// Describes the type of a graph input, output, or intermediate value.
///
/// Corresponds to `TypeProto` in the ONNX protobuf schema. The [`value`](TypeProto::value)
/// field holds the specific type variant (tensor, sequence, map, etc.).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TypeProto<'a> {
    pub value: Option<TypeValue<'a>>,
    pub denotation: &'a str,
}

/// Type and shape information for a graph input, output, or intermediate value.
///
/// Corresponds to `ValueInfoProto` in the ONNX protobuf schema.
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::*;
///
/// let vi = ValueInfo {
///     name: "input_0",
///     r#type: Some(TypeProto {
///         value: Some(TypeValue::Tensor(TensorTypeProto {
///             elem_type: DataType::Float,
///             shape: Some(TensorShape {
///                 dim: vec![
///                     TensorShapeDimension {
///                         value: Dimension::Param("batch"),
///                         denotation: "",
///                     },
///                     TensorShapeDimension {
///                         value: Dimension::Value(784),
///                         denotation: "",
///                     },
///                 ],
///             }),
///         })),
///         denotation: "",
///     }),
///     ..Default::default()
/// };
///
/// assert_eq!(vi.name, "input_0");
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ValueInfo<'a> {
    pub name: &'a str,
    pub r#type: Option<TypeProto<'a>>,
    pub doc_string: &'a str,
    pub metadata_props: Vec<StringStringEntry<'a>>,
}

/// Byte range within a segmented tensor. Used for very large tensors that
/// are split across multiple `TensorProto` messages.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorSegment {
    pub begin: i64,
    pub end: i64,
}

/// An ONNX tensor value.
///
/// Corresponds to `TensorProto` in the ONNX protobuf schema. Tensor data
/// can be stored in typed fields ([`float_data`](TensorProto::float_data),
/// [`int32_data`](TensorProto::int32_data), etc.) or as
/// [`raw_data`](TensorProto::raw_data) bytes.
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::*;
///
/// let tensor = TensorProto {
///     name: "weights",
///     dims: vec![2, 3],
///     data_type: DataType::Float,
///     float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
///     ..Default::default()
/// };
///
/// assert_eq!(tensor.dims, vec![2, 3]);
/// assert_eq!(tensor.float_data.len(), 6);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorProto<'a> {
    pub dims: Vec<i64>,
    pub data_type: DataType,
    pub segment: Option<TensorSegment>,
    pub float_data: Vec<f32>,
    pub int32_data: Vec<i32>,
    pub string_data: Vec<&'a [u8]>,
    pub int64_data: Vec<i64>,
    pub name: &'a str,
    pub raw_data: &'a [u8],
    pub double_data: Vec<f64>,
    pub uint64_data: Vec<u64>,
    pub doc_string: &'a str,
    pub external_data: Vec<StringStringEntry<'a>>,
    pub data_location: DataLocation,
    pub metadata_props: Vec<StringStringEntry<'a>>,
}

/// A sparse tensor stored as separate values and indices tensors.
///
/// Corresponds to `SparseTensorProto` in the ONNX protobuf schema.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SparseTensor<'a> {
    pub values: Option<TensorProto<'a>>,
    pub indices: Option<TensorProto<'a>>,
    pub dims: Vec<i64>,
}

/// Quantization annotation for a tensor in the graph.
///
/// Corresponds to `TensorAnnotation` in the ONNX protobuf schema.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorAnnotation<'a> {
    pub tensor_name: &'a str,
    pub quant_parameter_tensor_names: Vec<StringStringEntry<'a>>,
}

/// A named attribute on a graph [`Node`].
///
/// Corresponds to `AttributeProto` in the ONNX protobuf schema. The value is
/// stored in whichever field matches the [`type`](Attribute::type) discriminant
/// (e.g., `f` for floats, `i` for ints, `s` for byte strings).
///
/// # Examples
///
/// ```
/// use onnx_rs::ast::*;
///
/// let attr = Attribute {
///     name: "axis",
///     r#type: AttributeType::Int,
///     i: 1,
///     ..Default::default()
/// };
///
/// assert_eq!(attr.name, "axis");
/// assert_eq!(attr.i, 1);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Attribute<'a> {
    pub name: &'a str,
    pub ref_attr_name: &'a str,
    pub doc_string: &'a str,
    pub r#type: AttributeType,
    pub f: f32,
    pub i: i64,
    pub s: &'a [u8],
    pub t: Option<TensorProto<'a>>,
    pub g: Option<Box<Graph<'a>>>,
    pub sparse_tensor: Option<SparseTensor<'a>>,
    pub tp: Option<TypeProto<'a>>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<&'a [u8]>,
    pub tensors: Vec<TensorProto<'a>>,
    pub graphs: Vec<Graph<'a>>,
    pub sparse_tensors: Vec<SparseTensor<'a>>,
    pub type_protos: Vec<TypeProto<'a>>,
}

/// Training-specific information attached to a model.
///
/// Corresponds to `TrainingInfoProto` in the ONNX protobuf schema.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrainingInfo<'a> {
    pub initialization: Option<Graph<'a>>,
    pub algorithm: Option<Graph<'a>>,
    pub initialization_binding: Vec<StringStringEntry<'a>>,
    pub update_binding: Vec<StringStringEntry<'a>>,
}

/// A reusable function definition within an ONNX model.
///
/// Corresponds to `FunctionProto` in the ONNX protobuf schema.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Function<'a> {
    pub name: &'a str,
    pub input: Vec<&'a str>,
    pub output: Vec<&'a str>,
    pub attribute: Vec<&'a str>,
    pub attribute_proto: Vec<Attribute<'a>>,
    pub node: Vec<Node<'a>>,
    pub doc_string: &'a str,
    pub opset_import: Vec<OperatorSetId<'a>>,
    pub domain: &'a str,
    pub overload: &'a str,
    pub value_info: Vec<ValueInfo<'a>>,
    pub metadata_props: Vec<StringStringEntry<'a>>,
}
