#[derive(Debug, Clone, PartialEq, Default)]
pub struct Model {
    pub ir_version: i64,
    pub opset_import: Vec<OperatorSetId>,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub graph: Option<Graph>,
    pub metadata_props: Vec<StringStringEntry>,
    pub training_info: Vec<TrainingInfo>,
    pub functions: Vec<Function>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OperatorSetId {
    pub domain: String,
    pub version: i64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StringStringEntry {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Graph {
    pub node: Vec<Node>,
    pub name: String,
    pub initializer: Vec<TensorProto>,
    pub sparse_initializer: Vec<SparseTensor>,
    pub doc_string: String,
    pub input: Vec<ValueInfo>,
    pub output: Vec<ValueInfo>,
    pub value_info: Vec<ValueInfo>,
    pub quantization_annotation: Vec<TensorAnnotation>,
    pub metadata_props: Vec<StringStringEntry>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Node {
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub name: String,
    pub op_type: OpType,
    pub domain: String,
    pub overload: String,
    pub attribute: Vec<Attribute>,
    pub doc_string: String,
    pub metadata_props: Vec<StringStringEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
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

    Custom(String),
}

impl Default for OpType {
    fn default() -> Self {
        OpType::Custom(String::new())
    }
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpType::Custom(s) => write!(f, "{s}"),
            other => write!(f, "{}", op_type_to_str(other)),
        }
    }
}

impl From<&str> for OpType {
    fn from(s: &str) -> Self {
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
            other => OpType::Custom(other.to_string()),
        }
    }
}

fn op_type_to_str(op: &OpType) -> &'static str {
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

#[derive(Debug, Clone, PartialEq)]
pub enum Dimension {
    Value(i64),
    Param(std::string::String),
}

impl Default for Dimension {
    fn default() -> Self {
        Dimension::Value(0)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorShapeDimension {
    pub value: Dimension,
    pub denotation: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorShape {
    pub dim: Vec<TensorShapeDimension>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorTypeProto {
    pub elem_type: DataType,
    pub shape: Option<TensorShape>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SequenceTypeProto {
    pub elem_type: Box<TypeProto>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapTypeProto {
    pub key_type: DataType,
    pub value_type: Box<TypeProto>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptionalTypeProto {
    pub elem_type: Box<TypeProto>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SparseTensorTypeProto {
    pub elem_type: DataType,
    pub shape: Option<TensorShape>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeValue {
    Tensor(TensorTypeProto),
    Sequence(SequenceTypeProto),
    Map(MapTypeProto),
    Optional(OptionalTypeProto),
    SparseTensor(SparseTensorTypeProto),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TypeProto {
    pub value: Option<TypeValue>,
    pub denotation: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ValueInfo {
    pub name: String,
    pub r#type: Option<TypeProto>,
    pub doc_string: String,
    pub metadata_props: Vec<StringStringEntry>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorSegment {
    pub begin: i64,
    pub end: i64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorProto {
    pub dims: Vec<i64>,
    pub data_type: DataType,
    pub segment: Option<TensorSegment>,
    pub float_data: Vec<f32>,
    pub int32_data: Vec<i32>,
    pub string_data: Vec<Vec<u8>>,
    pub int64_data: Vec<i64>,
    pub name: String,
    pub raw_data: Vec<u8>,
    pub double_data: Vec<f64>,
    pub uint64_data: Vec<u64>,
    pub doc_string: String,
    pub external_data: Vec<StringStringEntry>,
    pub data_location: DataLocation,
    pub metadata_props: Vec<StringStringEntry>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SparseTensor {
    pub values: Option<TensorProto>,
    pub indices: Option<TensorProto>,
    pub dims: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorAnnotation {
    pub tensor_name: String,
    pub quant_parameter_tensor_names: Vec<StringStringEntry>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Attribute {
    pub name: String,
    pub ref_attr_name: String,
    pub doc_string: String,
    pub r#type: AttributeType,
    pub f: f32,
    pub i: i64,
    pub s: Vec<u8>,
    pub t: Option<TensorProto>,
    pub g: Option<Box<Graph>>,
    pub sparse_tensor: Option<SparseTensor>,
    pub tp: Option<TypeProto>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<Vec<u8>>,
    pub tensors: Vec<TensorProto>,
    pub graphs: Vec<Graph>,
    pub sparse_tensors: Vec<SparseTensor>,
    pub type_protos: Vec<TypeProto>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrainingInfo {
    pub initialization: Option<Graph>,
    pub algorithm: Option<Graph>,
    pub initialization_binding: Vec<StringStringEntry>,
    pub update_binding: Vec<StringStringEntry>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Function {
    pub name: String,
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub attribute: Vec<String>,
    pub attribute_proto: Vec<Attribute>,
    pub node: Vec<Node>,
    pub doc_string: String,
    pub opset_import: Vec<OperatorSetId>,
    pub domain: String,
    pub overload: String,
    pub value_info: Vec<ValueInfo>,
    pub metadata_props: Vec<StringStringEntry>,
}
