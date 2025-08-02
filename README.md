# Transformer 模型实现

这是一个从零开始实现的Transformer模型，用于序列到序列的翻译任务，特别是中文到英文的翻译。

## 项目概述

本项目完整实现了《Attention Is All You Need》论文中提出的Transformer架构，包括编码器-解码器结构、多头注意力机制、位置编码等核心组件。

## 项目结构

```
AttentionOfLLMs/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── train_transformer.py        # 主训练脚本
├── Transformer.py              # Transformer主模型
├── Encoder.py                  # 编码器实现
├── EncoderLayer.py             # 编码器层实现
├── Decoder.py                  # 解码器实现
├── DecoderLayer.py             # 解码器层实现
├── MultiHeadedAttention.py     # 多头注意力机制
├── SelfAttention.py            # 自注意力机制
├── FeedForward.py              # 前馈神经网络
├── PositionalEncodeing.py      # 位置编码
└── __pycache__/                # Python缓存文件
```

## 核心功能

### 1. Transformer模型架构

- **编码器**：多层编码器堆叠，每层包含多头自注意力和前馈网络
- **解码器**：多层解码器堆叠，包含掩码自注意力、编码器-解码器注意力和前馈网络
- **位置编码**：使用正弦和余弦函数为序列添加位置信息
- **多头注意力**：并行计算多个注意力头，增强模型表达能力

### 2. 训练示例

项目包含一个简单的中英文翻译示例：

- 输入："我爱水课"
- 输出："I love easy course"

## 安装和使用

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- NumPy

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行训练

```bash
python train_transformer.py
```

## 模型参数

### 默认配置

- `d_model`: 128 (模型维度)
- `num_heads`: 4 (注意力头数)
- `num_encoders`: 2 (编码器层数)
- `num_decoders`: 2 (解码器层数)
- `max_len`: 50 (最大序列长度)

### 自定义配置

```python
model = Transformer(
    d_model=512,  # 模型维度
    num_heads=8,  # 注意力头数
    num_encoders=6,  # 编码器层数
    num_decoders=6,  # 解码器层数
    src_vocab_size=10000,  # 源语言词汇表大小
    tgt_vocab_size=10000,  # 目标语言词汇表大小
    max_len=5000  # 最大序列长度
)
```

## 模型架构详解

### 1. 多头注意力机制 (MultiHeadedAttention)

- 将输入分割为多个头并行处理
- 每个头学习不同的表示子空间
- 最后将多个头的输出拼接

### 2. 位置编码 (PositionalEncoding)

- 使用正弦和余弦函数生成位置编码
- 为模型提供序列中的位置信息
- 可学习的位置嵌入

### 3. 前馈网络 (FeedForward)

- 两层全连接网络
- 使用ReLU激活函数
- 包含Dropout正则化

### 4. 编码器层 (EncoderLayer)

- 多头自注意力机制
- 残差连接和层归一化
- 前馈网络

### 5. 解码器层 (DecoderLayer)

- 掩码多头自注意力
- 编码器-解码器注意力
- 前馈网络
- 残差连接和层归一化

## 训练特性

### 1. 损失函数

- 使用交叉熵损失函数
- 忽略填充标记(pad token)的损失

### 2. 优化器

- Adam优化器
- 学习率：0.001

### 3. 掩码机制

- **填充掩码**：忽略填充位置的注意力
- **因果掩码**：防止解码器看到未来信息

## 推理和生成

模型支持自回归生成：

1. 使用编码器处理源序列
2. 解码器逐步生成目标序列
3. 每次预测下一个词
4. 遇到结束符停止生成

## 扩展功能

### 1. 词汇表扩展

可以通过修改词汇表支持更大的数据集：

```python
src_vocab = {'<pad>': 0, '<unk>': 1, ...}  # 源语言词汇
tgt_vocab = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3, ...}  # 目标语言词汇
```

### 2. 数据集适配

支持自定义数据集，只需按照指定格式准备数据。

## 性能优化

- 使用较小的模型尺寸以便快速训练和测试
- 支持GPU加速训练
- 包含详细的调试信息

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目仅用于学习和研究目的。

## 参考文献

- Vaswani, A., et al. (2017). Attention is all you need. In Advances in neural information processing systems.
