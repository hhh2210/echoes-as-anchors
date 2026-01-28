# 数据处理模块

## 概述

本模块负责处理实验所需的各种数据格式转换，包括DPO训练数据准备和MLP探针训练的完整流水线。

## 文件结构

```
data_processing/
├── convert_to_dpo.py          # AM样本转DPO格式转换器
├── dataset_downloader.py      # 数据集下载工具
└── mlp_pipeline/              # MLP探针训练流水线
    ├── preprocess.py          # 数据预处理
    ├── labeling.py            # 数据标注
    ├── trainer.py             # MLP训练器
    ├── inference.py           # 推理时修剪
    └── utils.py               # 工具函数
```

## 核心功能

### 2. 数据集下载器 (`dataset_downloader.py`)

**功能**：自动下载和准备实验所需的数据集。

**支持的数据集**：
- GSM8K
- MMLU
- 其他评估数据集

### 3. MLP Pipeline (`mlp_pipeline/`)

完整的MLP探针训练和使用流水线，与`train_mlp/`目录功能互补。

#### 3.1 数据预处理 (`preprocess.py`)

**功能**：准备MLP训练数据
- 提取(问题, 思考内容)对
- 数据清理和格式化
- 批量处理优化

#### 3.2 数据标注 (`labeling.py`)

**功能**：为MLP训练生成标签
- 支持多种标注策略
- LLM辅助标注
- 语义相似度标注

#### 3.3 MLP训练器 (`trainer.py`)

**功能**：训练重复检测MLP
- 支持不同架构配置
- 早停和最佳模型保存
- 训练监控和日志

#### 3.4 推理修剪 (`inference.py`)

**功能**：推理时实时检测和修剪重复内容

**核心组件**：
- `RepeatDetector`: MLP模型定义
- `RepetitionPruningLogitsProcessor`: LogitsProcessor实现

**修剪策略**：
- `terminate`: 检测到重复时终止
- `truncate_and_continue`: 移除重复后继续

**使用示例**：
```python
from mlp_pipeline.inference import RepeatDetector, RepetitionPruningLogitsProcessor

# 加载训练好的MLP
mlp = RepeatDetector(input_dim=1536, hidden_dim=32)
mlp.load_state_dict(torch.load("model.pt"))

# 创建修剪处理器
processor = RepetitionPruningLogitsProcessor(
    question=user_question,
    tokenizer=tokenizer,
    embedder=embedding_model,
    mlp_probe=mlp,
    device=device
)

# 生成时使用
outputs = model.generate(
    input_ids,
    logits_processor=[processor]
)
```

#### 3.5 工具函数 (`utils.py`)

**提供的功能**：
- 嵌入计算辅助函数
- 数据加载和批处理
- 评估指标计算

## 数据处理流程

### DPO训练数据准备流程：
```
原始AM数据 → 提取<think>内容 → 识别前导语 → 创建偏好对 → DPO格式
```

### MLP训练数据流程：
```
对话数据 → 提取(q,t)对 → 标注重复标签 → 训练MLP → 推理修剪
```

## 配置参数

### DPO转换参数：
- `--min-preamble-words`: 最小前导词数（默认5）
- `--max-samples`: 最大处理样本数
- `--output-format`: 输出格式（jsonl/json）

### MLP训练参数：
- `--hidden-dim`: 隐藏层维度（默认32）
- `--batch-size`: 批量大小（默认64）
- `--learning-rate`: 学习率（默认0.001）

## 统计信息

转换过程会生成详细统计：
- 总处理样本数
- 成功转换数
- 跳过原因分布
- 平均前导长度
- 转换成功率

## 注意事项

1. **数据格式**：确保输入数据为JSONL格式
2. **内存管理**：大数据集建议使用流式处理
3. **标注质量**：MLP性能依赖标注质量
4. **路径配置**：使用绝对路径避免错误

## 依赖要求

- transformers
- sentence-transformers
- torch
- numpy
- tqdm