# Transformer-based Machine Translation (IWSLT2014 En-De)
## 简介 (Project Overview)
本项目是一个基于 PyTorch 从零实现的 **Transformer** 模型，用了 **IWSLT 2014** 数据集进行训练，旨在完成 **英语到德语 (English-to-German)** 的机器翻译任务，
整个过程涵盖了数据预处理、Tokenizer 训练、模型构建、训练循环以及 BLEU 分数评估（未使用）的全流程。

**注**：由于本项目主要用于学习和演示 Transformer 架构的实现细节。目前的翻译效果尚未达到标准，具体原因见下方“实验分析”部分。

## 环境配置 (Requirements)
在运行代码前，请确保安装了以下依赖库：
```bash
pip install -r requirements.txt
```



## 当前的实验配置如下（可以在 main.py 和 model.py 中调整）：
| 参数 | 设定值 | 说明 |
| :--- | :--- | :--- |
| 模型架构 | Transformer (Base) | 包含 Encoder 和 Decoder |
| 网络层数 (Layers) | 6 | Encoder/Decoder 堆叠层数 |
| 隐藏层维度 (d_model) | 512 | 词向量维度 |
| 注意力头数 (Heads) | 8 | Multi-head Attention 的头数 |
| 前馈网络维度 (d_ff) | 2048 | Feed Forward 中间层维度 |
| 最大序列长度 | 256 | 最大Token数 |
| 学习率 (Learning Rate) |Noam Scheduler| 采用原论文 Warmup + Decay 策略 |
| Optimizer | Adam | 优化器 |
| Batch Size | 128 | 批次大小 |

## 实验结果 (Results)
1. 训练过程
训练过程已保存为 training_log.txt (可直接点击查看)。

2. 翻译示例 (Translation Examples)

以下是模型在当前状态下的部分推理结果：

| 源语言 (English) | 目标语言 (German - Predicted) |
| :--- | :--- |
| Hello, how are you? | Hallo , wie geht es Ihnen ? |
| I love machine learning. | Ich liebe Spielen . |
| The weather is nice today. | Die Größe ist hier . |

> **注**：可以看到模型能够学习到基本的语序和部分词汇对应关系，但在语义准确性和长句处理上仍有较大提升空间。

## 实验分析((Analysis)
目前模型的翻译质量未达到预期,经过分析(根据对 `train_debug.py` 和 `main_debug.py` 的调试分析，发现模型在少量数据或单句测试中能够准确翻译，这证明模型的核心代码逻辑（Attention 机制、前向传播等）是正确的，说明模型本身没有问题），可能原因如下：
1. 数据量不足 (Data Scarcity)
-- IWSLT 2014 数据集大约有 170k 句对，对于参数量巨大的 Transformer 模型来说，数据量可能不够，导致模型过拟合，无法泛化到未见过的句子。
2. 模型容量过大 (Model Capacity)
--  在训练中使用了标准的 Base 配置（6层，512维度）。可以尝试减小 d_model (如降至 128 或 256) 或减少层数 (num_layers=3)。
3. 超参数设置不当 (Hyperparameters)
-- 虽然代码实现了原论文推荐的 Noam Scheduler (Warmup + Decay) 策略，但该策略的默认超参数（特别是 `warmup_steps`）可能不适合**IWSLT 2014**数据集，原有的 Warmup 时长可能过长，或者峰值学习率不适合当前数据分布，从而造成收敛困难或陷入局部最优。
4. 训练时长 (Training Duration)
--受限于计算资源，训练的 Epoch 数量可能不足，模型尚未充分学习。

## 未来改进计划 (Future Work)
1. 尝试更小的模型架构以适应该数据集。
2. 更改学习率策略
3. 增加数据增强或混合更大规模的数据集。
4. 添加 Beam Search 解码策略以提升生成质量。

## 文件结构 (File Structcture)

```text
.
├── iws lt14/              # 数据缓存目录
├── __pycache__/           # Python 字节码
├── model.py               # Transformer 模型定义 (核心代码)
├── train.py               # 训练逻辑、评估函数、数据加载
├── train_debug.py         # 调试用的训练脚本
├── main.py                # 主程序入口
├── main_debug.py          # 调试入口
├── src_tokenizer.json     # 源语言 (英语) 分词器配置
├── tgt_tokenizer.json     # 目标语言 (德语) 分词器配置
├── training_log.txt       # 训练日志
└── requirements.txt       # 依赖库列表
