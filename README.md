# Transformer-based Machine Translation (IWSLT2014 En-De)
## 简介 (Project Overview)
本项目是一个基于 PyTorch 从零实现的 **Transformer** 模型，用了 **IWSLT 2014** 数据集进行训练，旨在完成 **英语到德语 (English-to-German)** 的机器翻译任务，
整个过程涵盖了数据预处理、Tokenizer 训练、模型构建、训练循环以及 BLEU 分数评估（未使用）的全流程。

**注**：由于本项目主要用于学习和演示 Transformer 架构的实现细节。目前的翻译效果尚未达到标准，具体原因见下方“实验分析”部分。

## 🛠️ 环境配置 (Requirements)
在运行代码前，请确保安装了以下依赖库：
```bash
pip install -r requirements.txt

