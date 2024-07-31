## 介绍
我们推出了一个用于工业领域的预训练语言模型——Luban。Luban 是一个强大的语言模型，专为解决工业领域的复杂问题而设计。目前，我们已经发布了包含 1B 参数的模型，并且正在训练包含 10B 参数的更大模型，以进一步提升其性能和应用范围。

Luban 项目不仅提供了模型生成测试，还包含了Fine-Tuning指令微调流程的完整代码，便于用户根据自己的需求进行定制化微调。

### 模型参数和特点
- **Luban-1B**：拥有 1B 参数，已经发布并可供使用。适用于一般工业领域的自然语言处理任务。
- **Luban-10B**：拥有 10B 参数，当前正在训练中。预计将显著提高模型在更复杂和细致任务上的表现能力。

### 项目特点
1. **预训练模型**：Luban 在大规模工业语料库上进行预训练，能够理解和生成与工业领域相关的高质量文本。
2. **Fine-Tuning微调**：支持使用Fine-Tuning微调流程来调整模型，使其更好地适应特定的工业应用场景。
3. **灵活生成**：提供简单易用的生成测试代码，方便用户快速验证模型效果。

### 项目结构
- **生成测试**：快速测试模型的生成能力，支持自定义问题。
- **Fine-Tuning微调**：详细的指令微调流程，包括数据处理、样本构建和全面微调的实现。

我们相信，Luban 将成为工业领域的有力工具，帮助用户在各种自然语言处理任务中获得更高的效率和精度。无论是用于技术文档的生成、产品描述的编写，还是复杂技术问题的问答，Luban 都能提供卓越的支持。

欢迎大家下载和使用Luban，并期待在未来发布更多改进和增强版本。

## 快速开始

### 环境搭建
```bash
pip install -r requirements.txt
```
### 预训练模型下载

| **模型名称** | **模型参数** | **下载地址** |
|--------------|---------------|--------------|
| **Luban-1B** | - **max_seq_len**: 2048<br>- **dim**: 2048<br>- **n_layers**: 18<br>- **n_heads**: 8 | [百度云盘下载](https://pan.baidu.com/s/1FkhcfS6CPreLZSgcZToBUg) 提取码：onk2<br>[Hugging Face下载](https://huggingface.co/ziyanglichuan/Luban-1B) |

### 测试模型生成
```bash
# Luban模型下载
cd model
wget https://huggingface.co/ziyanglichuan/Luban-1B/blob/main/Luban-1B.pth
# 测试Luban模型，运行eval.py
python eval.py
```

### 示例
```bash
# 示例一：Input：‘用挤出机挤塑生产塑料制品是’
Luban-1B response：‘国内外公认的最适合塑料加工的新途径。目前，挤塑工艺发展迅速，已经可以生产出厚度在0.03毫米至0.3毫米的无机高分子合金，而且能生产出耐腐蚀、重量轻、低密度、难燃、导电性能好等多种塑料制品。同时，塑料制品的成型技术、塑料制品加工工艺技术日趋完善和成熟，产品不断更新换代。’

# 示例二：Input：‘介绍一下数控加工：’
Luban-1B response：‘数控加工，又称数控加工技术，是由计算机控制，运用自动编程语言（即ERP软件）使数控机床实现自动控制的一种方法。其主要特点表现在：一是采用高速旋转的数控车床、车削中心或加工中心实现自动对刀，保证零件加工尺寸精度、表面加工质量、重复定位精度、重复定位行程精度和切削速度等方面。二是数控机床具有高能密度和多轴控制，可以自动实现多轴加工。三是数控机床的加工对象是以二维图形为主要特征的，因而适应性很强，其产品具有三维特征。’
```

## Fine-Tuning指令微调
LLM微调是将预训练模型中的知识引导出来的一种手段。

### Fine-Tuning微调数据
本项目可以在Fine-Tuning语料上进行模型微调，可参考语料如下：

**日常问答数据**：
| Fine-Tuning语料                                                            | 描述                                                                 |
|-----------------------------------------------------------------------------|--------------------------------------------------------------------|
| alpaca-zh：[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh) | 源自shibing624的一部分Fine-Tuning数据。该数据集是参考Alpaca方法基于GPT4得到的self-instruct数据，约5万条。 |
| bell：[bell](https://huggingface.co/datasets/BelleGroup/train_1M_CN)         | 源自BelleGroup的一部分Fine-Tuning数据。包含约100万条由BELLE项目生成的中文指令数据。|

### Fine-Tuning样本构建

在构建Dataloader时进行分词并构建batch送给模型。请参考`finetune_dataset.py`文件。以下是详细步骤和逻辑：

1. **语料处理**：需要准备好用于微调的语料。语料应包含prompt和answer的成对数据。

2. **分词处理**：在构建Dataloader时进行分词处理。使用分词器将每个prompt和answer转换为相应的token序列。

3. **数据格式**：确保prompt和answer之间有一个开始符`<bos>`隔开，并在answer后添加一个结束符`<eos>`。

4. **Masking**：在计算loss时，需要对prompt部分的loss进行mask操作，仅计算answer部分的loss。这样可以确保模型专注于生成正确的回答，而不是仅仅记住prompt部分的内容。

以下是具体脚本运行的示例：

```bash
# 下载微调预料
cd finetune_data
wget https://huggingface.co/datasets/shibing624/alpaca-zh/blob/main/alpaca_gpt4_data_zh.json
# 脚本中针对alpaca-zh语料进行处理，添加新的语料可以自行扩展。
python generate_finetune_data.py
# 运行结束后，会在finetune_data目录下产生finetune_data.csv文件
```

### 全面微调（Full Fine-tuning）

在完成数据处理和样本构建后，可以进行全面微调。以下是微调的基本步骤：

1. **加载预训练模型**：加载预训练好的Luban模型。

2. **设置微调参数**：设置微调所需的参数，包括学习率、batch size、训练轮数等。

3. **训练过程**：使用处理好的数据进行训练。

4. **保存微调模型**：训练结束后，将微调后的模型保存到指定目录。

具体的代码实现如下：

```bash
# 执行微调代码
python finetuning.py
# 运行结束后，模型会保存在‘finetune_out’文件夹中
```
通过上述步骤，可以有效地对Luban模型进行微调，使其更加适应特定的工业应用场景，从而提高模型的生成效果和准确性。

## 致谢
Baby-Llama2-Chinese项目为我们的项目提供了重要支持。基于该项目的框架，我们得以针对工业领域需求进行适配和优化。在此，我们对Baby-Llama2-Chinese项目表示衷心的感谢。

## 参考链接
[Baby-Llama2-Chinese](https://github.com/DLLXW/baby-llama2-chinese)