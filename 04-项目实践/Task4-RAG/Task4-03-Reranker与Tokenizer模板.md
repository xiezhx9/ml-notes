---
tags:
  - LLM
  - RAG
  - Reranker
  - Tokenizer
  - XLM-RoBERTa
aliases:
  - Task4 Reranker
  - Tokenizer 模板
---

# Task 4.3：Reranker 与 Tokenizer 模板

> [!summary]
> Tokenizer 不仅负责把文本切成 Token ID，还会按照模型的训练约定插入特殊 Token。Reranker 要把 Query 和每个候选 Document 作为文本对联合编码，因此必须使用与模型微调时相同的 pair 模板。

返回总览：[[Task4-RAG学习索引]]

## 1. 模板的三个层次

实际使用中常把不同概念都称为“模板”，需要区分三层：

| 层次          | 作用                           | 谁负责                     |
| ----------- | ---------------------------- | ----------------------- |
| 特殊 Token 模板 | 插入 BOS/CLS/SEP/EOS/PAD       | `tokenizer(...)`        |
| 任务模板        | 组织指令、问题、上下文和答案               | 数据处理代码                  |
| 对话模板        | 把 `role/content` 消息渲染为模型约定格式 | `apply_chat_template()` |

Tokenizer 的“分词”和“套特殊 Token 模板”是同一次调用中的两个步骤：

```text
原始文本
→ SentencePiece/BPE 切分
→ Token IDs
→ 插入模型约定的特殊 Token
→ input_ids / attention_mask
```

## 2. 当前 Reranker 的真实文本对模板

`bge-reranker-base` 基于 XLM-RoBERTa，不使用字面上的 `[CLS]` 和 `[SEP]`，而是：

```text
<s> Query </s></s> Document </s>
```

本地模型的特殊 Token：

| 作用 | Token | ID |
|---|---|---:|
| BOS/CLS | `<s>` | `0` |
| EOS/SEP | `</s>` | `2` |
| Padding | `<pad>` | `1` |
| Unknown | `<unk>` | - |
| Mask | `<mask>` | - |

例如：

```text
Query:    什么是注意力机制？
Document: 注意力机制让模型关注重要信息。
```

实际输入为：

```text
<s> 什么是 注意力 机制 ? </s></s>
注意力 机制 让 模型 关注 重要 信息 。 </s>
```

`<s>` 位置最终会汇总 Query 和 Document 的联合信息，分类头再将其变成相关性分数：

$$
h_{cls}=H[:,0,:],
\qquad
s(q,d)=Wh_{cls}+b.
$$

XLM-RoBERTa 的 pair 模板在两段文本之间使用 `</s></s>`。这是模型训练时一直使用的边界约定，推理时应保持一致。

## 3. 如何让 Tokenizer 正确使用 Pair 模板

单条文本对：

```python
encoded = tokenizer(query, document)
```

批量文本对：

```python
encoded = tokenizer(
    [query] * len(documents),
    documents,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
```

`add_special_tokens=True` 默认开启，Tokenizer 内部会调用类似 `build_inputs_with_special_tokens(...)` 的方法完成拼接。

> [!warning]
> 不要写成 `tokenizer(query + document)`。这会被视为一段普通文本，中间不会出现 pair 分隔符，也不再符合 Reranker 的训练输入分布。

有多个候选 Document 时，应该构造多个独立文本对：

```text
<s> Query </s></s> Document 1 </s>
<s> Query </s></s> Document 2 </s>
<s> Query </s></s> Document 3 </s>
```

不是把三份文档都塞进同一条序列。

### 3.1 `max_length` 与 Batch 维度

`max_length` 限制的是每个样本分词后的总 Token 数，并且包含特殊 Token。

对于 XLM-RoBERTa 的文本对：

$$
T_i=T_{q_i}+T_{d_i}+4\leq512.
$$

对于单文本：

$$
T_i=T_{text_i}+2\leq512.
$$

批量传入文本对时，两个列表按位置一一配对，不会做笛卡尔积：

```python
encoded = tokenizer(
    ["q1", "q2", "q3"],
    ["d1", "d2", "d3"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
```

最终形状为：

$$
\text{input\_ids.shape}=[B,T].
$$

- $B$ 是文本数或文本对数。
- $T$ 是当前 Batch 经过截断和 Padding 后的公共长度。
- 传入单个文本并设置 `return_tensors="pt"` 时，形状仍为 `[1,T]`。

`padding=True` 只补到当前 Batch 最长样本；`padding="max_length"` 才固定补到 `max_length`。

### 3.2 文本对的截断策略

`truncation=True` 对文本对默认相当于 `longest_first`，优先从当前更长的一段删除 Token：

```python
tokenizer(
    queries,
    documents,
    truncation="longest_first",
    max_length=512,
)
```

可选策略：

| 策略 | 含义 |
|---|---|
| `longest_first` | 动态从较长的一段删除 |
| `only_first` | 只截断第一段 Query |
| `only_second` | 只截断第二段 Document |

Reranker 一般希望完整保留 Query，可以使用：

```python
tokenizer(
    queries,
    documents,
    padding=True,
    truncation="only_second",
    max_length=512,
    return_tensors="pt",
)
```

当 Query 远短于 Document 时，`longest_first` 实际上也主要截断 Document。

### 3.3 单文本的截断逻辑

单文本没有“截哪一段”的选择。以 XLM-RoBERTa 为例，单文本模板是：

```text
<s> Text </s>
```

`max_length=512` 时内容最多保留 510 个 Token。如果原文有 600 个 Token，默认会删除 90 个。

Tokenizer 的普通截断不会理解语义，也不会自动寻找句号、标题或重要段落。它只在分词后按 Token 边界切片，因此仍可能切断自然语言中的词、句子和段落。

默认通常从右侧删除，即保留开头：

```python
tokenizer.truncation_side = "right"
```

需要保留末尾时可改为：

```python
tokenizer.truncation_side = "left"
```

常见选择：

- 文本分类和标题信息重要的文档常保留开头。
- GPT 多轮对话需要保留最新消息时常保留末尾。
- RAG 文档更应该先做合理 Chunking，不应主要依赖暴力截断。

### 3.4 保留超长部分：Overflow 与 Stride

默认截断会直接丢弃超出部分。需要将长文本拆成多个重叠 Token 窗口时，可以使用：

```python
encoded = tokenizer(
    long_text,
    truncation=True,
    max_length=512,
    stride=64,
    return_overflowing_tokens=True,
    return_tensors="pt",
)
```

概念上会得到：

```text
窗口 1：token[0:510]
窗口 2：token[446:956]
窗口 3：token[892:...]
```

`stride=64` 表示相邻窗口重叠 64 个 Token。这时一个原始文本可能产生多个窗口，所以 Tensor 第一维不再一定等于原始文本数量：

```text
原始文本数：1
滑动窗口数：3
input_ids.shape: [3,512]
```

## 4. 不同模型家族的特殊 Token 模板

| 模型家族 | 单文本 | 文本对 |
|---|---|---|
| BERT | `[CLS] A [SEP]` | `[CLS] A [SEP] B [SEP]` |
| RoBERTa/XLM-R | `<s> A </s>` | `<s> A </s></s> B </s>` |
| GPT 类 | 通常是 `A <eos>` | 通常没有原生 pair 模板 |
| T5 | 通常是 `A </s>` | 常通过任务前缀组织输入 |

BERT 通常使用 `token_type_ids=0/1` 辅助区分 A 和 B；当前 XLM-RoBERTa 的 `type_vocab_size=1`，不返回 `token_type_ids`，主要依靠分隔符和上下文区分 Query 与 Document。

## 5. Chat Template

对话模型的原始数据是结构化消息：

```python
messages = [
    {"role": "system", "content": "你是一个知识助手。"},
    {"role": "user", "content": "什么是注意力机制？"},
]
```

`apply_chat_template()` 根据 tokenizer 配置中的 Chat Template 渲染角色结构：

```python
token_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)
```

类 ChatML 结果可能是：

```text
<|im_start|>system
你是一个知识助手。
<|im_end|>
<|im_start|>user
什么是注意力机制？
<|im_end|>
<|im_start|>assistant
```

`add_generation_prompt=True` 表示在末尾补上 Assistant 回合的开头，提示模型接下来生成助手回答。

## 6. 哪些由工具辅助，哪些需要手动构造

| 场景 | 常用方式 |
|---|---|
| 单文本特殊 Token | `tokenizer(text)` 自动处理 |
| 文本对特殊 Token | `tokenizer(text, text_pair)` 自动处理 |
| 对话角色结构 | `apply_chat_template()` 渲染 |
| Encoder-Decoder 输入与目标 | `tokenizer(text=..., text_target=...)` 辅助处理 |
| Instruction 的标题和字段 | 通常由数据处理代码构造 |
| RAG 的资料编号、问题和约束 | 通常手动构造 Prompt |
| 动态 Mask | `DataCollatorForLanguageModeling` 等工具构造 |
| 图片、音频等多模态输入 | 通常交给 `AutoProcessor` |

RAG 和 Chat Template 常会叠加使用：

```text
手动组织检索资料、问题和回答约束
→ 放入 messages 的 system/user content
→ apply_chat_template 渲染角色格式
→ Tokenizer 切分并生成 Token IDs
```

## 7. Bi-Encoder 与 Cross-Encoder

Bi-Encoder 将 Query 和 Document 分开编码：

$$
e_q=E(q),
\qquad
e_d=E(d),
\qquad
s_{dense}(q,d)=e_q^Te_d.
$$

Document 向量可以离线预计算，在线阶段只需编码 Query 并搜索 FAISS，因此适合大规模召回。

Cross-Encoder Reranker 将两段文本拼成同一条序列：

```text
<s> Query </s></s> Document </s>
```

经过同一个 Transformer 后，用 `<s>` 位置的表示输出相关性分数：

$$
s_{rerank}(q,d)=Wh_{cls}+b.
$$

每换一个 Query-Document 组合都必须重新计算，不能预存通用的 Document 向量。

## 8. 双向 Encoder 与联合自注意力

“双向”指每个 Token 可以同时关注左侧和右侧的所有非 Padding Token。对于长度为 $T$ 的序列，普通 Encoder 不使用下三角 Causal Mask：

```text
          被关注的 Token
          T1  T2  T3  T4
查询 T1    ✓   ✓   ✓   ✓
查询 T2    ✓   ✓   ✓   ✓
查询 T3    ✓   ✓   ✓   ✓
查询 T4    ✓   ✓   ✓   ✓
```

因此，第 $i$ 个位置的表示可以依赖整条序列：

$$
h_i=f(x_1,x_2,\ldots,x_T).
$$

GPT 类 Decoder 则使用 Causal Mask，第 $i$ 个位置只能看到自己和左侧：

```text
          被关注的 Token
          T1  T2  T3  T4
查询 T1    ✓   ×   ×   ×
查询 T2    ✓   ✓   ×   ×
查询 T3    ✓   ✓   ✓   ×
查询 T4    ✓   ✓   ✓   ✓
```

$$
h_i=f(x_1,x_2,\ldots,x_i).
$$

将 Query 和 Document 在 Token 层拼接后，完整注意力矩阵可分块表示为：

$$
A=
\begin{bmatrix}
A_{QQ} & A_{QD}\\
A_{DQ} & A_{DD}
\end{bmatrix}.
$$

- $A_{QQ}$：Query 内部的 Token 交互。
- $A_{QD}$：Query Token 直接读取 Document Token。
- $A_{DQ}$：Document Token 根据 Query 更新自己的表示。
- $A_{DD}$：Document 内部的 Token 交互。

`</s></s>` 只负责标记两段文本的边界，不会阻断 $A_{QD}$ 和 $A_{DQ}$。Padding 位置仍由 `attention_mask` 屏蔽。

## 9. Reranker 的 PyTorch 模块结构

`bge-reranker-base` 本质是 `XLMRobertaForSequenceClassification`：

```text
input_ids [B,T]
→ Token + Position Embedding [B,T,768]
→ 12 个双向 Transformer Encoder Block
→ 取 <s> 位置 [B,768]
→ 分类头
→ relevance logits [B,1]
```

用 PyTorch 可近似表示为：

```python
token_embedding = nn.Embedding(250_002, 768, padding_idx=1)
position_embedding = nn.Embedding(514, 768)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=768,
    nhead=12,
    dim_feedforward=3072,
    dropout=0.1,
    activation="gelu",
    batch_first=True,
    norm_first=False,
)

encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

classifier = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(768, 768),
    nn.Tanh(),
    nn.Dropout(0.1),
    nn.Linear(768, 1),
)
```

前向过程的核心是：

```python
hidden_states = encoder(embeddings, src_key_padding_mask=padding_mask)
cls_hidden = hidden_states[:, 0]  # [B,768]
logits = classifier(cls_hidden)   # [B,1]
```

`logits` 是排序分数而不是概率，可以为负数。只需在同一 Query 的候选集合内按分数降序排列。

约 278M 参数的主要来源：

- 25 万规模的多语言词表 Embedding：约 192M。
- 12 层 Transformer Encoder：约 85M。
- 单输出分类头：不到 1M。

## 10. Transformer 计算复杂度从哪里来

设序列长度为 $T$，隐藏维度为 $D$，层数为 $L$，FFN 扩展倍数为 $r$。

| 模块 | 一层主要计算量 |
|---|---:|
| Q/K/V 与输出投影 | $4TD^2$ |
| $QK^T$ 与注意力权重乘 $V$ | $2T^2D$ |
| FFN 两次线性层 | $2rTD^2$ |

当 $r=4$ 时，一层约为：

$$
12TD^2+2T^2D.
$$

$L$ 层、Batch Size 为 $B$ 时：

$$
O\left(BLTD^2+BLT^2D\right).
$$

Reranker 需要对 $K$ 个候选分别计算 Query-Document pair，总计算量还要近似乘以 $K$。Batching 只提高硬件并行度，不会减少理论运算量。

不同参数对计算量的影响：

- 候选数 $K$ 翻倍，总计算量约翻倍。
- 层数 $L$ 翻倍，总计算量约翻倍。
- Token 长度 $T$ 翻倍，线性层部分翻倍，注意力矩阵部分约变成 4 倍。
- 隐藏维度 $D$ 翻倍，注意力矩阵部分约翻倍，线性层和 FFN 约变成 4 倍。

“Transformer 是 $O(T^2)$”主要指注意力矩阵；完整模型还有大量 $TD^2$ 的 QKV、输出投影和 FFN 计算。

## 11. S2 真实对照结果

固定 `chunk_size=512`、`candidate_k=20`，对 30 条 Gold QA 进行对照：

| 指标 | Dense-only | Dense + Reranker | 变化 |
|---|---:|---:|---:|
| Recall@1 | 0.200 | **0.600** | +0.400 |
| Recall@3 | 0.667 | **0.867** | +0.200 |
| Recall@5 | 0.733 | **0.933** | +0.200 |
| Recall@10 | **0.933** | **0.933** | 0 |
| MRR | 0.457 | **0.739** | +0.282 |

- 16 道题的首个正确 Anchor 排名提升。
- 3 道题排名下降，说明 Reranker 不保证每个样本都变好。
- 11 道题排名不变。
- `GraphSAGE/GAT` 题从第 10 名提升到第 1 名。

Recall@10 没有继续提升，因为两条未命中证据的 Dense 全库排名分别为 103 和 168，没有进入 Top-20 候选池：

```text
归纳偏置：Dense Anchor Rank = 103
FlashAttention/GQA：Dense Anchor Rank = 168
```

Reranker 只能重排已经召回的候选，无法找回候选池之外的 Chunk。

当前 M1 CPU 的 30 题实测耗时：

```text
Dense-only：       0.19 秒，约 0.006 秒/题
Dense + Reranker：58.54 秒，约 1.95 秒/题
```

相对耗时约为 303 倍，但绝对代价是单题增加约 1.94 秒。差距来自“预计算的短 Query 向量 + FAISS”与“20 个长 Query-Document pair 逐对运行 278M Cross-Encoder”的工作量差异。

## 12. Reranker 实现检查点

- [ ] Query 和每个 Document 独立组成 pair。
- [ ] 使用 `padding=True` 和 `truncation=True`。
- [ ] `max_length=512` 包含 4 个 pair 特殊 Token。
- [ ] 不手动拼接 Query 与 Document 字符串。
- [ ] 保留原始 dense `score`，新增 `rerank_score`。
- [ ] 每批 logits 形状从 `[B,1]` 变为 `[B]` 时使用 `squeeze(-1)`。
- [ ] 按 `rerank_score` 降序排列后再截取 `top_k`。
- [ ] 推理使用 `model.eval()` 和 `torch.inference_mode()`。

## 13. 自测

1. `tokenizer(query, document)` 与 `tokenizer(query + document)` 有什么根本差别？
	1. 一个会按照 query document对进行分词和模版渲染，一个是单纯分词
2. 为什么 XLM-RoBERTa 的 pair 中间是 `</s></s>`？
	1. 约定好的格式，表示query和document的分界
3. 为什么有 20 个候选时要构造 20 个 Query-Document pair？
	1. 计算rerank分数是按照pair进行计算的，每个候选都要计算rerank分数
4. `apply_chat_template()` 与 `tokenizer(text, text_pair)` 分别解决什么问题？
	1. 前者按照格式渲染prompt，chat用来模拟对话，可以继续分词，tokenizer按pair分词；两者工作在不同层级
5. RAG Prompt 为什么通常需要先手动构造，再放进 Chat Template？
	1. RAG的prompt需要指导llm进行rag搜索
6. `padding=True` 与 `padding="max_length"` 会生成什么不同的第二维？
	1. 前者为动态填充，先找当前批次最大length；后者固定
7. 为什么开启 `return_overflowing_tokens=True` 后，输出的 Batch 维可能大于原始文本数？
	1. 相当于把单文本内容进行裁剪，一个文本可能出现多个窗口
8. 双向 Encoder 和 Causal Decoder 的注意力许可范围有什么区别？
	1. 双向encoder每个token都可以感知到前后所有token，因果只能看到前面的
9. 联合注意力矩阵中的 $A_{QD}$ 和 $A_{DQ}$ 各表示什么？
	1. Query对document token的注意力后者相反
10. 为什么 Reranker 可以显著提高 MRR，却不一定能提高 Recall@10？
	1. reranker是精排行为，无法改善召回率
11. Transformer 复杂度中的 $T^2D$ 和 $TD^2$ 分别来自哪些运算？
	1. 前者来自注意力矩阵相乘，后者来自线性层

### 自测标准答案

> [!success]- 标准答案：1. 文本对保留边界，拼字符串不会自动保留
> `tokenizer(query, document)` 将两段文本作为 A/B 输入，套用该模型的 pair 特殊 Token 模板，并支持面向两段文本的截断策略；有的模型还返回 `token_type_ids`。`tokenizer(query + document)` 是一个普通单文本输入，不会自动知道哪里是问题、哪里是文档，也不会自动使用 pair 模板。

> [!success]- 标准答案：2. 双结束标记来自模型的输入约定
> XLM-RoBERTa 的文本对模板为 `<s> A </s></s> B </s>`，中间的两个 `</s>` 是其约定格式，不是我们临时添加的技巧。应交给对应 Tokenizer 构造，以保持与模型预训练时的格式一致。它们标记边界，但不是 causal mask，也不会禁止两段文本通过自注意力交互。

> [!success]- 标准答案：3. 每个候选都需要一个独立相关性分数
> Cross-Encoder 的评分对象是一个 `(query, document)` 对。20 个候选对应 20 个 pair，问题会在这些 pair 中重复，模型分别输出 20 个分数，再据此排序。它们可以分批推理，不必一次全部放入显存；也不能简单把 20 篇文档拼成一个 pair，还期待得到独立的 20 个分数。

> [!success]- 标准答案：4. 对话模板与文本对模板不是一回事
> `apply_chat_template()` 组织带 role 的消息、角色边界、轮次结束符和可选的 assistant 生成起始标记，也可以进一步分词。`tokenizer(text, text_pair)` 为两段文本应用模型的单句/文本对编码规则，生成 Token IDs、attention mask 等。前者服务对话模型的消息协议，后者常用于 Encoder 的句对任务。

> [!success]- 标准答案：5. 任务内容与模型协议分两层处理
> RAG 应用负责决定放哪些证据、如何编号引用、保留多少上下文，以及如何表达问题和拒答约束；Chat Template 不会自动完成检索和证据预算分配。因此先把这些内容组织成 system/user 等消息，再用模型的 Chat Template 渲染。若使用 Chat API，通常直接发送消息列表，由服务端处理模型模板，避免重复套模板。

> [!success]- 标准答案：6. 动态补齐与固定长度补齐
> 结合本项目的截断设置，`padding=True` 补齐到当前批次截断后最长序列的长度，因此第二维随批次变化。`padding="max_length"` 补齐到指定的 max_length（未指定时使用模型支持的默认上限）。Padding 本身不负责把超长序列裁短；要保证统一上限还需相应的 truncation 配置。

> [!success]- 标准答案：7. 一条文本可以变成多个窗口
> 开启截断与 `return_overflowing_tokens=True` 后，在支持该组合的 Tokenizer 中，一条长文本可以产生多个窗口；设置 stride 时窗口还可重叠。因此输出第一维表示窗口数，可能大于原始样本数。支持时可通过 `overflow_to_sample_mapping` 找回窗口对应的原样本；某些文本对截断组合不支持返回这种溢出窗口。

> [!success]- 标准答案：8. 双向与因果 mask 的许可范围
> 双向 Encoder 中，每个有效 Token 通常可以关注序列内前后所有有效 Token。Causal Decoder 中，位置 t 只能关注自己及更早的位置，不能读取未来 Token。两者都还需要处理 padding；在带 KV Cache 的增量计算中，因果关系按实际位置判断，注意力矩阵也不一定是方阵。

> [!success]- 标准答案：9. QD 与 DQ 是两段文本之间的信息读取
> 这里下标 Q/D 代表问题段和文档段，不是 QKV 投影的名称。若注意力矩阵的行是读取信息的位置、列是被读取的位置，$A_{QD}$ 表示问题 Token 对文档 Token 的注意力，$A_{DQ}$ 表示文档 Token 对问题 Token 的注意力。两者使文本对联合建模，但通常不互为转置，因为每一行分别归一化。

> [!success]- 标准答案：10. 排得更靠前，不一定多找回证据
> 若正确证据从第 8 名升到第 1 名，RR 从 $1/8$ 升到 1，但 Recall@10 仍为 1，所以 MRR 可以显著提高而 Recall@10 不变。本项目未命中的证据还在候选池之外，重排看不到它们。不过当候选池大于 10 且正确证据原在第 11 名之后时，重排也可能提高 Recall@10；不能把本次结果当作普遍不变规律。

> [!success]- 标准答案：11. 注意力矩阵与特征投影的成本不同
> $QK^\mathsf{T}$ 和注意力权重乘 V 涉及所有 Token 两两交互，总量级为 $O(T^2D)$。Q/K/V 投影、注意力输出投影，以及扩展倍数固定的 FFN，主要贡献 $O(TD^2)$。这里 D 是总隐藏维度；还要乘 batch 数和层数，并忽略常数。因此完整 Transformer 不能只用“序列长度平方”概括所有成本。
