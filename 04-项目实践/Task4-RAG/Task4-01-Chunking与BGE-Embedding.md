---
tags:
  - LLM
  - RAG
  - Chunking
  - BGE
  - Embedding
  - PyTorch
aliases:
  - Task4 Chunking
  - BGE Embedding
---

# Task 4.1：Chunking 与 BGE Embedding

> [!summary]
> RAG 的第一段数据流是把 PDF 变成可检索的 Chunk，再用小型 Transformer Encoder 把每个 Chunk 编码为定长向量。Chunking 决定证据边界，Embedder 决定语义检索空间。

返回总览：[[Task4-RAG学习索引]]

## 1. PDF 文本抽取与归一化

基线处理流程：

```text
PDF
→ 按页 extract_text
→ 统一 \r\n / \r / \n
→ 删除空行与行首尾空白
→ 保留 page/source 元数据
```

当前归一化是最小基线，能清理普通正文，但会丢失段落空行、代码缩进和部分公式版式。不应一开始就写过度复杂的规则；先用 gold anchor 和 Recall 判断实际影响，再针对失败样本改进。

## 2. 字符级滑动窗口

设窗口大小为 $C$，重叠长度为 $O$，则步长为：

$$
S=C-O,
\qquad C>0,\quad 0\le O<C.
$$

第 $i$ 个 Chunk 的边界：

$$
start_i=iS,
\qquad
end_i=\min(start_i+C,L).
$$

`overlap` 用来保护跨边界证据。如果一句关键定义恰好被切成两半，没有重叠时两边都可能无法命中 gold anchor。

### 2.1 已遇到的边界问题

- `chunk_size <= 0` 必须报错。
- `overlap < 0` 或 `overlap >= chunk_size` 必须报错。
- `overlap=0` 是合法配置。
- 空文本应返回 `[]`，不能返回 `[""]`。
- 已经到达文本末尾后应立即停止，不能额外生成一个完全包含在上一块中的尾部 Chunk。
- PDF 空页的 `extract_text()` 可能返回 `None`，需要转为空字符串。

### 2.2 为什么需要 overlap：保护跨边界的上下文

例如原文包含：

```text
梯度裁剪用于缓解梯度爆炸。当梯度范数超过阈值时，梯度裁剪会按比例缩小梯度，但不会直接限制参数大小。
```

没有 overlap 时，窗口边界可能恰好落在解释中间：

```text
Chunk 1：梯度裁剪用于缓解梯度爆炸。当梯度范数超过阈值时，
Chunk 2：梯度裁剪会按比例缩小梯度，但不会直接限制参数大小。
```

Chunk 1 缺少具体操作，Chunk 2 缺少触发条件。检索“梯度裁剪什么时候生效、怎么做”时，单个 Chunk 可能不足以回答完整问题。

如果将边界前的内容重复放入下一块：

```text
Chunk 1：梯度裁剪用于缓解梯度爆炸。当梯度范数超过阈值时，
Chunk 2：当梯度范数超过阈值时，梯度裁剪会按比例缩小梯度，但不会直接限制参数大小。
```

第二块就同时包含了条件与操作。这里是语义示意；实际字符窗口仍按照固定边界切分，不会自动识别句子。

例如 `chunk_size=100`、`overlap=20`，窗口每次前进 `80` 个字符：

```text
Chunk 1：[0, 100)
Chunk 2：    [80, 180)
Chunk 3：         [160, 260)
```

> [!important]
> Overlap 的作用是降低边界截断风险，不是保证所有知识点都完整。知识点过长、Overlap 太短，或 PDF 本身抽取错误时，仍可能缺失证据。当前按页切分时，页内 overlap 也不会自动解决跨页的语义截断。

### 2.3 为什么 overlap 不是越大越好

相邻 Chunk 重复内容越多，通常越容易得到相似的 Embedding。检索结果可能变成：

```text
Top 1：Chunk 10，某段内容
Top 2：Chunk 11，基本还是这段内容
Top 3：Chunk 12，基本还是这段内容
```

名义上召回了三条，实际新增信息很少，其他相关证据还可能被挤出 top-k。

- **存储与计算成本增加**：步长变小，Chunk 数量上升，需要更多编码、索引存储与候选处理。
- **检索多样性下降**：有限的候选名额可能被近重复内容占据。
- **上下文预算被重复消耗**：若直接拼接候选，Generator 会反复看到同一段文字。

因此需要平衡“保留跨边界证据”和“减少冗余”。应结合固定评测集上的 Recall/MRR、检索结果多样性与成本选取 overlap，而不是只看单个例子。我们的 S1 固定 overlap 比例为 `1/8`，比较的是不同 chunk size；它不能单独证明 `1/8` 是最优 overlap 比例。

### 2.4 按 chunk_id 去重，不等于去掉重叠文本

当前上下文组织按 `(source, chunk_id)` 去重，只能删除**同一个 Chunk 被重复召回**的情况。

```text
Chunk 10 + Chunk 10：可以按 ID 去重。
Chunk 10 + Chunk 11：ID 不同，即使文本重叠，也会保留。
```

减少不同 Chunk 间的冗余，需要额外方法，例如根据来源和文本区间合并相邻窗口、识别重复内容，或在候选选择中兼顾相关性与多样性。这些是进一步优化方向，不是当前 ID 去重已经实现的能力。

> [!question] 自测
> 1. 为什么没有 overlap 时，召回主题相关的 Chunk 仍可能无法完整回答问题？
> 	1. 核心内容可能拆分到不同chunk内部，单条chunk不能包含完整信息
> 2. 如果 overlap 接近 chunk_size，Chunk 数量与候选多样性会如何变化？
> 	1. 重复内容过多，多样性减少，chunk数量增多
> 3. 为什么按 chunk_id 去重不能删除相邻 Chunk 的重叠内容？
> 	1. `(source, chunk_id)` 识别的是一个具体 Chunk，而不是某段文本。`Chunk 10 + Chunk 10` 可按 ID 去重；`Chunk 10 + Chunk 11` 的 ID 不同，即使共享一段文字也都会保留。要消除这种冗余，需要额外识别重叠文本区间并合并，或进行内容去重、多样性筛选；不能把当前的 ID 去重当作已经完成这些优化。

> [!success]- 标准答案：1. 主题相关不等于证据完整
> 切分边界可能把定义、触发条件、操作步骤或限定条件拆到两个 Chunk 中。被召回的块虽然包含主题关键词，却缺少回答所需的另一半信息。例如一块只写“梯度范数超过阈值时”，另一块才说明“按比例缩小梯度”。Overlap 能让边界附近的内容同时出现在相邻块里，降低证据被截断的风险，但不能保证任意长度的知识点都完整。

> [!success]- 标准答案：2. Chunk 更多，但信息不一定更多
> 固定文本长度和 chunk_size 时，步长为 `chunk_size - overlap`。Overlap 越接近 chunk_size，步长越小，通常需要生成越多的 Chunk。相邻块包含大量重复内容，Embedding 往往也更相似，可能一起占据 top-k，降低候选的信息多样性。代价包括更多编码计算、索引存储和重复上下文；并不意味着每次检索的 Recall 都必然下降。

> [!success]- 标准答案：3. ID 去重与内容去重是两回事
> `(source, chunk_id)` 识别的是一个具体 Chunk，而不是某段文本。`Chunk 10 + Chunk 10` 可按 ID 去重；`Chunk 10 + Chunk 11` 的 ID 不同，即使共享一段文字也都会保留。要消除这种冗余，需要额外识别重叠文本区间并合并，或进行内容去重、多样性筛选；不能把当前的 ID 去重当作已经完成这些优化。

## 3. BGE 是什么

BGE 是 BAAI General Embedding 系列。它不是单独的 `nn.Embedding` 查找表，而是用于生成文本向量的小型 Transformer Encoder。

```text
文本
→ Tokenizer
→ Token IDs
→ Token/Position Embedding
→ 多层双向 Transformer Encoder
→ 取 CLS
→ L2 Normalize
→ 文本向量
```

当前 `bge-small-zh-v1.5` 的关键规模：

| 项目 | 数值 |
|---|---:|
| 参数量 | `23,953,920` |
| B 单位 | 约 `0.024B` |
| Encoder 层数 | `4` |
| 隐藏维度 | `512` |
| 注意力头 | `8` |
| 输出向量维度 | `512` |

`bge-reranker-base` 约为 `0.278B`，它联合读取 `[query, document]` 文本对，比 Bi-Encoder 召回慢，因此只对少量候选执行精排。

## 4. `nn.Embedding` 与 BGE Encoder

`nn.Embedding(V,D)` 只是参数矩阵查表：

$$
i\mapsto E_i\in\mathbb R^D.
$$

同一 Token ID 的初始向量固定。BGE 会在查表后继续通过双向自注意力读取上下文：

```text
“我吃了一个苹果” → 苹果偏向水果
“苹果发布了新手机” → 苹果偏向公司
```

因此，BGE 的“Embedding”指模型用途和最终产物，不代表整个模型只有一张 Embedding 表。

## 5. 输入输出形状

输入 $N$ 段文本，Tokenizer 对当前 batch padding 到长度 $T$：

$$
[N,T]
\xrightarrow{\text{Encoder}}
[N,T,512]
\xrightarrow{\text{CLS}}
[N,512].
$$

- $N$：文本或 Chunk 数量。
- $T$：当前 batch padding 后的 Token 数量。
- $D=512$：每段文本的向量维度。
- 成段的文本会进行pool池化，和mean pool不同，BGE选择首位token作为句子的统计信息

即使只编码一个问题，也应保留二维形状 `(1, 512)`。空输入建议返回 `float32` 的 `(0, 512)` 数组。

## 6. 为什么取 CLS，不取 `pooler_output`

BGE 训练 embedding 时使用最后一层 CLS 的原始隐藏状态：

$$
e_{CLS}=H[:,0,:].
$$

`pooler_output` 不是对全部 Token 求平均，而是对 CLS 再做一次变换：

$$
e_{pooler}=\tanh(We_{CLS}+b).
$$

这个额外映射会改变 BGE 对比学习已经建立的向量空间。推理时应与模型的训练约定保持一致：

```python
embeddings = outputs.last_hidden_state[:, 0]
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
```

CLS 能代表整段文本，是因为它在每一层自注意力中都能读取其他 Token，且 BGE 训练目标专门将语义信息压入该位置。

## 7. L2 归一化与余弦相似度

对文本向量执行：

$$
\hat e=\frac{e}{\lVert e\rVert_2}.
$$

归一化后 $\lVert\hat e\rVert_2=1$，两个向量的内积等于余弦相似度：

$$
\hat q^T\hat d=\cos(\hat q,\hat d).
$$

因此，FAISS `IndexFlatIP` 只有在 query 和 document 都已 L2 normalize 时，才可以按余弦相似度理解。

## 8. Batch Size 在 Embedder 中的作用

`batch_size` 只控制一次向 BGE 送入多少段文本，不改变最终输出形状。

例如 $N=1551$、`batch_size=32`：

```text
第 1 批:  [32, 512]
第 2 批:  [32, 512]
...
最后一批: [15, 512]
拼接:    [1551, 512]
```

一次处理全部文本可能因 $[B,H,T,T]$ 注意力中间量而耗尽内存；每次只处理一条又无法充分利用矩阵并行。分批是速度与内存的工程折中。

不同样本不会跨 batch 维度互相注意，所以分批不会改变每条文本的语义计算。

## 9. `model.eval()`、`no_grad()` 与 `inference_mode()`

| 方法 | 控制对象 | Dropout/BatchNorm | 是否构建反向图 | 主要用途 |
|---|---|---|---|---|
| `model.eval()` | 模型层的运行模式 | 改变 | 仍会构建 | 评估行为 |
| `torch.no_grad()` | Autograd | 不改变 | 不构建 | 灵活验证/推理 |
| `torch.inference_mode()` | Autograd 与 Tensor 跟踪 | 不改变 | 不构建 | 纯推理 |

### 9.1 `model.eval()`

```python
model.eval()
```

它会递归设置 `module.training=False`，主要关闭 Dropout 的随机丢弃，并让 BatchNorm 使用已保存的运行均值与方差。它不会修改参数的 `requires_grad`，也不会自动禁用计算图。

### 9.2 `torch.no_grad()`

```python
with torch.no_grad():
    outputs = model(inputs)
```

它临时不记录反向传播图，从而减少内存和计算开销，但不会改变 Dropout 或 BatchNorm 的行为。`model.train()` 状态下使用 `no_grad()`，Dropout 仍然开启。

装饰器写法：

```python
@torch.no_grad()
def evaluate(...):
    ...
```

### 9.3 `torch.inference_mode()`

```python
with torch.inference_mode():
    outputs = model(inputs)
```

它在 `no_grad()` 的基础上，进一步关闭 View 跟踪、Version Counter 等与训练有关的 Tensor 维护，通常更省内存、稍快，但生成的 inference tensor 也更难在之后重新接入训练图。

纯 Embedding 推理的推荐组合：

```python
model.eval()

with torch.inference_mode():
    outputs = model(**tokenized_inputs)
```

> [!important] 三句话记忆
> `eval()` 决定层怎么运行；`no_grad()` 决定要不要准备反向传播；`inference_mode()` 表示这是纯推理，尽量关闭所有训练跟踪。

## 10. Embedder 工程检查表

- [ ] `batch_size > 0`。
- [ ] 空输入返回 `(0, hidden_size)` 的 `float32` NumPy 数组。
- [ ] Tokenizer 开启 `padding=True` 和 `truncation=True`。
- [ ] Tokenizer 产生的 Tensor 与模型位于同一 device。
- [ ] 取 `last_hidden_state[:, 0]`，不取 `pooler_output`。
- [ ] 根据配置对 document/query 都执行 L2 normalize。
- [ ] 只对 query 添加 BGE 中文检索 instruction。
- [ ] 推理使用 `eval()` 和 `inference_mode()`。
- [ ] 分批结果先收集、最后一次拼接，避免反复 `cat` 复制。
- [ ] 返回前执行 `.float().cpu().numpy()`。
- [ ] 验证输出形状为 `[N,512]`，归一化后每行范数约为 1。

## 11. 自测

1. `model.eval()` 为什么不能替代 `torch.inference_mode()`？
	1. inferencemode会关闭计算图的梯度回传，eval会关闭dropout，两者功能不同
2. `torch.no_grad()` 内的 Dropout 为什么仍可能生效？
	1. no grad只是关闭autograd
3. `pooler_output` 为什么不等于对所有 Token 做 mean pooling？
	1. pool有多种方法吧，双重注意力的情况下，每个token都可以汇总其他token的信息，取cls就行
4. 为什么已归一化向量上的内积等于余弦相似度？
	1. 已归一化向量的模为1
5. Embedder 中的 batch size 为什么不会改变最终输出的 $N$？
	1. 只是为了缓解内存压力，分批次执行最后合并

### 自测标准答案

> [!success]- 标准答案：1. eval 与 inference_mode 的职责不同
> `model.eval()` 设置模块的运行模式，例如关闭 Dropout、让默认跟踪运行统计量的 BatchNorm 使用这些统计量，但不会关闭 Autograd。`torch.inference_mode()` 关闭梯度记录以及部分训练相关的张量维护，却不会自动调用 `eval()`。因此纯 Embedding 推理通常同时使用两者。

> [!success]- 标准答案：2. no_grad 不会关闭 Dropout
> `no_grad()` 控制是否记录计算图，不控制模块的 `training` 标志。如果模型仍处于 `train()` 模式，Dropout 就仍会随机丢弃元素。想获得确定的常规推理行为，应先调用 `eval()`；它与是否需要梯度是两个独立选择。

> [!success]- 标准答案：3. pooler_output 不是平均池化
> 以 BERT 风格模型为例，`pooler_output` 通常由最后一层 CLS 表示经过额外的线性层和 tanh 得到，而 mean pooling 是对各有效 Token 的隐藏表示求平均。两者计算方式和训练用途不同，具体定义还取决于模型。我们使用的 BGE 路径取 `last_hidden_state[:, 0]` 再做 L2 归一化，而不是直接使用 `pooler_output`。

> [!success]- 标准答案：4. 单位向量的内积就是余弦相似度
> 余弦相似度为 $\cos(q,d)=\frac{q^\mathsf{T}d}{\|q\|_2\|d\|_2}$。当两个非零向量都经过 L2 归一化后，范数均为 1，分母变为 1，因此内积就是余弦相似度。只归一化查询而不归一化文档，一般不能得到同样结论；零向量也没有通常意义下的余弦方向。

> [!success]- 标准答案：5. batch size 只决定分几次计算
> $N$ 是输入文本总数，batch size 是单次送入模型的文本数。每条文本得到一个 $D$ 维向量，各批沿第 0 维拼接，最终仍为 $[N,D]$。例如 10 条文本、batch size 为 4，三批形状分别是 `[4,D]`、`[4,D]`、`[2,D]`，拼接后是 `[10,D]`。
