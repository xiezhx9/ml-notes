---
tags:
  - LLM
  - SFT
  - ChatML
  - CrossEntropy
  - DataLoader
aliases:
  - SFT数据管线
  - Assistant Only Loss
---

# Task 3.2：SFT 数据与 Assistant-Only Loss

> [!summary]
> SFT 没有改变 causal LM 的本质：仍然用前文预测下一个 token。它新增的是对话模板和监督范围控制。模型读取 system/user/assistant 全部上下文，但只在 assistant 回答 token 上计算 CrossEntropy。

返回总览：[[Task3-SFT-DPO学习索引]]；参数更新方法见 [[Task3-01-LoRA与Adapter生命周期]]。

## 1. SFT 的输入和目标

一条多轮数据：

```python
[
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "2 加 2 等于多少？"},
    {"role": "assistant", "content": "2 加 2 等于 4。"},
]
```

被渲染为 ChatML：

```text
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
2 加 2 等于多少？<|im_end|>
<|im_start|>assistant
2 加 2 等于 4。<|im_end|>
```

tokenizer 再生成：

$$
\text{input\_ids}\in\mathbb Z^T,qquad
\text{attention\_mask}\in\{0,1\}^T,qquad
\text{labels}\in(\mathbb Z\cup\{-100\})^T.
$$

## 2. 三个张量各管什么

| 张量 | 送给谁 | 作用 |
|---|---|---|
| `input_ids` | 模型 Embedding | 模型实际读取的 token，包括 prompt 和回答 |
| `attention_mask` | Transformer attention | 区分真实 token 与 batch padding |
| `labels` | CrossEntropy | 指定哪些位置需要监督以及正确 token ID |

> [!important]
> `attention_mask=0` 和 `labels=-100` 不是一回事。前者控制模型是否把 padding 当上下文；后者控制该位置是否贡献 loss。Prompt token 通常 `attention_mask=1`、`labels=-100`：模型要读它，但不用预测它。

## 3. 为什么不是把整句话都当 label

如果所有 token 都产生 loss，模型会同时学习：

- system prompt；
- user 问题；
- assistant 回答；
- 特殊标记。

这会浪费优化目标，甚至强化模型模仿用户输入。Assistant-only SFT 使用掩码：

$$
m_t=
\begin{cases}
1,&t\text{ 属于 assistant 回答区域},\\
0,&\text{其他位置}.
\end{cases}
$$

$$
\mathcal L_{SFT}
=-\frac{1}{\sum_t m_t}
\sum_t m_t\log p_\theta(y_t|y_{<t}).
$$

PyTorch 用 `ignore_index=-100` 表示 $m_t=0$。

## 4. `-100` 到底发生了什么

```python
labels = input_ids.clone()
labels[not_assistant] = -100
loss = F.cross_entropy(logits, labels, ignore_index=-100)
```

它不是一个要预测的类别，也不是“这个 token 的预测概率为 0”。CrossEntropy 会把这些位置从求和与平均分母中移除：

$$
\frac{\partial\mathcal L}{\partial z_t}=0,
\qquad labels_t=-100.
$$

但被 ignore 的 prompt 仍影响后续 assistant token 的隐藏状态，所以仍会通过后续有效 loss 间接参与梯度路径。

### 4.1 `reduction` 的含义

| 配置 | 输出 | 用途 |
|---|---|---|
| `mean` | 有效 token loss 的平均值 | 默认训练 |
| `sum` | 有效 token loss 总和 | 自己控制全局归一化 |
| `none` | 每个位置的 loss | 分析 token 级损失 |

模型内置 causal LM loss 通常已经使用 `ignore_index=-100` 和平均 reduction。

## 5. Causal shift

模型位置 $t$ 的 logits 用来预测位置 $t+1$ 的 token：

```text
logits: [z0, z1, z2, ..., z(T-2)]
labels: [y1, y2, y3, ..., y(T-1)]
```

所以监督实际是：

$$
z_t\rightarrow y_{t+1}.
$$

若自己计算 loss，需要：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

使用 `AutoModelForCausalLM(..., labels=labels)` 时，Hugging Face 模型内部会完成 shift，不要再手动 shift 一次。

## 6. 如何定位 assistant 区域

本项目先 tokenizer 两段固定标记：

```text
<|im_start|>assistant\n
<|im_end|>\n
```

再用滑动窗口查找 token 子序列：

```python
matches = (
    input_ids.unfold(0, marker.numel(), 1) == marker
).all(-1).nonzero()
```

对每个 assistant start，找到它后面的第一个 end，把中间回答和 end token 复制到 labels，其余保持 `-100`。

### 6.1 为什么不能假设标记是一个 token

tokenizer 可能把 `assistant`、换行或完整 ChatML marker 拆成多个 token。正确做法是：

- 用当前模型的 tokenizer 对完整 marker 编码；
- 在 token ID 序列中匹配整个子序列；
- 不要硬编码某个 ID；
- 不要假设公共 tokenizer 的不同版本有相同特殊 token 表。

### 6.2 保留 `<|im_end|>` 好还是不保留

保留 end token 的 loss，模型会学习何时结束回答；不保留则减少格式监督。本项目保留 assistant 的 end marker，但关键是 SFT 与 DPO 的响应范围保持一致，不能一边保留、一边忽略导致 log-probability 口径变化。

## 7. 验证对话格式

至少检查：

- 角色只能是 `system/user/assistant`；
- system 若存在，只能在第一条；
- user 与 assistant 按顺序交替；
- 不允许缺失 content 或空对话；
- 多轮记录按真实 turn 编号排序，不能依赖 JSON 字典碰巧的顺序。

对于 MOSS 数据，还需要去掉原始 `<|Human|>`、`<|MOSS|>` 等外层标记，避免重复模板。

## 8. 截断的风险

```python
input_ids = input_ids[:max_length]
labels = labels[:max_length]
```

这种右侧截断简单，但可能把 assistant 回答全部截掉。若某条样本截断后：

$$
\sum_t [labels_t\ne-100]=0,
$$

那么 mean CrossEntropy 没有有效分母，可能返回 `NaN`。

> [!warning] `batch_size=1` 不能保证没有 NaN
> 关键不是一个 batch 有几条样本，而是整个 batch 是否至少存在有效 assistant token。增大 batch size 只是提高“混入一条有效样本”的概率，不能修复错误的截断策略。

更稳健的做法：

1. 编码后统计有效 label 数；
2. 无有效 token 的样本直接丢弃或重新截断；
3. 优先保留 assistant 回答预算；
4. 记录被丢弃/截断的样本比例。

## 9. Collate 与 batch padding

一批样本长度不同，collate 到当前 batch 的最大长度：

```python
input_ids       -> pad_token_id
attention_mask  -> 0
labels          -> -100
```

最终形状：

$$
input\_ids,attention\_mask,labels\in\mathbb Z^{B\times T_{batch}}.
$$

选择 batch 内动态 padding 而不是全数据集固定 padding，可以减少无效计算。

> [!tip] 性能技巧
> 当前实现循环中反复 `torch.cat`，学习规模可用；更高效的实现可以先收集 Tensor 列表，再用 `pad_sequence` 一次堆叠，避免反复分配和复制。

## 10. SFT 训练步骤

```python
model.train()
optimizer.zero_grad()

for micro_batch in dataloader:
    outputs = model(
        input_ids=micro_batch["input_ids"],
        attention_mask=micro_batch["attention_mask"],
        labels=micro_batch["labels"],
    )
    (outputs.loss / accumulation_steps).backward()

    if should_update:
        clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

只有 LoRA A/B 在 optimizer 中，base 权重虽然参与计算图传递，但不会累积参数梯度。

## 11. 梯度累积为何要除以次数

假设有效大 batch 由 $K$ 个 micro-batch 构成，目标是平均梯度：

$$
g=\frac1K\sum_{k=1}^K\nabla L_k.
$$

每次 backward 前除以 $K$：

$$
\nabla\left(\frac{L_k}{K}\right)=\frac1K\nabla L_k.
$$

否则累积得到梯度和，等效学习率随 $K$ 放大。

最后不足 $K$ 个 micro-batch 时，要除以实际数量 $K_{last}$。例如期望累积 5 次，最后只有 3 次，就分别使用 `loss / 3`。

有效 batch size：

$$
B_{effective}=B_{micro}\times K\times N_{devices}.
$$

更严格地说，变长序列任务最好按有效 token 数归一化，而不是默认每个 micro-batch 含有相同数量的监督 token。

## 12. Plugin SFT 与普通 SFT

工具轨迹被转换为：

```text
system: meta instruction
user: human request
assistant: inner thoughts + commands
user: tool responses
assistant: final answer
```

仍然复用同一个 ChatML、assistant-only labels、collate 和 SFT loss。区别主要在数据 schema 和评估：

- 训练目标同时包含命令格式和最终回答；
- 工具返回值作为上下文而不是监督目标；
- 必须校验 commands、tool responses 非空；
- 离线格式正确不等于真实工具能执行。

## 13. 自测

1. Prompt labels 为 `-100` 后，模型为什么仍能根据 prompt 学会回答？
2. `attention_mask=0` 能否代替 `labels=-100`？
3. 为什么整个 batch 全是 `-100` 会产生 NaN？
4. `input_ids=[B,T]`、模型输出 `[B,T,V]` 时，CrossEntropy 如何计算？
5. 梯度累积最后只剩 3 个 micro-batch 时，应除以几？

> [!answer]- 参考答案
> 1. Prompt 仍参与前向并影响 assistant hidden states，有效回答 loss 的梯度会穿过这段上下文。
> 2. 不能。attention mask 控制上下文可见性，ignore index 控制监督位置。
> 3. mean reduction 没有有效 token 作为分母。
> 4. 内部先 causal shift，再把 `[B,T-1,V]` 与 `[B,T-1]` 展平计算分类损失。
> 5. 除以 3，保持最后一次更新也是实际 micro-batch 的平均梯度。
