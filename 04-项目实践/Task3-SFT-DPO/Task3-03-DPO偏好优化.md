---
tags:
  - LLM
  - DPO
  - Preference-Optimization
  - Alignment
aliases:
  - DPO原理与实现
  - 偏好优化
---

# Task 3.3：DPO 偏好优化

> [!summary]
> DPO 从同一个 SFT checkpoint 构造可训练 policy 和冻结 reference。对每条 `(prompt, chosen, rejected)`，它推动 policy 相对于 reference 更偏好 chosen，而不是简单要求 `policy_chosen > policy_rejected`。

返回总览：[[Task3-SFT-DPO学习索引]]；SFT 数据基础见 [[Task3-02-SFT数据与Assistant-Only-Loss]]。

## 1. DPO 在后训练中的位置

SFT 数据告诉模型一个“标准答案”，DPO 数据告诉模型两个答案的相对偏好：

```python
{
    "conversations": [{"from": "human", "value": "问题"}],
    "chosen": {"from": "gpt", "value": "更好的回答"},
    "rejected": {"from": "gpt", "value": "较差的回答"},
}
```

典型顺序：

$$
\text{Base}\xrightarrow{SFT}\text{SFT Model}
\xrightarrow{DPO}\text{Aligned Model}.
$$

DPO 通常不是 SFT 的替代品。没有先学会基本指令行为，直接用少量偏好对很难得到稳定模型。

### 1.1 常见原始数据格式

DPO 样本的本质都是同一组 $(x,y_w,y_l)$：公共 prompt $x$、偏好回答
$y_w$（chosen）和较差回答 $y_l$（rejected）。常见 JSON/JSONL 形式如下。

**字符串三元组**：

```json
{
  "prompt": "如何缓解轻微的头痛？",
  "chosen": "可以先休息、补充水分；如果持续或加重，建议咨询医生。",
  "rejected": "不用管，肯定会自己好。"
}
```

**公共对话 Prompt 加两种回答**：

```json
{
  "prompt": [
    {"role": "system", "content": "你是一位严谨的助手。"},
    {"role": "user", "content": "解释一下梯度下降。"}
  ],
  "chosen": [
    {"role": "assistant", "content": "梯度下降沿损失函数的负梯度方向更新参数。"}
  ],
  "rejected": [
    {"role": "assistant", "content": "梯度下降就是随便调整参数。"}
  ]
}
```

**两段完整对话**：chosen 和 rejected 都保存完整 messages，解析器需要提取两边
完全相同的公共前缀作为 prompt，再分别保留后续回答。

无论源字段如何命名，进入训练前都要构造：

```text
chosen sequence   = prompt + chosen answer
rejected sequence = prompt + rejected answer
```

chosen/rejected 的回答长度可以不同，经过 collate 后才在 batch 内动态 padding。
Reference 的 log-probability 由冻结模型现场计算或预计算缓存，不是偏好数据必须提供的字段。

## 2. Policy 和 Reference 在 PyTorch 中是什么

两者都是完整的 `nn.Module`：

| 模型 | 初始权重 | 是否训练 | 作用 |
|---|---|---|---|
| policy $\pi_\theta$ | base + SFT adapter | 是 | 继续更新 LoRA，成为 DPO 模型 |
| reference $\pi_{ref}$ | base + 同一 SFT adapter | 否 | 提供固定比较基准 |

```python
policy = load_base_inject_and_load_sft()
reference = load_base_inject_and_load_sft()
reference.eval()
reference.requires_grad_(False)
```

reference 前向最好放在 `torch.no_grad()` / `torch.inference_mode()` 中，冻结参数虽不产生参数梯度，但显式 no-grad 还能减少 autograd 图和内存。

## 3. 偏好样本如何编码

同一个 prompt 分别拼接回答：

```text
chosen sequence  = prompt + chosen answer
rejected sequence = prompt + rejected answer
```

需要返回六个张量：

```text
chosen_input_ids / chosen_attention_mask / chosen_labels
rejected_input_ids / rejected_attention_mask / rejected_labels
```

labels 中：

- prompt 区域为 `-100`；
- padding 为 `-100`；
- 只有回答区域保留 token ID；
- chosen/rejected 可以不同长，collate 后才 pad 到同一个 batch 长度。

## 4. 长序列截断：Prompt Keep 与 Answer Budget

若直接保留序列开头，长 prompt 可能把答案全部挤掉。项目策略：

$$
prompt\_keep=\min(prompt\_length,\lfloor max\_length/2\rfloor).
$$

截取起点：

$$
start=prompt\_length-prompt\_keep.
$$

这样最多给 prompt 一半预算，剩余空间尽量留给回答。

`answer budget` 指最大长度中可供回答使用的 token 数：

$$
answer\_budget=max\_length-prompt\_keep.
$$

> [!warning]
> `max_length` 可能比回答本身还短，所以仍可能截断 answer 或 end tag。必须统计有效回答 token，并决定是丢弃、保留前缀还是提高长度。

## 5. 序列 log-probability

模型输出：

$$
logits\in\mathbb R^{B\times T\times V}.
$$

先计算词表维度的 log-softmax：

$$
\log p_{b,t,v}
=z_{b,t,v}-\log\sum_{j=1}^{V}e^{z_{b,t,j}}.
$$

`dim=-1` 只在每个位置的 $V$ 个候选 token 中归一化，并不是对序列 token 求和。

随后 causal shift，并用 `gather` 选出真实标签的 log-prob：

```python
log_probs = logits[:, :-1].log_softmax(dim=-1)
target = labels[:, 1:]
token_logps = log_probs.gather(-1, target.clamp_min(0).unsqueeze(-1))
token_logps *= (target != -100)
sequence_logp = token_logps.sum(dim=-1)
```

`clamp_min(0)` 只是避免 `gather` 用 `-100` 索引越界；随后 mask 会把这些位置乘零。

序列得分：

$$
\log\pi(y|x)=\sum_{t\in response}\log\pi(y_t|x,y_{<t}).
$$

### 5.1 求和还是平均

- **求和**：与标准 DPO 公式一致，但长回答天然累积更多负 log-prob；
- **平均**：更接近长度归一化质量，但改变目标含义。

本项目使用求和，并让 chosen/rejected 共享 prompt 与统一截断协议。报告结果时仍要关注答案长度偏差。

## 6. DPO Loss

Policy 的偏好差：

$$
\Delta_\pi
=\log\pi_\theta(y_c|x)-\log\pi_\theta(y_r|x).
$$

Reference 的偏好差：

$$
\Delta_{ref}
=\log\pi_{ref}(y_c|x)-\log\pi_{ref}(y_r|x).
$$

相对偏好 margin：

$$
m=\beta(\Delta_\pi-\Delta_{ref}).
$$

损失：

$$
\mathcal L_{DPO}=-\log\sigma(m).
$$

### 6.1 为什么用 `logsigmoid`

$\sigma(m)$ 可以解释为 chosen 胜过 rejected 的概率。最小化负对数似然：

- $m$ 越大，$\sigma(m)$ 越接近 1，loss 越小；
- $m=0$ 时，loss 为 $-\log 0.5\approx0.693$；
- $m$ 为负时，loss 增大。

直接使用 `F.logsigmoid(m)` 比先 `sigmoid` 再 `log` 数值更稳定。

### 6.2 Loss 看起来为什么不是“反方向”

代码写成：

```python
loss = -F.logsigmoid(beta * (diff_policy - diff_reference))
```

虽然 `diff_policy - diff_reference` 越大越好，前面的负 log 把“最大化偏好 margin”转换成优化器熟悉的“最小化 loss”。

## 7. 隐式 Reward

Chosen reward：

$$
r_c=\beta(\log\pi_\theta(y_c|x)-\log\pi_{ref}(y_c|x)).
$$

Rejected reward：

$$
r_r=\beta(\log\pi_\theta(y_r|x)-\log\pi_{ref}(y_r|x)).
$$

Reward margin：

$$
r_c-r_r=m.
$$

训练目标是增大 margin。reference 是常量，因此梯度只改变 policy：可以提高 chosen、降低 rejected，或两者同时发生。

> [!important]
> DPO 没有硬性保证最终一定满足 `policy_chosen_logp > policy_rejected_logp`。它训练的是 policy 相对于 reference 的偏好改善。若 reference 原本强烈偏向 rejected，小幅正 margin 后 policy 仍可能绝对偏向 rejected。

## 8. $\beta$ 的作用

$\beta$ 缩放 policy 偏离 reference 的偏好信号：

- 太小：loss 对 margin 不敏感，更新信号弱；
- 太大：容易让 sigmoid 饱和，训练激进并放大噪声；
- 不能脱离数据规模、学习率和实现口径单独判断。

本项目使用 `beta=0.1`。

## 9. 一次 DPO Batch 的成本

朴素实现每个 batch 有四次完整模型前向：

1. policy chosen；
2. policy rejected；
3. reference chosen；
4. reference rejected。

只有 policy 路径反向，但 reference 仍占前向时间和激活/输出内存。序列长度翻倍时，Transformer attention 的成本可能接近平方增长，所以 `max_length` 是 DPO 耗时的核心配置之一。

可优化方向：

- chosen/rejected 拼成一个 concatenated batch，减少调用开销；
- reference 使用 `inference_mode`；
- 预计算固定 reference log-probs；
- 使用混合精度、Flash Attention、gradient checkpointing；
- 过滤过长或无有效回答的样本。

## 10. DPO Adapter 的继承关系

本项目 DPO policy 先加载 SFT adapter，再继续修改同一组 LoRA A/B，最终保存 `ckpt/dpo`：

$$
\text{DPO adapter}=\text{SFT adapter 经过 DPO 后的新状态}.
$$

所以推理时只需要：

```text
base model + DPO adapter
```

不需要先加载 SFT adapter 再叠加 DPO adapter。这里保存的是完整的“当前 A/B 状态”，不是相对于 SFT adapter 的二阶差分。

## 11. DPO 评估

不能只看训练 loss。至少记录：

| 指标 | 含义 |
|---|---|
| reward margin mean/median | 相对 reference 的平均偏好变化 |
| chosen win rate | margin 大于 0 的样本比例 |
| margin 分布 | 是否只有少数极端样本拉高平均值 |
| held-out preference set | 排除训练对记忆 |
| generation comparison | 真实回答风格和质量是否改善 |
| 能力/安全回归 | 偏好优化是否损伤原能力 |

SFT 与自己作 reference 时 margin 恒为 0，所以本项目把平局计为 `0.5`，baseline win rate 是 50%。DPO 结果为 `54.69%`，只是轻微正信号。

## 12. DPO 常见误区

- 把 DPO 和 LoRA 当成同一类概念；
- policy 从 base 开始，而 reference 从 SFT 开始；
- reference 没有 `eval()`，导致 dropout 引入随机基线；
- prompt token 没有 mask，偏好 loss 被公共 prompt 主导；
- chosen/rejected 使用不同模板或截断方式；
- 对 logits 先 softmax，再取 log，造成数值不稳定；
- 忘记 causal shift；
- 只看 mean margin，不看 win rate 与分布；
- 认为 win rate 会自动参与训练。它只是评估指标，真正反向的是 DPO loss。

## 13. 自测

1. 为什么 reference 必须与 policy 从同一个 SFT 状态开始？
2. `log_softmax(dim=-1)` 的 `-1` 表示什么？
3. 为什么回答序列概率使用 token log-prob 的和，而不是概率直接相乘？
4. DPO 是否保证 policy 绝对更喜欢 chosen？
5. 为什么 DPO 推理只加载最终 DPO adapter 即可？

> [!answer]- 参考答案
> 1. DPO 测量的是相对固定 SFT 基线的偏好变化；起点不同会混入无关模型差异。
> 2. 对每个时间位置的词表 $V$ 维做归一化。
> 3. 条件概率乘积取 log 后变成求和，更稳定，也便于 mask。
> 4. 不保证；它鼓励相对 reference 的 chosen-vs-rejected 差值增大。
> 5. 保存的是从 SFT 初始化后继续训练得到的完整 A/B 状态，不是额外增量层。
