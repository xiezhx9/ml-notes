---
tags:
  - LLM
  - LoRA
  - PEFT
  - PyTorch
aliases:
  - LoRA原理与实现
  - Adapter生命周期
---

# Task 3.1：LoRA 与 Adapter 生命周期

> [!summary]
> LoRA 冻结原权重 $W$，只训练低秩更新 $\Delta W=\frac{\alpha}{r}BA$。它降低的是可训练参数、梯度、优化器状态和保存成本；基础模型仍参与前向与反向，因此计算量不会按参数比例下降。

返回总览：[[Task3-SFT-DPO学习索引]]

## 1. 从全量微调到低秩更新

全量 Linear：

$$
y=xW^T+b.
$$

微调后：

$$
W'=W+\Delta W.
$$

LoRA 假设有效更新近似低秩：

$$
\Delta W=\frac{\alpha}{r}BA,
$$

其中：

$$
A\in\mathbb R^{r\times d_{in}},\qquad
B\in\mathbb R^{d_{out}\times r}.
$$

对 batch 输入 $X\in\mathbb R^{B\times T\times d_{in}}$：

$$
Y=XW^T+b+\frac{\alpha}{r}(XA^T)B^T.
$$

这正对应 PyTorch：

```python
base_output = base_layer(x)
lora_output = B(A(dropout(x)))
output = base_output + alpha / r * lora_output
```

## 2. 为什么可以写成两个 `nn.Linear`

若：

```python
A = nn.Linear(in_features, r, bias=False)
B = nn.Linear(r, out_features, bias=False)
```

那么：

$$
B(A(x))=(xA^T)B^T=x(BA)^T.
$$

因此模块写法和矩阵公式完全等价。公式中的转置来自 `nn.Linear` 把权重保存为
`[out_features, in_features]`，不是 LoRA 多做了一次特殊运算。

### 2.1 A、B 必须使用 `bias=False` 吗

数学上并非强制，但标准 LoRA 通常让 A、B 都不带 bias：

```python
A = nn.Linear(in_features, r, bias=False)
B = nn.Linear(r, out_features, bias=False)
```

这样 LoRA 分支严格满足：

$$
B(A(x))=x(BA)^T,
$$

因此可以清楚地解释为只对原权重增加低秩更新 $\Delta W=BA$。

如果 A、B 分别带有 bias $a$、$c$：

$$
A(x)=xA^T+a,
\qquad
B(h)=hB^T+c,
$$

则复合结果变成：

$$
B(A(x))=x(BA)^T+aB^T+c.
$$

后两项是额外的输出偏置，不再是纯粹的低秩权重更新。尤其当 B 的 bias $c$
初始非零时，即使 `B.weight` 全零，刚注入 LoRA 也会立刻改变 base 模型的输出。

工程上应区分两件事：

- LoRA 新增的 A/B 分支通常都使用 `bias=False`；
- 原始 `base_layer.bias` 仍然存在，通常随 base 一起冻结，也可以在特定微调策略中单独放开训练。

因此，无 bias 不是 LoRA 能工作的硬性条件，而是为了保持零影响初始化、减少参数，
并维持“只学习低秩 $\Delta W$”的清晰定义。

## 3. 参数量为何显著下降

全量权重参数量：

$$
N_{full}=d_{out}d_{in}.
$$

LoRA 参数量：

$$
N_{LoRA}=r(d_{in}+d_{out}).
$$

当 $r\ll d_{in},d_{out}$ 时，差距很大。本项目只替换 `q_proj`、`v_proj`，最终：

```text
LoRA tensors: 96
Trainable parameters: 540,672
Total parameters after injection: 494,573,440
Trainable ratio: 0.109%
```

> [!note]
> 注入 LoRA 后“总参数”会略高于 base，因为模型同时保留 frozen $W$ 和新建的 A/B。

## 4. 初始化为何是 A 随机、B 全零

本项目：

```python
nn.init.kaiming_normal_(A.weight)
nn.init.zeros_(B.weight)
```

初始时：

$$
BA=0\quad\Rightarrow\quad\Delta W=0.
$$

所以注入前后模型输出完全一致，不会一开始破坏 base 能力。

但梯度仍能流动。第一步时：

$$
\frac{\partial L}{\partial B}\propto
\frac{\partial L}{\partial Y}(XA^T),
$$

随机 A 让 B 获得非零梯度。由于 B 初始为零，A 的第一步梯度可能为零；B 更新后，A 随后开始学习。这是有意设计，不是训练失败。

## 5. Dropout 放在哪里

LoRA 常写为：

$$
\Delta y=\frac{\alpha}{r}B(A(\mathrm{Dropout}(x))).
$$

它只扰动 LoRA 分支的输入，base 分支保持确定：

```python
base_layer(x) + scaling * B(A(dropout(x)))
```

这与 Transformer attention dropout 不同：

- LoRA dropout 正则化“增量适配路径”；
- attention dropout 正则化 token 之间的注意力连接；
- residual dropout 正则化子层输出。

Dropout 位置会改变随机变量作用对象，不能因为名字相同就随意移动。

## 6. 注入算法

核心流程：

1. 先把原模型所有参数 `requires_grad_(False)`。
2. 遍历 `model.named_modules()`。
3. 匹配叶子名，例如 `q_proj`、`v_proj`。
4. 找到父模块，用 `setattr(parent, leaf, LoRALinear(old_linear))` 替换。
5. optimizer 只接收 `requires_grad=True` 的 A/B。

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable, lr=learning_rate)
```

### 6.1 为什么匹配叶子名

Qwen 的完整路径可能是：

```text
model.layers.0.self_attn.q_proj
model.layers.1.self_attn.q_proj
...
```

匹配最后的 `q_proj` 可以一次覆盖每个 block。需要注意：若别的子结构也有同名 Linear，它也会被替换；生产代码通常同时约束模块类型或完整路径。

### 6.2 遍历时替换的风险

修改模块树时应避免重复包裹：

- 注入前确认目标还是 `nn.Linear`；
- 合并时先 `list(iter_lora_modules(model))` 固化遍历结果；
- 不要对已经注入的模型再次调用 `inject_lora`。

## 7. 保存与恢复

紧凑 adapter 只保存：

```text
lora_dict.pt
metadata.json
```

权重键类似：

```text
model.layers.0.self_attn.q_proj.A.weight
model.layers.0.self_attn.q_proj.B.weight
```

metadata 至少包含：

```json
{
  "base_model": "models/Qwen2.5-0.5B",
  "target_modules": ["q_proj", "v_proj"],
  "r": 8,
  "alpha": 16.0,
  "dropout": 0.0
}
```

恢复顺序必须是：

```mermaid
flowchart LR
    A[加载同一 base] --> B[读取 metadata]
    B --> C[按相同配置注入 LoRA]
    C --> D[校验 expected keys]
    D --> E[复制 A/B 权重]
```

> [!warning] 常见误区
> Adapter 不是完整模型。只有 adapter 文件而没有 base checkpoint，无法恢复 SFT/DPO 模型；只创建空模型但模块名、rank 或 target 不一致，也无法严格加载。

## 8. 合并 LoRA

推理前可以把增量写回原 Linear：

$$
W_{merged}=W+\frac{\alpha}{r}BA.
$$

```python
merged.weight.add_(delta_weight)
```

合并后的优点：

- 推理不再执行 A/B 两个额外 Linear；
- 模型结构恢复成普通 Linear；
- 更容易交给不认识自定义 LoRA 类的推理框架。

代价：

- 完整 checkpoint 重新变大；
- 若不保留原 base 或 adapter，难以撤销；
- 多 adapter 动态切换不再方便。

## 9. LoRA 的局限

- 低秩假设不保证适合所有任务；
- rank 增大只提高容量，不保证验证质量；
- base 激活仍需保存以完成反向传播；
- 只改 Q/V 可能不足以学习复杂领域知识；
- adapter 与 base 版本强绑定；
- 小 adapter 也可能造成遗忘或安全回归。

## 10. 自测

1. 为什么 B 全零不会让 LoRA 永远学不动？

    1. B的梯度更新和A有关，A为随机矩阵，第一次更新后，B就有值了

2. `B(A(x))` 为什么等价于一次权重更新 $\Delta W=BA$？

    1. 矩阵乘法的性质，结合律

3. LoRA 参数减少 99.9%，训练时间为什么只快约 1.89 倍？

    1. LoRA训练时，所有参数都需参与计算，只是只有部分参数需要保留梯度而已

4. 为什么加载 adapter 前必须先注入完全相同的模块结构？

    1. adapter保存对应的A，B矩阵参数，需要有正确的基模base model和对应lora结构meta data才能恢复，模块名和shape需要一致

5. `alpha` 固定时提高 rank，缩放系数会怎样变化？

    1. 变小缩放系数是 $\alpha / r$
6. 为什么 LoRA 的 A/B 通常设置 `bias=False`？原始 Linear 的 bias 会被删除吗？
	1. 为了保持lora运算的简洁 $\Delta W = BA$
	2. 原始Linear的bias会保留，一起冻结

> [!answer]- 参考答案
> 1. A 随机，B 第一轮能从 `XA^T` 获得梯度；B 非零后 A 也开始获得梯度。
> 2. 两个无 bias Linear 的复合为 `xA^TB^T=x(BA)^T`。
> 3. frozen base 仍执行前向和反向；LoRA 主要节省参数梯度、优化器状态和保存成本。
> 4. adapter state dict 只含 A/B，需要一致的模块名和 shape 才能找到并复制。
> 5. `alpha/r` 变小；所以固定 alpha 的 rank 消融同时改变容量和更新缩放。
> 6. 无 bias 时 A/B 的复合严格等价于低秩权重更新 $BA$；加入 bias 会额外产生输出偏置项。原始 Linear 的 bias 不会被删除，通常只是随 base 一起冻结。
