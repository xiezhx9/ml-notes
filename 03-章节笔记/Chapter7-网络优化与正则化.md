# Chapter 7：网络优化与正则化

> [!summary] 核心问题
> 神经网络训练中，常见问题包括：梯度下降太慢或震荡、不同方向尺度差异大、激活值/梯度爆炸或消失、模型过拟合，以及训练后期学习率过大。Chapter 7 的各种方法，分别针对这些问题。

## 1. 方法总览

| 训练问题            | 主要方法                    | 核心作用         |
| --------------- | ----------------------- | ------------ |
| 只能看到部分数据、全量计算太贵 | Mini-batch              | 用小批量样本近似全量梯度 |
| 梯度方向震荡、收敛慢      | Momentum                | 累积历史方向，减少震荡  |
| 不同参数梯度尺度不同      | RMSprop / Adam          | 为不同参数自适应调整步长 |
| 激活值或梯度爆炸、消失     | Xavier / Kaiming 初始化    | 控制网络初始信号尺度   |
| 隐藏层激活尺度不断漂移     | BatchNorm               | 稳定中间激活值和梯度   |
| 模型过度拟合训练集       | L2、Weight Decay、Dropout | 限制模型复杂度，提高泛化 |
| 训练后期学习率过大       | Learning-rate scheduler | 逐步减小学习率，细化收敛 |

可以把它们分成四类：

- **优化器**：决定参数每一步往哪里走、走多远。
- **初始化**：决定训练开始时参数和信号处于什么尺度。
- **归一化**：稳定网络中间的激活值。
- **正则化**：限制模型复杂度，减少过拟合。

## 2. Mini-batch

### 解决的问题

如果每次使用全部训练集计算梯度：

- 内存消耗大。
- 一次更新计算量大。
- 参数更新频率低。

Mini-batch 每次只取一小批样本：

```python
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for X_batch, y_batch in loader:
    loss = loss_fn(model(X_batch), y_batch)
    loss.backward()
    optimizer.step()
```

`batch_size=64` 就表示每次用 64 条样本估计梯度。

### 数学理解

全量梯度是：

$$
g=\frac{1}{N}\sum_{i=1}^{N}\nabla_\theta \ell_i(\theta)
$$

Mini-batch 梯度是：

$$
\hat g=\frac{1}{B}\sum_{i\in\mathcal B}\nabla_\theta \ell_i(\theta)
$$

随机抽取 batch 时：

$$
\mathbb E[\hat g]\approx g
$$

因此它是全量梯度的随机估计。batch 太小，噪声大；batch 太大，计算稳定但更新次数少。常见起点是 32、64、128 或 256。

### 优势

- 适合 GPU 并行计算。
- 内存占用低于全量训练。
- 更新更频繁。
- 适量的梯度噪声有时有助于泛化。

## 3. 优化器

### 3.1 SGD

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

**解决的问题**：用梯度直接降低损失。

**优势**：简单、内存开销低，调好学习率后泛化能力通常不错。

**不足**：容易在狭长谷底中震荡，对学习率敏感，收敛可能较慢。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### 3.2 SGD + Momentum

$$
v_t=\beta v_{t-1}+g_t
$$

$$
\theta_{t+1}=\theta_t-\eta v_t
$$

**解决的问题**：普通 SGD 在某些方向来回震荡、在平坦方向前进缓慢。

**直觉**：把历史梯度累积成速度；长期同向时加速，方向反复变化时相互抵消。

**优势**：减少震荡、加速沿稳定方向前进，仍然保留 SGD 较好的泛化特性。

```python
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9
)
```

### 3.3 RMSprop

$$
s_t=\rho s_{t-1}+(1-\rho)g_t^2
$$

$$
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}
$$

**解决的问题**：不同参数或不同方向的梯度尺度差异很大。

**直觉**：梯度长期较大的方向自动缩小步长，梯度较小的方向相对增大步长。

**优势**：对梯度尺度差异和非平稳目标更友好，曾经常用于 RNN。

**不足**：仍需调学习率；在现代 Transformer/LLM 中通常不是首选。

### 3.4 Adam

Adam 同时使用 Momentum 和 RMSprop：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

经过偏置修正后：

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

**解决的问题**：同时处理方向震荡和不同参数的梯度尺度差异。

**优势**：收敛通常较快，对初始学习率相对不敏感，是复杂模型和快速实验的常用默认选择。

**不足**：需要保存一阶矩、二阶矩，额外内存较大；最终泛化不一定优于 SGD；特殊损失面上也可能不稳定。

### 3.5 AdamW

AdamW 将 Adam 的梯度更新和 Weight Decay 解耦：

$$
\theta\leftarrow
\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}
-\eta\lambda\theta
$$

**解决的问题**：Adam 中直接加入 L2 项时，自适应梯度缩放会影响正则化效果。

**优势**：权重衰减更可控，通常是 Transformer/LLM 的常用优化器。

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=0.01
)
```

## 4. 参数初始化

### 解决的问题

如果初始权重太大：

- 激活值可能层层变大。
- 梯度可能爆炸。

如果初始权重太小：

- 激活值可能层层变小。
- 梯度可能消失。

对于：

$$
z=Wx
$$

如果输入方差为 $\mathrm{Var}(x)$，权重方差为 $\mathrm{Var}(w)$，则近似有：

$$
\mathrm{Var}(z)\approx\mathrm{fan\_in}\cdot\mathrm{Var}(w)\cdot\mathrm{Var}(x)
$$

因此初始化的目标是让信号通过多层网络时，方差不要快速变大或变小。

### Xavier 初始化

适合 `tanh`、`sigmoid` 等近似对称的激活函数：

$$
\mathrm{Var}(w)\approx\frac{2}{\mathrm{fan\_in}+\mathrm{fan\_out}}
$$

### Kaiming 初始化

适合 ReLU：

$$
\mathrm{Var}(w)\approx\frac{2}{\mathrm{fan\_in}}
$$

ReLU 会把一部分负值变成 0，Kaiming 用更大的初始方差补偿这一影响。

```python
nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
```

**优势**：让深层网络在训练开始时保持较稳定的激活和梯度尺度。

**注意**：初始化只控制起点，训练开始后权重会继续变化，不能保证每一层永远保持原来的标准差。

## 5. BatchNorm

### 解决的问题

隐藏层的权重不断更新，会导致后续层收到的激活值尺度不断变化。如果激活值忽大忽小，可能导致梯度不稳定、学习率难以设置。

对 batch 中某个特征做：

$$
\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

再进行可学习的缩放和平移：

$$
y=\gamma\hat x+\beta
$$

### 优势

- 稳定每个特征的激活尺度。
- 通常允许使用更大的学习率。
- 让训练曲线更平滑、收敛更快。
- batch 统计量带来少量随机噪声，可能有轻微正则化效果。

### 重要区别

BN 做的是标准化，不是高斯化：

$$
\mathbb E[\hat x]\approx0,\qquad \mathrm{Var}(\hat x)\approx1
$$

但不保证 $\hat x$ 服从高斯分布。

训练和推理的统计量不同：

```python
model.train()  # 使用当前 batch 的统计量，并更新 running stats
model.eval()   # 使用训练期间累计的 running mean/var
```

BN 在 CNN 中很常见，但现代 LLM 通常使用 LayerNorm 或 RMSNorm，因为 BN 依赖 batch 统计量，不适合自回归生成和变化的序列长度。

## 6. 正则化

### 6.1 L2 正则与 Weight Decay

**解决的问题**：模型权重过大、模型过于复杂、训练集拟合噪声，即过拟合。

目标函数：

$$
J(\theta)=L(\theta)+\frac{\lambda}{2}\|w\|_2^2
$$

梯度下降时：

$$
w\leftarrow(1-\eta\lambda)w-\eta\nabla L(w)
$$

**优势**：限制权重规模，让模型倾向于更平滑、更简单的解；通常比单纯追求最低训练损失有更好的测试表现。

对于 SGD，L2 正则和 weight decay 等价；对于 Adam，推荐使用 AdamW 做解耦的权重衰减。

### 6.2 Dropout

**解决的问题**：神经元之间过度协同，模型依赖某些固定路径，导致过拟合。

训练时：

$$
\tilde h_i=\frac{m_i h_i}{1-p},\qquad m_i\sim\mathrm{Bernoulli}(1-p)
$$

每次随机丢弃一部分激活值，相当于训练不同的随机子网络。

**优势**：

- 减少神经元之间的过度依赖。
- 提高对部分特征缺失的鲁棒性。
- 在线性平方损失的简单情形中，期望效果类似加权 L2 正则。

```python
nn.Dropout(p=0.2)
```

推理时关闭：

```python
model.eval()
```

Dropout 不是永久删除神经元，而是训练时随机将激活值置零。

## 7. 学习率调度器

### 解决的问题

学习率太大时，后期可能在最优点附近震荡；学习率太小时，前期收敛太慢。因此常见策略是：

```text
前期学习率较大：快速找到较好区域
后期学习率较小：在最优点附近精细调整
```

### StepLR：阶梯式衰减

$$
\eta_t=\eta_0\gamma^{\lfloor t/s\rfloor}
$$

每隔 `step_size` 个 epoch 乘一次 `gamma`。

**优势**：简单、可控；**不足**：学习率突然跳变，需要提前指定节点。

### ExponentialLR：指数衰减

$$
\eta_t=\eta_0\gamma^t
$$

每个 epoch 都按比例衰减。

**优势**：平滑；**不足**：`gamma` 不好选，可能衰减过快。

### CosineAnnealingLR：余弦衰减

$$
\eta_t=\eta_{\min}+\frac12(\eta_0-\eta_{\min})
\left(1+\cos\frac{\pi t}{T_{\max}}\right)
$$

**优势**：平滑、后期下降自然，固定训练轮数时通常是很好的默认选择。

### ReduceLROnPlateau：根据验证集指标衰减

当验证损失连续若干轮没有改善时降低学习率：

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

scheduler.step(val_loss)
```

**优势**：根据训练效果动态调整；**不足**：验证指标有噪声时可能误触发。

### 调用顺序

普通 scheduler：

```python
optimizer.step()
scheduler.step()
```

`ReduceLROnPlateau` 需要传入验证指标，通常在验证完成后调用。

## 8. 方法之间的区别

| 方法 | 改变什么 | 主要解决的问题 |
|---|---|---|
| SGD / Adam / AdamW | 参数更新方向和步长 | 梯度下降效率 |
| Mini-batch | 每次参与计算的数据 | 计算成本和梯度估计 |
| Xavier / Kaiming | 初始权重分布 | 初始激活/梯度爆炸或消失 |
| BatchNorm | 中间激活分布 | 激活尺度漂移、训练不稳定 |
| L2 / Weight Decay | 参数大小 | 过拟合、权重过大 |
| Dropout | 激活连接 | 神经元过度协同、过拟合 |
| Scheduler | 学习率随时间的变化 | 前期太慢、后期震荡 |

它们并不是互相替代的关系，而是作用在训练流程的不同位置。

## 9. 实践组合

### CNN 常见组合

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs
)
```

通常配合：

```text
Kaiming 初始化 + Conv-BN-ReLU + SGD Momentum + Weight Decay + Cosine
```

### 普通 MLP

```text
合适初始化 + 可选 BatchNorm + Adam/AdamW + Dropout（过拟合时）
```

### Transformer / LLM

```text
RMSNorm 或 LayerNorm + AdamW + Weight Decay + Warmup + Cosine Decay
```

通常不使用 BatchNorm；Dropout 是否使用取决于模型规模和数据量。

## 10. 最后记忆

1. **优化器**解决“参数怎么走”。
2. **Mini-batch**解决“每次用多少数据估计梯度”。
3. **初始化**解决“训练刚开始时信号是否稳定”。
4. **BN**解决“隐藏激活尺度是否稳定”。
5. **正则化**解决“模型是否过度记忆训练集”。
6. **学习率调度器**解决“训练不同阶段应该走多大步”。

相关笔记：[[优化器与学习率]]、[[梯度下降原理]]、[[结构风险]]
