# Chapter 2：最小二乘、中心化与正则化

## 1. 问题设定

线性回归模型：

$$
\hat y_i=x_i^\top w+b
$$

其中：

| 符号 | 含义 | 形状 |
|---|---|---|
| $N$ | 样本数量 | 标量 |
| $D$ | 每个样本的特征数量 | 标量 |
| $x_i$ | 第 $i$ 个样本的特征向量 | $D\times1$ |
| $X$ | 所有样本组成的特征矩阵 | $N\times D$ |
| $y_i$ | 第 $i$ 个样本的真实标签 | 标量 |
| $y$ | 所有真实标签 | $N\times1$ |
| $w$ | 模型权重 | $D\times1$ |
| $b$ | 偏置 | 标量 |
| $\hat y$ | 模型预测值 | $N\times1$ |
| $r$ | 残差 | $N\times1$ |

## 2. 具体数据

```python
X = torch.tensor([
    [1., 2.],
    [3., 4.],
    [5., 6.],
])

y = torch.tensor([
    [4.],
    [8.],
    [12.],
])
```

这里：

$$
N=3,\qquad D=2
$$

矩阵 $X$ 的每一行是一个样本，每一列是一个特征：

$$
X=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix}
$$

数学上，单个样本通常写成列向量：

$$
x_i=\begin{bmatrix}x_{i1}\\x_{i2}\end{bmatrix}
$$

放入 $X$ 时，使用它的转置作为一行：

$$
X=\begin{bmatrix}x_1^\top\\x_2^\top\\x_3^\top\end{bmatrix}
$$

## 3. 模型预测

设：

$$
w=\begin{bmatrix}1\\2\end{bmatrix},\qquad b=0
$$

批量预测公式：

$$
\hat y=Xw+b
$$

矩阵计算：

$$
Xw=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix}
\begin{bmatrix}1\\2\end{bmatrix}
=
\begin{bmatrix}5\\11\\17\end{bmatrix}
$$

所以：

$$
\hat y=\begin{bmatrix}5\\11\\17\end{bmatrix}
$$

对应每个样本：

$$
\hat y_1=1\times1+2\times2=5
$$

$$
\hat y_2=3\times1+4\times2=11
$$

## 4. 残差

残差是真实值与预测值之间的差异。这里约定：

$$
r=\hat y-y
$$

本例中：

$$
r=
\begin{bmatrix}5\\11\\17\end{bmatrix}
-
\begin{bmatrix}4\\8\\12\end{bmatrix}
=
\begin{bmatrix}1\\3\\5\end{bmatrix}
$$

在代码中：

```python
residual = X @ w + b - y
```

## 5. 最小二乘损失

最小二乘最小化残差的平方和：

$$
J(w,b)=\frac12\sum_{i=1}^{N}(\hat y_i-y_i)^2
$$

或者使用均方误差：

$$
J(w,b)=\frac1N\sum_{i=1}^{N}(\hat y_i-y_i)^2
$$

本例残差为 $[1,3,5]$，所以平方误差为：

$$
[1^2,3^2,5^2]=[1,9,25]
$$

平方误差和：

$$
1+9+25=35
$$

均方误差：

$$
\operatorname{MSE}=\frac{35}{3}\approx11.67
$$

前面的 $1/2$ 只是为了求导时抵消平方产生的系数 2，不改变最优参数。

## 6. 带偏置的最小二乘求解

目标函数：

$$
J(w,b)=\frac12\|Xw+b\mathbf1-y\|^2
$$

其中 $\mathbf1$ 是全 1 列向量：

$$
\mathbf1=\begin{bmatrix}1\\1\\\vdots\\1\end{bmatrix}\in\mathbb R^{N\times1}
$$

定义残差：

$$
r=Xw+b\mathbf1-y
$$

对权重求梯度：

$$
\frac{\partial J}{\partial w}=X^\top r
$$

对偏置求梯度：

$$
\frac{\partial J}{\partial b}=\mathbf1^\top r
$$

令偏置梯度为 0：

$$
\mathbf1^\top(Xw+b\mathbf1-y)=0
$$

整理得到：

$$
\boxed{b=\bar y-\bar x^\top w}
$$

这里：

$$
\bar x=\frac1N\sum_{i=1}^{N}x_i
$$

是每个特征的平均值，形状为 $D\times1$；

$$
\bar y=\frac1N\sum_{i=1}^{N}y_i
$$

是标签的平均值，通常是标量。

## 7. 均值向量的计算

对于：

$$
X=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}
$$

按列计算均值：

$$
\bar x=\begin{bmatrix}
(1+3+5)/3\\
(2+4+6)/3
\end{bmatrix}
=\begin{bmatrix}3\\4\end{bmatrix}
$$

PyTorch：

```python
x_bar = X.mean(dim=0, keepdim=True)
```

结果形状是 `[1, 2]`：

```text
tensor([[3., 4.]])
```

因为代码中的 `x_bar` 已经是行向量，所以偏置写成：

```python
b = y_bar - x_bar @ w
```

数学中把均值写成列向量时，则写成：

$$
b=\bar y-\bar x^\top w
$$

两种写法完全等价。

## 8. 中心化

中心化只做减均值，不除以标准差：

$$
X_c=X-\bar x^\top
$$

在代码中：

```python
X_centered = X - x_bar
```

本例：

$$
X_c=
\begin{bmatrix}
1-3&2-4\\
3-3&4-4\\
5-3&6-4
\end{bmatrix}
=
\begin{bmatrix}
-2&-2\\
0&0\\
2&2
\end{bmatrix}
$$

标签也中心化：

$$
y_c=y-\bar y
$$

如果 $y=[4,8,12]^\top$，则：

$$
\bar y=8,\qquad y_c=\begin{bmatrix}-4\\0\\4\end{bmatrix}
$$

中心化之后，每一列和标签的平均值都是 0。

## 9. 中心化后的最小二乘解

将：

$$
b=\bar y-\bar x^\top w
$$

代回原模型，残差变成：

$$
(x_i-\bar x)^\top w-(y_i-\bar y)
$$

于是只需要求解：

$$
J(w)=\frac12\|X_cw-y_c\|^2
$$

对 $w$ 求导并令 0：

$$
X_c^\top(X_cw-y_c)=0
$$

得到：

$$
\boxed{
w=(X_c^\top X_c)^{-1}X_c^\top y_c
}
$$

最后恢复偏置：

$$
\boxed{
b=\bar y-\bar x^\top w
}
$$

PyTorch 写法：

```python
x_bar = X.mean(dim=0, keepdim=True)
y_bar = y.mean()

X_centered = X - x_bar
y_centered = y - y_bar

w = torch.linalg.lstsq(X_centered, y_centered).solution
b = y_bar - x_bar @ w
```

## 10. 增广矩阵方法

也可以把偏置看成一个额外权重。

给 $X$ 增加一列 1：

$$
\tilde X=[X\ \mathbf1]
$$

把参数合并：

$$
\tilde w=\begin{bmatrix}w\\b\end{bmatrix}
$$

于是：

$$
Xw+b\mathbf1=\tilde X\tilde w
$$

最小二乘解变成：

$$
\boxed{
\tilde w=(\tilde X^\top\tilde X)^{-1}\tilde X^\top y
}
$$

代码：

```python
ones = torch.ones(X.shape[0], 1)
X_aug = torch.cat([X, ones], dim=1)
theta = torch.linalg.lstsq(X_aug, y).solution

w = theta[:-1]
b = theta[-1]
```

中心化方法和增广矩阵方法得到的是同一个线性回归问题的解。

## 11. L2 正则化

没有正则化时：

$$
J(w,b)=\frac12\|Xw+b\mathbf1-y\|^2
$$

加入 L2 正则后：

$$
\boxed{
J_\lambda(w,b)=
\frac12\|Xw+b\mathbf1-y\|^2
+\frac\lambda2\|w\|^2
}
$$

其中：

$$
\|w\|^2=w_1^2+w_2^2+\cdots+w_D^2
$$

$\lambda$ 是正则强度：

| $\lambda$ | 含义 |
|---|---|
| $0$ | 不使用正则化 |
| 较小 | 轻微压缩权重 |
| 较大 | 强烈压缩权重，可能欠拟合 |

通常不惩罚偏置 $b$，因为偏置负责整体平移，不代表模型的复杂度。

## 12. L2 正则化的闭式解

对中心化数据：

$$
J_\lambda(w)=
\frac12\|X_cw-y_c\|^2+ \frac{\lambda}{2}\|w\|^2

$$

对 $w$ 求导：

$$
\frac{\partial J_\lambda}{\partial w}
=X_c^\top(X_cw-y_c)+\lambda w
$$

令梯度为 0：

$$
X_c^\top X_cw-X_c^\top y_c+\lambda w=0
$$

整理：

$$
(X_c^\top X_c+\lambda I)w=X_c^\top y_c
$$

所以：

$$
\boxed{
w=(X_c^\top X_c+\lambda I)^{-1}X_c^\top y_c
}
$$

代码：

```python
D = X.shape[1]
A = X_centered.T @ X_centered
A = A + reg_lambda * torch.eye(D)
rhs = X_centered.T @ y_centered
w = torch.linalg.solve(A, rhs)
b = y_bar - x_bar @ w
```

## 13. L2 正则化的具体例子

考虑一维数据：

$$
x=\begin{bmatrix}1\\2\\3\end{bmatrix},
\qquad
y=\begin{bmatrix}2\\4\\6\end{bmatrix}
$$

均值：

$$
\bar x=2,\qquad \bar y=4
$$

中心化：

$$
x_c=\begin{bmatrix}-1\\0\\1\end{bmatrix},
\qquad
y_c=\begin{bmatrix}-2\\0\\2\end{bmatrix}
$$

有：

$$
x_c^\top x_c=2,
\qquad
x_c^\top y_c=4
$$

### 不加正则

$$
w=\frac{4}{2}=2
$$

$$
b=4-2\times2=0
$$

模型：

$$
\hat y=2x
$$

### 加入 $\lambda=2$ 的正则

$$
w=\frac{4}{2+2}=1
$$

$$
b=4-2\times1=2
$$

模型变成：

$$
\hat y=x+2
$$

权重从 2 被压缩到 1，这就是 L2 正则的作用。

## 14. 维度检查表

设 $X\in\mathbb R^{N\times D}$：

| 表达式 | 形状 | 作用 |
|---|---|---|
| $Xw$ | $N\times1$ | 所有样本的线性预测 |
| $X^\top$ | $D\times N$ | 把样本贡献聚合到特征维度 |
| $X^\top X$ | $D\times D$ | 特征之间的相关矩阵 |
| $X^\top y$ | $D\times1$ | 特征与标签的相关量 |
| $X_c^\top X_c+\lambda I$ | $D\times D$ | 正则化后的线性方程左侧 |
| $X_c^\top y_c$ | $D\times1$ | 正则化方程右侧 |
| $\bar x^\top w$ | 标量 | 平均输入对应的预测值 |
| $\bar y-\bar x^\top w$ | 标量 | 偏置 $b$ |

## 15. 一句话总结

$$
\boxed{
\text{中心化求 }w，\quad
\text{均值恢复 }b，\quad
\text{正则化压缩 }w
}
$$

完整流程：

```text
X, y
  ↓
按列计算 x_bar，计算 y_bar
  ↓
X_centered = X - x_bar
y_centered = y - y_bar
  ↓
w = solve(X_centered.T @ X_centered + λI,
          X_centered.T @ y_centered)
  ↓
b = y_bar - x_bar @ w
  ↓
y_hat = X @ w + b
```
