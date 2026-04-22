# nn0.py — 自動微分引擎

提供純 Python 實現的自動微分（Autograd）與優化器，是構建神經網路的基礎元件。

## Value 類 — 自動微分節點

### 核心思想

使用鏈式法則（Chain Rule）自動計算梯度。每個 `Value` 節點記錄：
- `data`：數值
- `grad`：梯度（預設為 0）
- `_children`：輸入節點（依賴項）
- `_local_grads`：局部梯度（對每個 child 的偏導數）

### 支援的運算

| 運算 | 局部梯度 |
|------|----------|
| `a + b` | (1, 1) |
| `a * b` | (b, a) |
| `a ** n` | n * a^(n-1) |
| `log(a)` | 1 / a |
| `exp(a)` | exp(a) |
| `relu(a)` | 1 if a > 0 else 0 |

### 反向傳播（Backpropagation）

```
backward():
    1. 建構拓撲排序（topo），確保子節點先被處理
    2. 從輸出梯度 1 開始
    3. 逆序遍歷 topo，將梯度傳給每個 child:
       child.grad += local_grad * parent.grad
```

## Adam 優化器

自適應學習率優化演算法，結合動量與 RMSProp：

```
m = β1 * m + (1-β1) * grad      # 一階矩估計（動量）
v = β2 * v + (1-β2) * grad²     # 二階矩估計（方差）
m_hat = m / (1 - β1^t)          # 偏差修正
v_hat = v / (1 - β2^t)          # 偏差修正
p -= lr * m_hat / (√v_hat + ε) # 參數更新
```

預設參數：β1=0.85, β2=0.99，學習率線性衰減。

## 核心函式

### linear(x, W) — 矩陣乘法
```python
y = W @ x
```
實現：`sum(wi * xi for wi, xi in zip(wo, x))`

### softmax(logits) — 數值穩定 Softmax
使用最大值平移技巧防止溢位：
```
e^{x_i} / Σe^{x_j} = e^{x_i-M} / Σe^{x_j-M}  (M = max(logits))
```

### rmsnorm(x) — RMS Normalization
```python
scale = (Σx_i² / n + ε)^-0.5
output = x * scale
```
比 LayerNorm 簡單（無需計算均值）。

### cross_entropy(logits, target) — 數值穩定 Cross-Entropy
使用 Log-Sum-Exp 技巧：
```
Loss = log(Σe^{x_i-M}) - (x_target - M)
```
避免先算 Softmax 可能導致的 log(0) 錯誤。

### gd() — 單步梯度下降
1. 對 `block_size` 個位置做前向傳播
2. 計算平均 loss
3. backward() 計算梯度
4. Adam 更新參數（學習率線性衰減）