# RSVPStream

RSVPStream is a real-time EEG classification system that processes brain signals to detect responses to Rapid Serial Visual Presentation (RSVP) stimuli. This repository provides the necessary code and tools to deploy, train, and test the RSVP-based brain-computer interface on EEG data.

## Features
- Real-time RSVP signal processing.
- EEG data preprocessing and classification.
- Support for various machine learning models.
- Modular and scalable architecture.

## Prerequisites
Before getting started, ensure the following dependencies are installed:

- Python 3.8 or higher
- NumPy
- SciPy
- scikit-learn
- MNE (for EEG data processing)
- TensorFlow or PyTorch (depending on the model used)

You can install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```


Installation
Clone the repository:

```bash
git clone --recurse-submodules https://github.com/KylinGR/RSVPStream.git
```

Navigate to the project directory:


```bash
cd RSVPStream
```

If you cloned the repository directly, you can use the following command to update the submodules:


```bash
git submodule update --init --recursive

```

Install the required Python packages:


```bash
pip install -r requirements.txt
```




# 模型
## 模型推理

### 1. **归一化全局数据（Batch Normalization of Global Data）**

**步骤描述：**

- 模型在训练过程中对全局数据进行了归一化，因此在推理过程中，我们也需要对测试数据进行相同的归一化处理，以确保数据分布与训练时一致。
- 归一化是通过减去均值并除以标准差来完成的。全局数据的均值和方差在训练时就已经计算好并存储了，我们只需将它们应用到推理数据上。

**需要做的事：**

- 使用训练时保存的均值（`M_global`）和方差（`Sigma_global`），对新的全局数据进行归一化。

**伪代码：**

```python
PYTHON

X_global_BN = self.batchnormalize_global(Tset_global, self.M_global[:, :, idx_group], self.Sigma_global[:, :, idx_group])

```

**解释：**

- `Tset_global` 是预处理后的全局数据。
- `self.M_global` 和 `self.Sigma_global` 是在训练时计算并保存下来的全局数据的均值和方差。
- 调用的 `batchnormalize_global()` 函数会使用这些均值和方差对数据进行标准化处理。

---

### 2. **归一化局部数据（Batch Normalization of Local Data）**

**步骤描述：**

- 类似于全局数据，局部数据也需要进行归一化，以确保输入到模型中的数据与训练时保持一致。
- 对每个局部模型，我们都需要分别对数据块进行归一化。

**需要做的事：**

- 使用训练时保存的局部数据的均值（`M_local`）和方差（`Sigma_local`），对局部数据进行归一化。
- 由于局部数据有多个块（`N_local_model`），我们需要对每个块进行单独的归一化。

**伪代码：**

```python
PYTHON

X_minibatch_BN = torch.zeros((Ns, self.T_local, self.N_local_model)).float().cuda(0)
for idx_local in range(self.N_local_model):
    X_minibatch_BN[:, :, idx_local] = self.batchnormalize(
        Tset[:, :, idx_local],
        self.M_local[idx_local, :, idx_group],
        self.Sigma_local[idx_local, :, idx_group]
    )

```

**解释：**

- `Tset` 是预处理后的局部数据。
- `self.M_local` 和 `self.Sigma_local` 是训练过程中计算的局部数据的均值和方差。
- `batchnormalize()` 函数对局部数据每个块进行归一化。

---

### 3. **全局模型推理（Global Model Inference）**

**步骤描述：**

- 输入的数据首先通过 **全局模型**（`model_GSTF`），该模型会根据输入的全局数据来生成初步的预测结果。
- 全局模型的输出是一个概率值（sigmoid函数的输出），表示输入属于某一类的可能性。

**需要做的事：**

- 使用全局数据经过全局模型进行推理，得到初步的预测结果。

**伪代码：**

```python
PYTHON

gstf = model_onegroup[0]
s, h = gstf(X_global_BN.permute(2, 1, 0))
s = s.detach()  # 将结果从计算图中分离出来以避免之后的梯度计算
h = h.detach()

```

**解释：**

- `gstf` 是全局模型（`model_GSTF`）。
- `X_global_BN` 是归一化后的全局数据。
- `s` 是模型的输出，表示全局预测的概率值。
- `h` 是模型的中间结果，后续还会使用它与局部模型的结果相结合。

---

### 4. **局部模型推理（Local Model Inference）**

**步骤描述：**

- 在全局模型的基础上，输入的数据还需要通过多个 **局部模型** 进行推理。
- 局部模型会根据各自的输入块给出一个局部的预测结果，这个结果会与全局模型的结果相结合。
- 每个局部模型的输出会加权后与全局模型的结果相加，形成最终的预测结果。

**需要做的事：**

- 使用局部数据经过多个局部模型进行推理，并与全局模型的结果相结合。

**伪代码：**

```python
PYTHON

if N_model > 1:
    h = self.gstf_weight * h  # 初始的全局结果加上权重

    for idx_model in range(1, N_model):
        local_model = model_onegroup[idx_model]
        f = local_model(X_minibatch_BN[:, :, idx_model - 1]).detach()
        h = h + self.lr_model[idx_model - 1, 0] * f  # 局部模型的结果与全局结果相加

    s = torch.sigmoid(h)  # 最终应用sigmoid函数得到概率

```

**解释：**

- `model_onegroup` 是一个模型组，其中第一个是全局模型，后面是局部模型。
- `local_model` 是每个局部模型。
- `X_minibatch_BN` 是归一化后的局部数据。
- `f` 是局部模型的输出，表示局部预测的结果。
- `h` 是累积的预测结果，最终会通过sigmoid函数转换为概率值。

---

### 5. **组合结果并输出最终决策（Combine Results and Output Final Decision）**

**步骤描述：**

- 将全局模型和局部模型的输出结果相结合后，经过sigmoid函数，得到最终的预测概率。
- 这个概率表示输入数据属于某一类别的可能性，通常大于0.5的被认为是正类，小于0.5的被认为是负类。

**需要做的事：**

- 最终的预测结果是通过sigmoid函数计算得到的概率值。

**伪代码：**

```python

s_mean = s_mean + s  # 将每个组的预测结果累加起来
s_mean = s_mean / self.N_multiple  # 取多个组预测结果的平均值

```

**解释：**

- `s_mean` 是累积的多个模型组的预测结果的平均值。
- `self.N_multiple` 是模型组的数量，最终取每个组预测结果的平均值作为最终决策。