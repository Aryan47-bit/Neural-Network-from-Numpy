# MNIST Neural Network from Scratch (NumPy)

A 2-layer fully connected neural network built from scratch using only **NumPy**, **Pandas**, and **Matplotlib**—no PyTorch, TensorFlow, or scikit-learn—to classify handwritten digits from the MNIST dataset.

---

## Architecture

The network uses a `[784 -> 128 -> 10]` dense architecture trained with full-batch gradient descent and categorical cross-entropy loss:

- **Input Layer ($A^{[0]}$):** 784 normalized pixel values (`28x28` flattened grayscale image)
- **Hidden Layer ($Z^{[1]}, A^{[1]}$):** 128 neurons with **He Initialization** and **ReLU** activation
- **Output Layer ($Z^{[2]}, A^{[2]}$):** 10 neurons with numerically stable **Softmax** activation

### Tensor Dimensions ($m$ = batch size)

| Tensor / Parameter | Shape | Description |
| :--- | :---: | :--- |
| `X` ($A^{[0]}$) | `(784, m)` | Normalized input pixel matrix |
| `W1`, `B1` | `(128, 784)`, `(128, 1)` | Hidden layer weights & biases |
| `Z1`, `A1` | `(128, m)` | Hidden pre-activations & ReLU outputs |
| `W2`, `B2` | `(10, 128)`, `(10, 1)` | Output layer weights & biases |
| `Z2`, `A2` | `(10, m)` | Raw logits & class probability distributions |

---

## Code Walkthrough

### 1. Data Ingestion, Transposing & Normalization
- **Train/Test Split & Transpose (`.T`):** The raw CSV has shape `(42000, 785)`, where each row is one sample. After shuffling and splitting (`1,000` test samples, `41,000` training samples), transposing flips the matrices so **each column is one image** (`(785, m)`). Row `0` holds the digit labels (`testY` / `trainY`), and rows `1:785` hold the 784 pixel intensities (`testX` / `trainX`).
- **Normalization (`/ 255.0`):** Scales raw `0–255` grayscale pixel intensities into `[0.0, 1.0]` to prevent large dot products in the first layer and keep gradients stable.

### 2. Initialization & Activation Math
- **`init_params()`:** Uses **He Initialization** (`np.random.randn(...) * np.sqrt(2.0 / fan_in)`), which keeps the variance of activations constant across ReLU layers so signals neither explode nor vanish at iteration `0`. Biases (`B1`, `B2`) are initialized to zeros.
- **`ReLU(x)` & `deriv_ReLU(Z)`:** The forward pass clamps negative pre-activations to `0` (`np.maximum(0, x)`). The backward pass returns a boolean mask (`Z > 0`, evaluated as `1` and `0` in multiplication) that blocks gradients from flowing through inactive neurons.
- **`softmax(x)`:** Subtracts `np.max(x, axis=0, keepdims=True)` from logits before exponentiating so large values never trigger `exp()` overflow (`inf`/`NaN`), then normalizes each image column (`axis=0`) into valid probabilities summing to `1.0`.
- **`one_hot(x)`:** Expands a 1D vector of integer labels `(m,)` into a `(10, m)` binary matrix with `1` at the ground-truth digit row and `0` elsewhere so it can be subtracted directly from `A2`.

### 3. Forward Propagation, Backpropagation & Updates
- **`forward_prop`:** Computes $Z^{[1]} \rightarrow A^{[1]} \rightarrow Z^{[2]} \rightarrow A^{[2]}$ across all `41,000` training images simultaneously via vectorized matrix multiplication (`.dot()`).
- **`backward_prop`:** Computes exact gradients via the chain rule, averaged across all $m$ samples (`1 / m`):
  - `dZ2 = A2 - one_hot_Y` represents the combined derivative of Categorical Cross-Entropy Loss and Softmax.
  - `dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)` propagates error backward through `W2` and applies the ReLU derivative mask element-wise.
  - `keepdims=True` on `np.sum(..., axis=1, keepdims=True)` keeps `db1` `(128, 1)` and `db2` `(10, 1)` aligned with `B1` and `B2` without unintended broadcasting.
- **`update_params` & `gradient_descent`:** Updates weights and biases for `500` iterations at learning rate `alpha = 0.1`, logging training accuracy every 50 steps using `np.argmax(A2, 0)`.

### 4. Inference & Visualization
- **Dimension Preservation (`None`):** Slicing a single sample via `trainX[:, index, None]` preserves the second axis (`(784, 1)` instead of a 1D `(784,)` array) so matrix multiplication and bias broadcasting work identically to full-batch inference.
- **Image Rendering:** `test_prediction` reshapes the `(784, 1)` column vector back into a `(28, 28)` pixel grid, rescales by `255`, and plots the grayscale digit alongside its predicted and actual labels using `matplotlib`.

---

## Project Structure

```text
├── data/
│   └── train.csv        # Kaggle Digit Recognizer dataset (42,000 samples)
├── model.py             # Training loop, forward/backward propagation, and inference
└── README.md            # Project documentation and code walkthrough
```

---

## Getting Started

### 1. Prerequisites
Install the required Python libraries:
```bash
pip install numpy pandas matplotlib
```

### 2. Dataset Setup
Download `train.csv` from the [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data) competition and place it inside the `data/` directory.

### 3. Run Training & Inference
Execute the script directly (or run cell-by-cell in VS Code using the `# %%` interactive markers):
```bash
python model.py
```

The model trains for `500` iterations (`alpha = 0.1`), prints training accuracy every 50 iterations, and displays sample digit predictions alongside their ground-truth labels.