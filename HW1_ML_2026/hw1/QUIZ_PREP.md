# Quiz Prep: HW1 Regression & Classification

Quick reference for concepts, formulas, and typical quiz-style questions from the assignment.

---

## 1. Linear Regression

**Problem:** Find $\mathbf{w}$ such that $\mathbf{X}\mathbf{w} \approx \mathbf{y}$ (design matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$, targets $\mathbf{y} \in \mathbb{R}^n$).

**Loss (squared error / ERM):**

$$\mathcal{L}(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 = (\mathbf{X}\mathbf{w} - \mathbf{y})^T(\mathbf{X}\mathbf{w} - \mathbf{y})$$

**Normal equations (set gradient to zero):**

$$\nabla_\mathbf{w} \mathcal{L} = 2\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y}) = 0 \quad \Rightarrow \quad \mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}$$

**Closed-form solution:**

$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

Requires $\mathbf{X}^T\mathbf{X}$ to be **invertible**.

**Warm-up (no intercept, 1D):** $w^* = \frac{\sum_i x_i y_i}{\sum_i x_i^2}$.

---

## 2. Why Ridge Exists: Numerical Stability

**Condition number** of $\mathbf{A}$:

$$\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

- **High $\kappa$** ⇒ small changes in $\mathbf{b}$ cause large changes in $\mathbf{x}$ when solving $\mathbf{A}\mathbf{x} = \mathbf{b}$.
- For regression we solve $(\mathbf{X}^T\mathbf{X})\mathbf{w} = \mathbf{X}^T\mathbf{y}$.
- **Multicollinearity** (highly correlated features) ⇒ $\mathbf{X}^T\mathbf{X}$ becomes **ill-conditioned** (high $\kappa$) ⇒ OLS is unstable.

---

## 3. Ridge Regression (L2)

**Loss:**

$$\mathcal{L}_{\text{ridge}}(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$

**Closed-form solution:**

$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

- Adding $\alpha \mathbf{I}$ **improves condition number** ⇒ more stable than OLS when $\mathbf{X}^T\mathbf{X}$ is ill-conditioned.
- **Ridge shrinks weights toward zero** but does **not** set them exactly to zero.

---

## 4. LASSO Regression (L1)

**Loss:**

$$\mathcal{L}_{\text{LASSO}}(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \alpha \|\mathbf{w}\|_1$$

**Why no closed-form?** The L1 penalty $|w|$ is **not differentiable at $w = 0$**. Use **iterative** methods.

**Coordinate descent + soft-thresholding:**

$$S(x, \lambda) = \operatorname{sign}(x) \cdot \max(|x| - \lambda, 0)$$

- **LASSO can set weights exactly to zero** ⇒ **sparse solutions** and **automatic feature selection**.

| | Ridge (L2) | LASSO (L1) |
|---|------------|------------|
| **Solution** | Closed-form | Iterative (e.g. coordinate descent) |
| **Sparsity** | No (weights → 0 but ≠ 0) | Yes (some weights = 0) |
| **Correlated features** | Handles well | May pick one arbitrarily |
| **Use when** | All features matter | Few features matter |

**Elastic Net:** $\alpha_1 \|\mathbf{w}\|_1 + \alpha_2 \|\mathbf{w}\|_2^2$ (combines L1 and L2).

---

## 5. Kernel Ridge Regression

**Idea:** Work in a high-dimensional (possibly infinite) feature space via a **kernel** without computing $\phi(\mathbf{x})$ explicitly.

**Kernel:** $k(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T \phi(\mathbf{x}')$.

**Dual form:** Solve for coefficients $\boldsymbol{\alpha}$ (one per training point):

$$\boldsymbol{\alpha} = (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{y}, \qquad \mathbf{K}_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$$

**Prediction at $\mathbf{x}$:**

$$\hat{y} = \sum_{i=1}^{n} \alpha_i \, k(\mathbf{x}, \mathbf{x}_i)$$

**Kernels:**
- **Linear:** $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$ → $\mathbf{K} = \mathbf{X}_1 \mathbf{X}_2^T$.
- **RBF (Gaussian):** $k(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$. Larger $\gamma$ ⇒ more local (smaller "bandwidth").

**Vectorized RBF trick:** $\|\mathbf{x}_i - \mathbf{x}_j\|^2 = \|\mathbf{x}_i\|^2 + \|\mathbf{x}_j\|^2 - 2\mathbf{x}_i^T\mathbf{x}_j$.

---

## 6. Logistic Regression (Binary Classification)

**Goal:** Predict $y \in \{0, 1\}$ from $\mathbf{x}$. Model probability of class 1.

**Sigmoid:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Use `np.clip(z, -500, 500)` before `exp` to avoid overflow. $\sigma(0) = 0.5$; $\sigma(z) \to 1$ as $z \to +\infty$, $\to 0$ as $z \to -\infty$.

**Probability:** $p_i = \sigma(\mathbf{w}^T\mathbf{x}_i + b)$.

**Loss (binary cross-entropy / log loss):**

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{n}\sum_{i=1}^{n} \Big[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \Big]$$

**Gradient descent updates:**

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{w}}, \qquad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$

With the usual derivation:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{n} \mathbf{X}^T (\mathbf{p} - \mathbf{y}), \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_i (p_i - y_i)$$

**Takeaways:**
- **No closed-form solution** → use iterative optimization (e.g. gradient descent).
- **Decision boundary** is linear: $\mathbf{w}^T \mathbf{x} + b = 0$.
- Logistic regression is the **classification analogue** of linear regression.

---

## Quick Quiz-Style Checklist

- [ ] Write the OLS loss and normal equations; state $\mathbf{w}^*$.
- [ ] Explain condition number and why OLS can be unstable (ill-conditioned $\mathbf{X}^T\mathbf{X}$).
- [ ] Write Ridge loss and closed-form; say why it stabilizes (adding $\alpha\mathbf{I}$).
- [ ] Ridge vs LASSO: closed-form vs iterative, sparsity, when to use which.
- [ ] Soft-thresholding formula and that LASSO gives sparse weights.
- [ ] Kernel trick: $k(\mathbf{x},\mathbf{x}') = \phi(\mathbf{x})^T\phi(\mathbf{x}')$; dual Ridge solution $(\mathbf{K}+\lambda\mathbf{I})\boldsymbol{\alpha} = \mathbf{y}$.
- [ ] Linear kernel $K = X_1 X_2^T$; RBF $k = \exp(-\gamma \|\mathbf{x}-\mathbf{x}'\|^2)$.
- [ ] Sigmoid formula and numerical trick (clip).
- [ ] Log loss (binary cross-entropy) formula.
- [ ] Logistic regression: gradient descent, no closed-form, linear decision boundary.

Good luck on the quiz.
