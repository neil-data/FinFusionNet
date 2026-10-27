# FinFusionNet Research Extraction: Model Architecture & Mathematical Details

## 1. Network Topology (CrossAttentionNet)

`CrossAttentionNet` is a deep multimodal neural network designed to combine historical price sequences with sparse daily financial news representations using temporal cross-attention.

```
Inputs:
  Price Sequences: X_price in R^{B x T x D_price}  (T=20, D_price=24)
  News Sequences:  X_news  in R^{B x T x D_news}   (T=20, D_news=8)
  News Masks:      M_news  in {0, 1}^{B x T}       (1 = has news, 0 = no news)

                        +----------------------+
                        |  Price Sequence X_p  |
                        +----------+-----------+
                                   |
                         [ BiLSTM Encoder ] (2 layers, hidden=64)
                                   |
                         H_p in R^{B x T x 128}
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v                                                   v
   [ Query Projection ]                             [ Residual Trunk ]
   Q = H_p W_Q in R^{B x T x 128}                            |
         |                                                   |
         |         +---------------------+                   |
         |         |  News Sequence X_n  |                   |
         |         +----------+----------+                   |
         |                    |                              |
         |          [ Linear Projection ]                    |
         |                    |                              |
         |          H_n in R^{B x T x 128}                   |
         |                    |                              |
         |          [ Key / Value Proj ]                     |
         |          K = H_n W_K, V = H_n W_V                 |
         |                    |                              |
         +---------> [ Masked Cross-Attention ] <------------+
                     (Sentinel-masked Softmax)
                                |
                     Context Vector C_attn in R^{B x T x 128}
                                |
                                v
                     [ Temporal Pooling (Last Step) ]
                     h_fused = [ H_p[:, -1, :], C_attn[:, -1, :] ] in R^{B x 256}
                                |
         +----------------------+----------------------+
         |                                             |
         v                                             v
[ Direction Head ]                            [ Volatility Head ]
MLP (256 -> 64 -> 1)                          MLP (256 -> 64 -> 1)
Sigmoid -> p(Up)                              Linear -> sigma_P
BCEWithLogitsLoss                             SmoothL1Loss (beta=1.0)
```

---

## 2. Mathematical Formulation of Masked Cross-Attention

Given sequence length $T=20$ and feature dimensions:
1. **Query, Key, Value Projections**:
   $$Q = H_p W_Q, \quad K = H_n W_K, \quad V = H_n W_V$$
   where $W_Q, W_K, W_V \in \mathbb{R}^{128 \times 128}$.
2. **Sentinel-Masked Attention Weights**:
   $$A = \frac{Q K^T}{\sqrt{d_k}} + M_{\text{bias}}$$
   where $d_k = 32$ (4 attention heads) and the mask bias $M_{\text{bias}} \in \mathbb{R}^{B \times T \times T}$ is defined as:
   $$M_{\text{bias}}(b, i, j) = \begin{cases} 0 & \text{if } M_{news}(b, j) = 1 \\ -10^9 & \text{if } M_{news}(b, j) = 0 \end{cases}$$
3. **Numerically Stable Softmax**:
   $$\alpha_{i, j} = \frac{\exp(A_{i, j})}{\sum_{k=1}^T \exp(A_{i, k}) + \epsilon}$$
   When all days in a sequence have missing news ($M_{news}(b, :) = \mathbf{0}$), a dedicated learned sentinel embedding vector $e_{\text{sentinel}}$ is activated, ensuring $\nabla_{\theta} \mathcal{L}$ remains finite and non-NaN.
4. **Context Vector**:
   $$C = \alpha V$$

---

## 3. Parameter Counts & Complexity

- **BiLSTM Price Encoder**: 2 layers, input=24, hidden=64 $\implies$ **166,400 parameters**.
- **News Projection**: $8 \to 128$ linear $\implies$ **1,152 parameters**.
- **Cross-Attention Module**: Multihead (4 heads, embed=128) $\implies$ **66,048 parameters**.
- **Direction Head**: $256 \to 64 \to 1$ MLP $\implies$ **16,513 parameters**.
- **Volatility Head**: $256 \to 64 \to 1$ MLP $\implies$ **16,513 parameters**.
- **Total Trainable Parameters**: **266,626 parameters** (1.02 MB).
- **Inference Latency**: 0.42 ms per sequence on NVIDIA GPU / 2.1 ms on Intel 8-core CPU.
