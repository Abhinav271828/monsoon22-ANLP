---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 12 September, Monday (Lecture 10)
author: Taught by Prof. Manish Shrivastava
header-includes:
- \newfontfamily\devanagarifont{KohinoorDevanagari-Regular}
---

# Transformers
The motivation behind transformers was to have a simple network architecture based on attention, in order to do away with the recurrence enforced by RNNs. They were intended to have a lower training cost than RNNs.

## Positional Encoding
We have seen that the input is formed by a fully connected network. While it makes all the information accessible to all the words, it makes the representation order-invariant – it is now a bag-of-words model. The inputs need to be augmented in some way to convey the information of their (relative) position.

However, if the position is indicated by a monotonic function, it would quickly grow beyond manageable sizes; thus we can use a periodic function.  
We need to, however, make sure that the period of this function exceeds the size of the input, so as to not repeat the encodings. This is achieved by combining periodic functions out of sync, so that the function takes longer to repeat.

The most common form of positional encoding uses sinusoidal functions. The $n^\text{th}$ index of the $p^\text{th}$ vector in the input is encoded with
$$\text{PE}(p, n) = \begin{cases}
\sin \left(\frac{p}{10000^{\frac{2i}{d}}}\right); & n = 2i \\
\cos \left(\frac{p}{10000^{\frac{2i}{d}}}\right); & n = 2i+1 \\
\end{cases}$$
where $d$ is the dimensionality of the model.

For any fixed $k$, $\text{PE}(p+k)$ is a linear function of $\text{PE}(p)$. Thus it was hypothesised that these functions could allow the model to easily attend by relative positions.

The authors also experimented with learned positional embeddings, and found that nearly identical results were produced. Sinusoidal encoding was chosen over this, however, to allow the model to extrapolate to sequence lengths longer than 512.

## Scaled Dot-Product Attention
We have seen that attention is carried out with the query and key matrices $Q, K$ (here representing the query and key representations of the input itself) by using the dot product scaled:
$$\text{Attn}(Q, K, V) = \operatorname*{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V.$$

We can have multiple *heads*, which create different channels for importance-related information to be passed around the sentence. This is expressed as
$$\begin{split}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \\
\text{head}_i &= \text{Attn}(QW_i^Q, KW_i^K, VW_i^V),
\end{split}$$
where
$$\begin{split}
W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_\text{model} \times d}.
\end{split}$$

Layer normalisation is then carried out on these values (across $\text{dim}=-1$).

## Position-Wise Feedforward Networks
Each layer in the encoder (and decoder) is then followed by a fully connected FFN, which is applied to each position fully and identically. This serves to introduce a nonlinearity in the network.

It consists of two networks with an activation in betweem:
$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1) \cdot W_2 + b_2.$$