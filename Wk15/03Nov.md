---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 03 November, Thursday (Lecture 17)
author: Taught by Prof. Manish Shrivastava
header-includes:
- \newfontfamily\devanagarifont{KohinoorDevanagari-Regular}
---

# Generative Models (contd.)
## Abstractive Summarisation
OOV tokens pose a problem for abstractive summarisation – a simple generative model will never generate them. To remedy this, we define a soft switch $p_\text{gen}$, to model whether the word is likely to be found in the vocabulary or not. This is calculated as
$$p_\text{gen} = \sigma(w_c c_i w_s s_i + w_x x_i + b).$$
It is a kind of confidence value (confidence that the word to be generated can be found in the vocabulary).

The output word is then sampled as
$$P(w) + p_\text{gen}P_\text{vocab}(w) + (1-p_\text{gen})\sum_{w_j=w}a_j^i.$$

Note that nothing stops the model here from copying the entire input sentence as is. This means that even abstractive models are partially extractive – large phrases are often lifted verbatim from the input.

Such models also suffer from another issue – the generation goes into a loop since each timestep is not informed of the previous decisions taken at previous timesteps. Coverage mechanisms are used to remedy this. We maintain a *coverage vector*
$$\text{cov}_t = \sum_{j=0}^{t-1} a_j.$$
We then use this vector to score the input terms for copying (?):
$$e_{ij} = v^T \tanh(Ws_{i-1} + Uh_j + V\text{cov}_j + b_\text{attn}).$$

# Machine Reading Comprehension
In essence, the task of reading comprehension involves answering a question provided with one or more pieces of evidence.