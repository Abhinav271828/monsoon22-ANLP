---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 01 September, Thursday (Lecture 7)
author: Taught by Prof. Manish Shrivastava
---

# Attention (contd.)
Thus, we have a *query* (the previous output hidden state $s_{t-1}$) that gives us the weights. Using these weights, we then obtain, for each *key* (input token) a *value* (a new vector representation that incorporates the weight).

We can use attention for sentence-level classification tasks as well. The general mechanism for this would be
$$\begin{split}
c_\text{sent} &= \operatorname*{argmax}_C \text{attn}(S_\text{task}, H_\text{sent}), \\
H_\text{sent} &= \text{RNN}(\text{sent})
\end{split}$$
where $C$ is the set of classes and $S_\text{task}$ is some random query vector. The attention is then tuned to extract information relevant to the task.

It is also possible to fix the attention layer at a random initial value and tune only the query vector $S_\text{task}$. This can now be taken as a representation for the task itself.