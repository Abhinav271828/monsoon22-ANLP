---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 29 August, Monday (Lecture 6)
author: Taught by Prof. Manish Shrivastava
---

# Attention
There are three important factors in machine translation: the order of the output tokens (language modelling), the syntactic correlation between input and output tokens, and the semantic correlation between input and output tokens.

The importance of these notions motivates a small change to the decoder architecture of a seq2seq model – the target sequence generation takes an additional input (at each timestep $t$) which is the weighted sum of all input tokens $\sum \alpha_{ti} h_i$, where the weights $\alpha_{ti}$ are proportional to the "focus" each one requires. This idea is the basis of *attention*.  
The actual values of the weights $\alpha_{ti}$ are dependent on $s_{t-1}$ (the output of the previous timestep) and on $h_i$. This dependence could be of the form
$$\alpha_{ti}' = \text{MLP}(s_{t-1} \oplus h_i)$$
or
$$\alpha_{ti}' = s_{t-1}^T W h_i.$$
(where $W$ could be trainable or fixed to $I$).  
Note that we need to normalise these scores to get the final weights:
$$\alpha_{ti} = \operatorname*{softmax}_i \alpha_{ti}'$$

Using these weights, we obtain a context that changes with time to (ideally) focus on the relevant parts of input:
$$c_t = \sum_i \alpha_{ti} h_i.$$