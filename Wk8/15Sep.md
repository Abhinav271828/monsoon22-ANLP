---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 15 September, Monday (Lecture 11)
author: Taught by Prof. Manish Shrivastava
header-includes:
- \newfontfamily\devanagarifont{KohinoorDevanagari-Regular}
---

We know that an artificial neuron has two parts – an aggregator
$$\text{net}_j = \left(\sum_i W_{ij}x_i \right) + b_j$$
or
$$\overrightarrow{\text{net}} = \overrightarrow{x}W + \overrightarrow{b}$$
and an activation
$$z = \tanh(\overrightarrow{\text{net}}).$$

$\overrightarrow{\text{net}}$ represents the distance of the point from the classification boundary. The tanh function (or any other activation) then scales this down to 1 or -1 according as the point is on one side or another of the boundary.

# Transformers (contd.)
## Add & Norm
We have seen that transformers add the input and normalise at the end of every layer. This comes from the idea of giving the output access to more than one "view" of the input; for example, after each encoder layer, the output is of the form
$$\text{PE}(x) + \text{MHA}(\text{PE}(x)) + \text{FFN}(\text{PE(x)} + \text{MHA}(\text{PE}(x))).$$
or
$$((\text{Id} + \text{FFN}) \circ (\text{Id} + \text{MHA}) \circ \text{PE})(x).$$

This idea is central to the resnet (*residual networks*) architecture as well.

## Decoder
The decoder relies on many of the same ideas as the encoder, with one generalisation – *masked* attention. Decoder self-attention requires us to attend only to the outputs that have already been generated, while the others are hidden, or masked. Further, the multi-headed attention is carried out over the encoder outputs.