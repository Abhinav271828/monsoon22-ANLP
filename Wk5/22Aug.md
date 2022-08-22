---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 22 August, Monday (Lecture 5)
author: Taught by Prof. Manish Shrivastava
---

# NLP with ML (contd.)
The development of RNNs produced an architecture for generating output *sequences* (rather than single tokens), called the seq2seq (or sequence-to-sequence) model, which found special use in automatic machine translation. This model takes a sequence of inputs and generates a meaning-aware representation of it (*encodes* it). It is a direct borrowing of the traditional translation model, which attempts to calculate $P(\overrightarrow{s} \mid \overrightarrow{t})$.  

This, however, makes the encoding vector an information bottleneck during translation. A single `$*@#` vector cannot encode the entire `$*@#` sentence's semantic content. This led to the idea of *attention* – the output makes use of the entire input at each step of generation, ranking the tokens by a set of weights.