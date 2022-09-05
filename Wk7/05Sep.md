---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 05 September, Monday (Lecture 8)
author: Taught by Prof. Manish Shrivastava
header-includes:
- \newfontfamily\devanagarifont{KohinoorDevanagari-Regular}
---

# Components of Seq2seq
We have seen the major components of neural models: the encoder (which carries out analysis), the decoder (which carries out generation), and attention.

Attention is important for many reasons – for example, the lexical choice of target tokens depends on the register of the corresponding token on the source side (*e.g.*, "mother" translates to [माता]{lang=hi} but "mom" to [मम्मी]{lang=hi}).  
Attention also helps to lessen the "information burden" (albeit in an ill-defined sense, since we lack an information-theoretic bound) on the single vector conveying the meaning of the sentence from the encoder to the decoder.

The main purpose of the decoder (a language model) is to enforce the *acceptability* of the generated sequence. In a more general sense, it interprets the guidance received by attention to generate the appropriate output, which may incorporate meaning, style and register.

The encoder is the component responsible for providing the decoder with this guidance. In a more general sense, it needs to create a representation from which the relevant information can be extracted. This is done by *contextual representation* models, like ELMo.

## Encoding
Encoding, at its core, is a task of *meaning aggregation* – the meanings of individual tokens (represented as vectors) need to be combined into a single representation.

One way of improving on RNNs is to add skip connections between input tokens. This creates an undirected, fully connected graph, allowing every token to directly influence every other. However, RNNs in general have a time efficiency problem.

We can allow each word to simply pay attention to every word in the input.