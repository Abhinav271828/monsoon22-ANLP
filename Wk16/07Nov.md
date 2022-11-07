---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 07 November, Monday (Lecture 18)
author: Taught by Prof. Manish Shrivastava
header-includes:
- \newfontfamily\devanagarifont{KohinoorDevanagari-Regular}
---

# Generative Models (contd.)
## GPT
The decoder side of a transformer has three main parts – masked self-attention, decoder-encoder attention, and a feed forward NN. When we consider a decoder in isolation (which is what GPT is), the middle component cannot be considered (why?).

A natural task for pretraining a decoder is, for example, next-word prediction. Its huge parameter space is intended to allow it to generalise easily to downstream tasks – this can be done in a finetuning, zero-shot, one-shot, or few-shot setting.