---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 20 October, Thursday (Lecture 14)
author: Taught by Prof. Manish Shrivastava
header-includes:
- \newfontfamily\devanagarifont{KohinoorDevanagari-Regular}
---

# Generative Models
Deep learning generative models have the advantages that they do not make grammatical errors, and generalise well.  
However, they tend to be incoherent or inconsistent, and to suffer from hallucination (factuality is hard to guarantee). They are also limited in the amount of input they can take at one time (the *horizon problem*).

Factuality is a main focus in text generation. One type of problem is the inability to replicate extremely specific information present in the source (like a year or a football match score). There are various approaches to this – copy generation, vocabulary augmentation, etc.