---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 29 September, Thursday (Lecture 13)
author: Taught by Prof. Manish Shrivastava
header-includes:
- \newfontfamily\devanagarifont{KohinoorDevanagari-Regular}
---

# BERT
BERT is often called "the ImageNet of language", as it was built on the same principle – pretrain a model on the most complex possible task (language modelling in the case of language), and abstract from it for other tasks. This abstraction is carried out by adding layers on top of the pretrained component.

The pretraining is done in a manner analogous to the skip-gram model; the specific name of this task is *masked language modelling* (MLM). Some percentage of the words (usually about 15%) are randomly replaced (with either a `[MASK]` token or a random word) and the model tries to predict them.

BERT is also pretrained on a *next sentence prediction* task. The `[CLS]` token of a pair of sentences is used to predict whether or not one follows the other.