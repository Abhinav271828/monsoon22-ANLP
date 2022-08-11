---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 08 August, Monday (Lecture 2)
author: Taught by Prof. Manish Shrivastava
---

# NLP with ML
At its core, machine learning is simply learning knowledge representations, or abstractions, which generalise to unseen data.

The architecture of a model is determined by the nature of the task it needs to carry out. However, all architectures have in common that they attempt to simplify (or sometimes oversimplify) the input and process it. For example, a spam classification model might take the BOW (bag-of-words) approach, which makes the simplifying assumption that order does not matter, and operates on the set of words in the input text.  
The bottleneck in machine learning, then, is *representation*. The harder a task is, the better we need its representation to be; simpler tasks can often be performed without meaning-aware representations, while more complex tasks need representations corresponding to semantic features.

How do we model meaning of, say, words? Intuitively, we know that a part of the meaning of any word comes from its context (*e.g.*, "I bank at SBI" vs. "The river bank was muddy"). The notion that the meaning of a word is spread out over its contexts is called *distributional semantics*, and it forms the basis of word representation in deep learning.  
This idea leads to a method of embedding words into a vector space. We use a *co-occurrence* matrix $A$, which contains the number of times each word occurs *in the context of* some $n$ words, and decompose it into its singular values $U\SigmaV^T$. In this decomposition, $U$ is a *dense* representation of (the rows of) $A$, which we can take to represent the meanings of the words. Further, it can be compressed by truncation according to space or time constraints.

There are also neural methods to model word meaning as numerical data – popular systems include word2vec and GloVe. These methods are similarly based on the distributional semantics approach: they use context words to predict a missing "focus" word (or vice versa). We try to train the model to create a representation that achieves this as closely as possible.