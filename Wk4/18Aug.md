---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 18 August, Thursday (Lecture 4)
author: Taught by Prof. Manish Shrivastava
---

# NLP with ML (contd.)
A typical language model processes text in a strictly *linear* fashion – no structural information of the sentences being processed is taken into account. This is not merely for the sake of computational convenience; it also prevents the propagation of errors through the multiple submodules that structural analysis of the input would entail.  
We hypothesise that the representations of words that neural LMs learn (which enable them to predict the next word) have *some* semantic information, even though they are learnt without any real grounding. This arises out of the distributional semantics assumption, that a word's meaning is spread out over the contexts it appears in.

Word2vec and GloVe are implementations of this idea – each word (more specifically, *word form*) is associated with a unit vector representing its meaning. However, these vectors are *global* – the various senses of polysemous or homonymous words are compressed into a single representation. Some other approaches to neural distributional semantics were CoVe (contextual vectors) and FastText (character-based, in addition to word-based, modelling).

RNNs are a method of processing language which is linear in nature. However, they can be modified to introduce further information "channels" that allow for the modelling of a wider range of structures. For example, one RNNs can be stacked on another to traverse the input in the reverse direction and the outputs of both layers combined, forming a *bidirectional* RNN. *Multilayered* RNNs (with layers stacked on each other) allow for the formation of information complexes, based on the context of words, which can be aggregated in higher layers.