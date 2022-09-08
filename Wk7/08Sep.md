---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 08 September, Thursday (Lecture 9)
author: Taught by Prof. Manish Shrivastava
header-includes:
- \newfontfamily\devanagarifont{KohinoorDevanagari-Regular}
---

# Components of Seq2seq
The individual components (encoder and decoder) of the model may be *pre-trained* (on a language generation task) and frozen, with *adapter layers* after each of them to extract the relevant information from them. Using PLMs has the advantage that they are more robust and have been exposed a large amount of open-domain data.

## Encoder (contd.)
We have seen that words need information from other words in the sentence, which takes a number of "steps" if we restrict ourselves to sequential processing. We can instead allow any word to influence any other, *i.e.*, start with a fully-connected graph, and prune it. This is the approach taken by transformers.  
However, different words are important in different ways. For example, a noun's modifier and its verb both have relevant information, but not the same kind. This gives rise to the idea of *multi-headed attention* – a number of attention *heads*, each distributing attention differently over the sentence.

Each head is associated with three matrices $Q$, $K$ and $V$. The $Q$ matrix converts a word to a representation that *requests* relevant information from other words (a query); the $K$ matrix converts a word to one that *provides* relevant information (a key); and the $V$ matrix is gives the final representation that is weighted and summed (a value). The exact formula in this model is
$$c(w_i) = \sum \alpha_{ij} Vw_j,$$
where
$$\alpha_{ij} = \operatorname*{softmax}_j \left(\frac{(Qw_i) (Kw_j)^T}{\sqrt{d_k}} \right).$$
[$\sqrt{d_k}$ is a normalising term that we add to prevent the values from being in too wide a range.]

We add (or, more generally, combine) this context representation to the original representation $w_i$ (followed by normalisation), combining the information from both these vectors. This way we ensure that the core of information from $w_i$ itself is not lost, while at the same time diluting the noise present in the contextual representation.