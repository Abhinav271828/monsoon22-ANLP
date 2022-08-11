---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 11 August, Thursday (Lecture 3)
author: Taught by Prof. Manish Shrivastava
---

# NLP with ML (contd.)
The way SVD functions ensures that the word vectors we obtain are normalised, *i.e.*, their amplitudes are uniformly 1. Neural methods, however, may not learn normalised vectors.

We have seen that neural models may use a focus word to predict a context word (called *skip-gram*), or the context to predict a focus word (called *continuous bag of words*, or CBOW). The idea of using next-word prediction as a task (*language modelling*) started from Claude Shannon's information-theoretic study of the latent properties of language. This, however, viewed language as a linearly generated system, which we know it is not – the *latent* structuring of language is recursive and nonlinear.  

The fact that there is *some* structure to language means that the *entropy* of language is restricted. The prediction distribution at each point would be approximately uniform if language was a completely chaotic system, but it is not; there is some order in the chaos.

There are a number of ways language modelling can be done (statistically as well as neurally).  

Statistical language models generally make the *Markov assumption* – the current word is dependent only on the previous $n$ words, for some finite $n$.  
A simplistic method would be to count the number of occurrences of the previous $n$ words and divide by it the number of occurrences of all $n+1$ words together:
$$p(w_i \mid w_{i-n} \dots w_{i-1}) = \frac{c(w_{i-n} \dots w_{i-1}w_i)}{c(w_{i-n} \dots w_{i-1})}.$$
However, this suffers from a sparsity problem; as $n$ increases, $c(w_{i-n:i-1})$ will tend to become 0. This is usually solved by techniques like *backoff* or *interpolation*.  
Using these models, we calculate the *probability of sequences of words* using the chain rule
$$p(w_1 \dots w_k) = \prod_{i=1}^k p(w_i \mid w_1 \dots w_{i-1}).$$
We therefore wish to prevent these conditional probabilities from having zero values, which would collapse the entire sequence probability to zero. This issue is solved by *smoothing* the probabilities.  
Language models find application in *text-generative* tasks, like response generation, translation, summarisation, caption generation and so on. In all these tasks, we always try to *maximise the likelihood* of the generated output. However, it must be noted that these language models are specific to the domain of their training data.

The most simple neural language model is exactly similar to statistical models – a fixed-width input of the previous $n$ words, which are passed through one or more hidden layers and used to predict a distribution over the vocabulary (using a softmax). This is treated as a multi-class classification problem *over the entire vocabulary*. This is the basis of the CBOW task (with the modification that the context includes the next words as well as the previous ones), which is one way to train the model (the skip-gram task can be seen as a generalised version of bigram-based modelling).

However, the Markov assumption continues to restrict the model space here. Models that do away with this assumption use *each word to predict the next*, maintaining a context as they proceed – this context stores information from the history that was used to predict the current word, and the confidence of the model in that prediction. This context is the *hidden state* of the model, which is used in prediction. Mathematically,
$$h_i = \sigma(Wh_{u-1} + Ux_i)$$
$$x_{i+1} = \sigma(W'h_{i} + U'x_i)$$
Such models are called *recurrent neural networks*, and have many variants that modify this basic idea.  
In theory, the information related to $x_0$ may influence the distribution for $x_{99}$ (in practice, however, the effect does not tend to be significant over such long distances; this is called the *vanishing gradient problem*).

Besides the vanishing gradient problem, RNNs also had problems related to computational efficiency – since each word is used to predict the next, it is *at least linear* in the size of the input.

Language models are evaluated on their *perplexity*, defined in terms of entropy $\mathcal{H}$ as
$$P = 2^\mathcal{H} = 2^{-\sum p_i \log p_i}.$$
This value encodes the *average number of decisions* taken by the model at every decision point.