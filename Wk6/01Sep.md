---
title: Advanced NLP (CS7.501)
subtitle: |
          | Monsoon 2022, IIIT Hyderabad
          | 01 September, Thursday (Lecture 7)
author: Taught by Prof. Manish Shrivastava
---

# Attention (contd.)
Thus, we have a *query* (the previous output hidden state $s_{t-1}$) that gives us the weights. Using these weights, we then obtain, for each *key* (input token) a *value* (a new vector representation that incorporates the weight).