



\# Transformer for Sequence-to-Sequence Tasks



\## Objective



The purpose of this assignment is to provide a comprehensive understanding of Transformer models in sequence-to-sequence learning. By completing this assignment, students should be able to:



\* Explain the concept of attention mechanisms and multi-head attention.

\* Understand how encoder-decoder architectures process input and target sequences.

\* Implement a Transformer architecture using PyTorch.

\* Train and evaluate a Transformer for tasks such as arithmetic expression evaluation.



---



\## What This Assignment Does



This assignment implements a Transformer for sequence-to-sequence tasks. The workflow consists of:



1\. **Data Loading:** A custom PyTorch Dataset prepares input-target sequences, applies tokenization, and generates positional encodings.



2\. **Encoder:** Processes the input sequence with multiple layers of multi-head self-attention and feed-forward networks, applying residual connections and layer normalization.



3\. **Decoder:** Generates output sequences using masked multi-head attention for autoregressive modeling, cross-attention with encoder outputs, feed-forward layers, and residual connections.



4\. **Positional Encoding:** Adds positional information to embeddings using either simple or sinusoidal encodings.



5\. **Training and Evaluation:** Supports standard cross-entropy loss as well as label smoothing for more robust training.



---



\## Dataset



The dataset consists of pairs of input and target sequences. For example, arithmetic expressions:



```

Input:  "1+2"

Target: "3"

```



Sequences are tokenized and converted to numerical tensors, and special tokens (`BOS`, `EOS`) are added to mark sequence start and end.



---



\## Key Components



\* **Encoder-Decoder Architecture:** Standard Transformer blocks with multi-head attention, feed-forward layers, residual connections, and layer normalization.

\* **Masked Multi-Head Attention:** Ensures autoregressive prediction in the decoder.

\* **Cross-Attention:** Allows the decoder to attend to encoder outputs for effective sequence modeling.

\* **Feed-Forward Networks:** Capture complex sequence patterns.

\* **Positional Encoding:** Injects sequence order information into embeddings.

\* **Label Smoothing:** Helps prevent overconfidence during training.



---



\## References / Helping Material



1\. \[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

2\. \[Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

3\. \[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

4\. \[PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer\_tutorial.html)







