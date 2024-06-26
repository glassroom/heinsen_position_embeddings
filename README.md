# heinsen_position_embeddings

Encoding position information in token sequences by decaying and updating the exponentiated states of different position-encoding features differently. At each step, we exponentiate the previous state of each position-encoding feature, decay it by a hidden probability, add to it an exponentiated hidden logit, and take the logarithm, obtaining the feature's updated state. We compute each hidden probability and logit dynamically from token state, making the method trainable by stochastic gradient descent.

The figure below shows how we update a single position-encoding feature's state $s_i \in \mathbb{R}$, with initial state $s_0$, given a hidden probability $p_i \in (0, 1)$ and logit $h_i \in \mathbb{R}$ at each step $i = (1, 2, \dots)$. For $d$ position-encoding features, $s_i \in \mathbb{R}^d$, $p_i \in (0, 1)^d$, and $h_i \in \mathbb{R}^d$. For more details, see our [research note preprint](https://github.com/glassroom/heinsen_position_embeddings/blob/main/assets/preprint.pdf).

![State of a single position-encoding feature](assets/single_position_encoding_feature.png)

We believe this is a new method for encoding position information in token sequences.


## Installing

```
pip install git+https://github.com/glassroom/heinsen_position_embeddings
```

Alternatively, you can download a single file to your project directory: [heinsen_position_embeddings.py](heinsen_position_embeddings/heinsen_position_embeddings.py).

The only dependency is PyTorch.


## Using

Our implementation is a PyTorch `nn.Module`, easily added as a component to any model:

```python
from heinsen_position_embeddings import EmbedPosition

batch_sz, n_tok, d_emb, d_hid = (8, 1000, 1024, 1024)  # setup for toy example
embed_pos = EmbedPosition(d_emb, d_hid)                # instantiate module

x = torch.randn(batch_sz, n_tok, d_emb)  # token states without position info
x = embed_pos(x)                         # token states with position info
```

In practice, for numerical stability, we have found it useful to apply LayerNorm (or some other kind of normalization) before computing any subsequent transformations of token states in a model.


### Recurrent Application

Our method for encoding position information is recurrent, so you can embed position information in sequences of tokens that are split in chunks, with no preset limit on sequence length.

To encode position information in each new chunk from a stream of chunks, specify `using_prev_context=True` in each forward pass after the first one:

```python
chunk1 = torch.randn(batch_sz, n_tok, d_emb)         # first chunk of tokens
chunk1 = embed_pos(chunk1)                           # module caches its ending state

chunk2 = torch.rand(batch_sz, n_tok, d_emb)          # continues first chunk
chunk2 = embed_pos(chunk2, using_prev_context=True)  # starts from cached state

chunk3 = torch.rand(batch_sz, n_tok, d_emb)          # continues second chunk
chunk3 = embed_pos(chunk3, using_prev_context=True)  # starts from cached state
```


## Customizing

All code is in a single file, at `heinsen_position_embeddings/heinsen_position_embeddings.py`, for easy customization. The module incorporates two feed-forward components, `H` and `R`, defined by default as `nn.Linear` layers with biases, that you can replace with other feed-forward transformations. Component `H` corresponds to function $\mathcal{H}$ in our preprint, but without the Sigmoid function, as we already apply it subsequently in the domain of logarithms with `F.logsigmoid`. Component `R` corresponds to function $\mathcal{R}$ in our preprint.


## Compared to Other Methods

In limited comparison experiments, we have found that our method for encoding position information performs similarly to other methods (_i.e._, neither significantly better nor significantly worse). However, our method offers many benefits that make it a worthwhile candidate for application, including large representational capacity, low compute cost, and small memory footprint -- in addition to unbounded sequence length. In our limited experiments, we have always kept `d_hid` equal to `d_emb`. We have not yet tested our method with `d_hid` different from `d_emb`.

As always, _we recommend testing and comparing against other alternatives to determine which one will work best for your specific application_. For an overview of many other proposed methods, see [here](https://direct.mit.edu/coli/article/48/3/733/111478/Position-Information-in-Transformers-An-Overview).


## Frequently Asked Questions

_Q: Isn't this a type of recurrent neural network (RNN)?_

Yes. We formulate our method as a recurrent transformation, so it is an RNN -- albeit a really simple one. We like to think of it as a "minimally viable RNN." Like all RNNs, this one enables past tokens to "send information" to the current token via a hidden state.


_Q: Couldn't I use this RNN for sequence modeling on its own, say, by stacking multiple layers of it in a deep model?_

Yes. Keep in mind that like other RNNs, this one lacks the ability to query past tokens as a function of the current token's state. To the best of our knowledge, at present only attention mechanisms can query past tokens as a function of current token state.


_Q: Why does the module detach the ending state before caching it?_

We assume you will train the module in parallel over whole sequences, as is conventional. If for some reason you want to train the module one token at a time, you can change our code so it doesn't automatically detach state, and handle detaching on your own. Keep in mind that training one token at a time may be significantly slower.


_Q: Can I use these position embeddings in multiple blocks of my Transformer model?_

Yes. We have not tested it, but we would expect it to work well.


## Notes

We have tested the code in this repository only on Ubuntu Linux 22.04 with Python 3.10+.


## Citing

If you find our work useful, please cite it:

```
@misc{heinsen2024position,
      title={Encoding Position by Decaying and Updating Different Exponentiated States Differently}, 
      author={Franz A. Heinsen},
      year={2024},
      primaryClass={cs.LG}
}
```


## How is this used at GlassRoom?

We conceived and implemented our attention mechanism for proprietary use. Most of the original work we do at GlassRoom tends to be tightly coupled to internal code, so we cannot share it with outsiders. In this case, however, we were able to isolate our code and release it as stand-alone open-source software without having to disclose any key intellectual property. We hope others find our work and our code useful.

