# heinsen_position_embeddings

Reference implementation of "[Encoding Position by Decaying and Updating Different Exponentiated States Differently](assets/preprint.pdf)" (Heinsen, 2024) (arXiv link pending).


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

embed_pos = EmbedPosition(d_emb=1024, d_hid=1024)

x = torch.randn(1000, 1024)  # 1000 tokens, each with 1024 elements
x = embed_pos(x)             # tokens with position info embedded in them
```
In practice, for numerical stability, we have found it useful to apply LayerNorm (or some other kind of normalization) before computing any subsequent transformations of token states in a model.


### Recurrent Application

Our method for encoding position information is recurrent, so you can embed position information in sequences of tokens that are split in chunks, with no preset limit on sequence length.

To encode position information in each new chunk from a stream of chunks, specify `using_prev_context=True` in each forward pass after the first one:

```python
chunk1 = torch.randn(1000, 1024)                     # first chunk of tokens
chunk1 = embed_pos(chunk1)                           # module caches its ending state

chunk2 = torch.rand(1000, 1024)                      # continues first chunk
chunk2 = embed_pos(chunk2, using_prev_context=True)  # starts from cached state

chunk3 = torch.rand(1000, 1024)                      # continues second chunk
chunk3 = embed_pos(chunk3, using_prev_context=True)  # starts from cached state
```


## Customizing

All code is in a [single file](heinsen_position_embeddings/heinsen_position_embeddings.py) for easy customization. The module incorporates two feed-forward components, `H` and `R`, defined by default as `nn.Linear` layers with biases, that you can customize. (`H` corresponds to function $\mathcal{H}$ in the paper, but without the Sigmoid function, and `R` corresponds to function $\mathcal{R}$).


## Compared to Other Methods

In limited comparison experiments, we have found that our method for encoding position information performs similarly to other methods for encoding position information, but offers many benefits that make it a worthwhile choice, including large representational capacity, low compute cost, and small memory footprint -- in addition to unbounded sequence length. As always, we recommend testing and comparing against other alternatives to determine which one will work best for your specific application.


## Notes

We have tested the code in this repository only on Ubuntu Linux 22.04 with Python 3.10+.


## Citing

Until our arXiv link goes live, please cite as follows:

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
