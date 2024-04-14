# heinsen_position_embeddings

Reference implementation of "[Encoding Position by Decaying and Updating Different Exponentiated States Differently](assets/preprint.pdf)" (Heinsen, 2024) (arXiv link pending).


* [Installing](#installing)

* [Using](#using)

* [Notes](#notes)

* [Citing](#citing)


## Installing

```
pip install git+https://github.com/glassroom/heinsen_position_embeddings
```

Alternatively, you can download a single file to your project directory: [heinsen_position_embeddings.py](heinsen_position_embeddings/heinsen_position_embeddings.py).

The only dependency is PyTorch.


## Using

```
from heinsen_position_embeddings import EmbedPosition

embed_pos = EmbedPosition(d_emb=1024, d_hid=1024)

x = torch.randn(1000, 1024)  # 1000 tokens, each with 1024 elements
x = embed_pos(x)             # with position info embedded in them
```

In practice, we have found it useful to apply LayerNorm afterwards, for numerical stability. As always, you should compare to other aproaches for encoding position information to determine which method works best for your particular use case.


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
