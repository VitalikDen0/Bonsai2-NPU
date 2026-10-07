# Hadamard MLX preview

Text-only affine 2-bit/group-128 weights. Load with the bundled runtime:

```python
import sys
sys.path.insert(0, '/path/to/pack/runtime')
from artifact import load_model
model, config = load_model('/path/to/pack')
```

Install `runtime/requirements.txt` on Apple Silicon. Ordinary MLX loaders do not apply the required transforms. Vision and MTP are not included.

`hadamard.json` uses the version-1 Prism contract accepted by MLX Swift's PrismHadamardConfiguration. Names refer to the saved MLX tensor namespace. GDN activations are already grouped in the bundled runtime; do not permute them again. Swift full-model loading still requires model integration; layer support alone is insufficient.

Reload validation checks serialization, not model quality or cross-runtime equivalence. Tokenizer validation checks vocabulary IDs and BPE merge ranks; it does not certify pre-tokenizer behavior. The chat template is copied from the source GGUF.
