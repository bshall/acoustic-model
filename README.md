# Acoustic-Model

The HuBERT-Soft and HuBERT-Discrete acoustic models for [soft-vc](https://github.com/bshall/soft-vc).

Relevant links:
- [Soft-VC repo](https://github.com/bshall/soft-vc)
- [Soft-VC paper]()

## Example Usage

```python
import torch
import numpy as np

# Load checkpoint
acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_soft").cuda()

# Load speech units
units = torch.from_numpy(np.load("path/to/units"))

# Generate mel-spectrogram from units
mel = acoustic.generate(units)
```