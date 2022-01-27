# EDEN's pytorch implementation

This sub-project contains EDEN's pytorch implementation. This code and precise experiments' code will be released as open-source with the camera-ready version.

### Usage example

```python
import torch
from eden import eden_builder

eden = eden_builder(bits=4)

x = torch.randn([2 ** 10])
encoded_x, context = eden.forward(x)
reconstructed_x, metrics = eden.backward(encoded_x, context)

# or just
reconstructed_x, metrics = eden.roundtrip(x)
```
