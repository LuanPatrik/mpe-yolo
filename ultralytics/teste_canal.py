import torch
from ultralytics.nn.modules.mpe import VegIndexLayer

layer = VegIndexLayer()

x = torch.randn(1, 4, 256, 256)
y = layer(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
