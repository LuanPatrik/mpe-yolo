import torch

from ultralytics import YOLO

model = YOLO("models/mpe-yolov8.yaml")
x = torch.randn(1, 3, 256, 256).cuda()
y = model.model[0](x)
print("Após VegIndexLayer:", y.shape)
