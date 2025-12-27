# Carrega o modelo do YAML e verifica se as saídas correspondem ao MPE-YOLO do paper
from ultralytics import YOLO
import torch

m = YOLO('models/mpe-yolov8.yaml').model
print('Model loaded')

# print layer strides if available
strides = []
for name, layer in m.named_modules():
  if hasattr(layer, 'stride'):
    s = layer.stride
    print(name, 'stride=', s)
