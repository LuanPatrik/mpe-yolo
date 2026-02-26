import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from ultralytics import RTDETR


class RTDETRClassTarget:
    def __init__(self, class_id):
        self.class_id = class_id

    def __call__(self, model_output):
        """model_output: saída do RT-DETR Retorna um ESCALAR.
        """
        # model_output[0]: scores do decoder
        # shape típico: [num_queries, num_classes]
        scores = model_output[0]

        # pega scores da classe desejada
        class_scores = scores[:, self.class_id]

        # retorna escalar
        return class_scores.mean()


model = RTDETR("../../resultado/runs/detect-detr/img/train/weights/best.pt")
model.model.eval()
model.model.requires_grad_(True)

# Identifica qual camada será utilizada
# for name, module in model.model.named_modules():
#    print(name)

target_layer = model.model.model[27].cv3

# Carregar imagem
img = cv2.imread("C:/Users/luanp/OneDrive/Documents/CNN-pesquisa/data/img/test/images/DJI_0606.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img_norm = img.astype(np.float32) / 255.0

input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

# Ativa gradiente
input_tensor = input_tensor.requires_grad_(True)

# Grad-CAM
cam = GradCAM(
    model=model.model,
    target_layers=[target_layer],
    # use_cuda= False torch.cuda.is_available()
)

num_classes = 3

for class_id in range(num_classes):
    # classe alvo
    targets = [RTDETRClassTarget(class_id)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Overlay
    visualization = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

    cv2.imwrite(f"heatmap_rtdetr_classe_{class_id}.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
