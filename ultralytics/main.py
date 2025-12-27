import sys
import yaml
import os
from ultralytics import YOLO

# Receber o caminho da pasta via argumento do terminal
if len(sys.argv) > 1:
    caminho_pasta = sys.argv[1]
else:
    caminho_pasta = "default_run"
    print(f"Nenhum argumento passado, usando pasta padrão: {caminho_pasta}")

# Função para carregar o YAML e substituir variáveis
def load_and_replace_yaml_var(yaml_file, var_name, var_value):
    # Abrir o arquivo YAML original
    with open(yaml_file, 'r') as file:
        content = file.read()

    # Substituir a variável ${variavel} pelo valor passado via terminal
    content = content.replace(f"${{{var_name}}}", var_value)

    # Carregar o conteúdo YAML atualizado em memória
    config = yaml.safe_load(content)

    return config  # Retornar o dicionário com as substituições feitas

# Carregar e substituir variáveis no arquivo .yaml
yaml_file = 'custom_dataset.yaml'
config = load_and_replace_yaml_var(yaml_file, 'variavel', caminho_pasta)

# Escrever o arquivo YAML atualizado em um arquivo temporário
temp_yaml_file = 'custom_dataset_temp.yaml'
with open(temp_yaml_file, 'w') as file:
    yaml.dump(config, file)

out_result = "./runs/detect/" + caminho_pasta


# Carrega o mode - caminho do YAML do seu modelo
model = YOLO("models/mpe-yolov8.yaml")

#Treinar o modelo utilizando o novo arquivo .yaml
train_results = model.train(
    data="custom_dataset_temp.yaml",  #caminho para o dataset no arquivo YAML
    epochs=200,  # Número de épocas
    imgsz=640,  # Tamanho da imagem
    optimizer="AdamW",
    weight_decay=0.01, # Penaliza os pesos grandes
    lr0=0.0005,  # Define a taxa de aprendizado inicial
    #momentum=0.937,  # Configura o momentum para estabilizar as atualizações de gradiente
    batch=8, #Subconjunto menor da base de dados
    device="cuda",  # Usa CPU ou GPU
    workers=0, # Reduz o numero de workers do DataLoader
    cos_lr=True,     # decay cosseno, como no paper
    pretrained=True, # Pretreinado
    #augment=True,
    degrees= 10,
    translate= 0.1,
    scale= 0.5,
    shear= 0.0,
    fliplr= 0.5,
    flipud= 0.3,
    mosaic= 0.8,
    mixup= 0.3,
    hsv_h= 0.015,
    hsv_s= 0.3,
    hsv_v= 0.3,
    project=out_result, # Saída
)

#Avaliar o desempenho do modelo no conjunto de validação
model = YOLO(f'./runs/detect/{caminho_pasta}/train/weights/best.pt')

metric_valid = model.val(
    split='val',
    workers=0,
    project=out_result
)

metrics_test = model.val(
    split='test',
    workers=0,
    project=out_result,
)
