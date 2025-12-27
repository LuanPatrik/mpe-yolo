# MPE-YOLO

Este repositório contém uma **versão modificada do Ultralytics YOLO**, desenvolvida
para fins **acadêmicos e experimentais**, com foco em **detecção de pequenos objetos
em imagens aéreas**, especialmente imagens capturadas por VANTs (drones).

As modificações implementadas neste projeto são **baseadas e inspiradas** no artigo
científico que propõe o modelo **MPE-YOLO**, o qual apresenta melhorias arquiteturais
sobre o YOLOv8 para cenários de imagens aéreas.

---

## 📌 Origem do Projeto

Este projeto é **derivado do framework Ultralytics YOLO**, um sistema open-source
amplamente utilizado para detecção de objetos.

- Repositório original: https://github.com/ultralytics/ultralytics
- Licença original: **AGPL-3.0**

Todo o crédito pelo desenvolvimento do YOLO pertence à equipe da **Ultralytics**.

---

## 📚 Base Científica (Artigo de Referência)

As modificações realizadas neste repositório são **baseadas no seguinte artigo**:

**MPE-YOLO: Enhanced Small Target Detection in Aerial Imaging**  
Jia Su, Yichang Qin, Ze Jia, Ben Liang  
Hebei University of Science and Technology, China

O artigo propõe uma série de melhorias no YOLOv8 visando:
- Melhor representação de características de pequenos alvos
- Maior eficiência na fusão de características multiescala
- Manutenção de um modelo leve, adequado para aplicações em imagens aéreas

⚠️ **Observação importante**:  
Este repositório **NÃO é uma implementação oficial** do artigo.  
Trata-se de uma **implementação independente**, desenvolvida para fins de pesquisa,
avaliação e extensão das ideias propostas pelos autores.

---

## 🔧 Modificações Implementadas

Com base no artigo MPE-YOLO e em necessidades experimentais específicas, foram
realizadas as seguintes modificações:

- Ajustes na arquitetura do YOLOv8 para melhorar a detecção de pequenos objetos
- Alterações no pipeline de treinamento
- Implementação/adaptação de módulos inspirados no MPE-YOLO
- Inclusão de métricas adicionais de avaliação:
  - Precisão (Precision)
  - Revocação (Recall)
  - F1-score
  - mAP@50
- Adequações para experimentos com imagens aéreas e datasets específicos

Essas modificações foram desenvolvidas de forma **independente** e **não representam
alterações oficiais** do Ultralytics YOLO nem dos autores do artigo.

---

## ⚠️ Aviso Legal (Disclaimer)

- Este repositório **não é oficial** e **não possui afiliação** com a Ultralytics.
- Este repositório **não é uma implementação oficial** do artigo MPE-YOLO.
- O código é disponibilizado **exclusivamente para fins acadêmicos e experimentais**.

---

## 📄 Licença

Este projeto mantém a licença **AGPL-3.0**, herdada do Ultralytics YOLO.

O uso, modificação e redistribuição deste código devem obedecer aos termos dessa
licença.

---

## 📌 Citação

Se você utilizar este repositório em trabalhos acadêmicos, por favor cite:

1. **Ultralytics YOLO**
2. O artigo **MPE-YOLO: Enhanced Small Target Detection in Aerial Imaging**
3. Este repositório (se aplicável)

---

## 🎯 Objetivo do Projeto

Avaliar e aplicar melhorias arquiteturais no YOLO para **detecção de pequenos alvos
em imagens aéreas**, contribuindo para pesquisas em:
- Visão computacional
- Sensoriamento remoto
- Agricultura de precisão
- Monitoramento aéreo com VANTs
