# 🧠 NeuralMind — Rede Neural do Zero

Demonstração educativa de **Redes Neurais Artificiais** implementadas em **NumPy puro** — sem PyTorch, TensorFlow ou Keras. Classifica flores Iris e visualiza o aprendizado em tempo real.

> Quarto projeto da série **ML Educativo**.

---

## Conceitos demonstrados

| Conceito | Descrição |
|---|---|
| **Forward Pass** | Propagação da entrada camada a camada: `z = Wx + b`, `a = f(z)` |
| **Backpropagation** | Gradiente do erro propagado de trás para frente pela regra da cadeia |
| **Gradient Descent** | `W = W - lr × dL/dW` — pesos ajustados para reduzir o erro |
| **Cross-Entropy Loss** | Função de perda para classificação multi-classe |
| **Softmax** | Converte saída da rede em probabilidades por classe |
| **ReLU / Sigmoid / Tanh** | Funções de ativação com comportamentos distintos |
| **Loss Landscape** | Superfície 3D do erro — o gradiente desce o vale |
| **Overfitting** | Observável quando val_loss sobe enquanto train_loss cai |

---

## Funcionalidades

- **Rede configurável** — 1 ou 2 camadas ocultas, neurônios, learning rate, épocas, ativação
- **SVG animado** — neurônios brilhando com intensidade proporcional à ativação, conexões com espessura = peso
- **Curvas de loss e acurácia** — treino vs. validação em tempo real (Plotly interativo)
- **Loss Landscape 3D** — superfície de perda interativa com ponto do mínimo atual
- **Navegação por snapshots** — veja os pesos em qualquer época do treinamento
- **Mapas de calor dos pesos** — visualização da magnitude e sinal de cada conexão
- **Fronteira de decisão** — projeção 2D do que a rede aprendeu
- **Código exposto** — implementação NumPy comentada na aba "Como Funciona"

---

## Como executar

```bash
git clone https://github.com/seu-usuario/neuralmind
cd neuralmind
pip install -r requirements.txt
streamlit run app.py
```

---

## Implementação

A rede foi construída **do zero com NumPy** — sem frameworks de deep learning:

```python
# Forward Pass
z = X @ W + b           # transformacao linear
a = np.maximum(0, z)    # ReLU

# Backpropagation
delta = y_pred - y_true                        # erro na saida
dW    = a_prev.T @ delta / n                   # gradiente dos pesos
W    -= lr * dW                                # gradient descent
```

---

## Stack

- Python 3.10+
- NumPy (rede neural)
- Streamlit (interface)
- Plotly (graficos 3D interativos)
- scikit-learn (apenas dataset e split)

---

## Série ML Educativo

| # | Projeto | Conceito |
|---|---|---|
| 1 | 🎮 Jogo da Velha | Q-Learning |
| 2 | 🌿 Diagnóstico de Plantas | Árvore de Decisão |
| 3 | 🚜 TractorMind | Regras de Associação (Apriori) |
| 4 | 🧠 NeuralMind | Rede Neural do Zero |

---

**Autor:** Victor  
**Licença:** MIT
