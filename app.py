"""
NeuralMind - Rede Neural do Zero
=================================
Demonstracao educativa de Redes Neurais implementadas em NumPy puro.
Classifica flores Iris e visualiza o aprendizado em tempo real.

Conceitos demonstrados:
- Forward Pass (propagacao direta)
- Backpropagation (retropropagacao do gradiente)
- Gradient Descent (descida do gradiente)
- Funcoes de ativacao: ReLU, Sigmoid, Tanh
- Loss landscape (superficie de perda 3D)
- Pesos e bias mudando a cada epoca
"""

import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGINA
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralMind - Rede Neural do Zero",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
:root{
    --bg:#07090f;--surface:#0d1117;--surface2:#12181f;--surface3:#171e27;
    --border:#1e2a38;--border2:#243344;
    --purple:#a78bfa;--purple2:#7c3aed;--purple3:#4c1d95;
    --cyan:#22d3ee;--cyan2:#0891b2;
    --pink:#f472b6;--pink2:#be185d;
    --green:#4ade80;--amber:#fbbf24;--red:#f87171;
    --text:#e2e8f0;--text2:#94a3b8;--muted:#475569;
}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;background:var(--bg);color:var(--text);}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;font-weight:700;}
code,.mono{font-family:'Space Mono',monospace;font-size:.82rem;}

.stButton>button{
    background:linear-gradient(135deg,var(--purple2),var(--cyan2));
    color:#fff;border:none;border-radius:6px;
    font-family:'Space Mono',monospace;font-size:.78rem;
    letter-spacing:.08em;padding:.55rem 1.4rem;
    transition:all .2s;font-weight:700;
    box-shadow:0 0 20px rgba(124,58,237,.3);
}
.stButton>button:hover{
    box-shadow:0 0 30px rgba(124,58,237,.5);
    transform:translateY(-1px);
}

.card{background:var(--surface);border:1px solid var(--border2);border-radius:8px;padding:1.1rem 1.3rem;}
.cp{background:var(--surface2);border:1px solid var(--purple2);border-left:3px solid var(--purple);border-radius:0 8px 8px 0;padding:.8rem 1rem;margin:.4rem 0;}
.cc{background:var(--surface2);border:1px solid var(--cyan2);border-left:3px solid var(--cyan);border-radius:0 8px 8px 0;padding:.8rem 1rem;margin:.4rem 0;}
.cg{background:var(--surface2);border:1px solid #166534;border-left:3px solid var(--green);border-radius:0 8px 8px 0;padding:.8rem 1rem;margin:.4rem 0;}
.ca{background:var(--surface2);border:1px solid #92400e;border-left:3px solid var(--amber);border-radius:0 8px 8px 0;padding:.8rem 1rem;margin:.4rem 0;}

.mb{background:var(--surface);border:1px solid var(--border2);border-radius:8px;padding:.9rem 1rem;text-align:center;position:relative;overflow:hidden;}
.mb::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--purple),var(--cyan));}
.ml{font-family:'Space Mono',monospace;font-size:.58rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:.2rem;}
.mv{font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700;line-height:1;}
.pv{color:var(--purple);}.cv{color:var(--cyan);}.gv{color:var(--green);}.av{color:var(--amber);}.rv{color:var(--red);}.pkv{color:var(--pink);}

.sl{font-family:'Space Mono',monospace;font-size:.58rem;letter-spacing:.2em;text-transform:uppercase;
    color:var(--muted);margin-bottom:.5rem;margin-top:1rem;padding-bottom:.3rem;border-bottom:1px solid var(--border);}

.epoch-row{display:flex;align-items:center;gap:.8rem;padding:.4rem .6rem;margin:.2rem 0;
           background:var(--surface2);border-radius:4px;font-family:'Space Mono',monospace;font-size:.72rem;}
.loss-bar{height:6px;border-radius:3px;background:linear-gradient(90deg,var(--purple),var(--cyan));margin-top:.15rem;}

.formula{background:var(--surface3);border:1px solid var(--border2);border-radius:6px;
         padding:.8rem 1rem;margin:.5rem 0;font-family:'Space Mono',monospace;font-size:.8rem;
         color:var(--cyan);line-height:1.7;}

.weight-chip{display:inline-block;padding:.12rem .5rem;border-radius:4px;
             font-family:'Space Mono',monospace;font-size:.68rem;margin:.1rem;
             background:rgba(167,139,250,.1);color:var(--purple);border:1px solid rgba(167,139,250,.3);}

[data-testid="stSidebar"]{background:var(--surface) !important;border-right:1px solid var(--border2);}
hr.dv{border:none;border-top:1px solid var(--border2);margin:1rem 0;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface) !important;border-bottom:1px solid var(--border2);gap:.5rem;}
.stTabs [data-baseweb="tab"]{font-family:'Space Mono',monospace !important;font-size:.72rem !important;
    letter-spacing:.08em !important;color:var(--muted) !important;background:transparent !important;border:none !important;padding:.6rem 1rem !important;}
.stTabs [aria-selected="true"]{color:var(--purple) !important;border-bottom:2px solid var(--purple) !important;}
.stTabs [data-baseweb="tab-panel"]{background:var(--bg) !important;padding-top:1.2rem !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# REDE NEURAL DO ZERO — NumPy puro
# ─────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    Rede Neural Multicamada implementada do zero com NumPy.
    Suporta camadas ocultas arbitrarias, multiplas ativacoes e SGD.
    """

    ACTIVATIONS = {
        "relu":    (lambda z: np.maximum(0, z),
                    lambda z: (z > 0).astype(float)),
        "sigmoid": (lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
                    lambda z: (sig := 1/(1+np.exp(-np.clip(z,-500,500)))) * (1-sig)),
        "tanh":    (lambda z: np.tanh(z),
                    lambda z: 1 - np.tanh(z)**2),
    }

    def __init__(self, layer_sizes, activation="relu", lr=0.01, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.activation  = activation
        self.lr          = lr
        self.weights     = []
        self.biases      = []
        self.history     = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], "weights_snapshot": []}

        # He initialization para ReLU, Xavier para sigmoid/tanh
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            if activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def _act(self, z):
        return self.ACTIVATIONS[self.activation][0](z)

    def _act_deriv(self, z):
        return self.ACTIVATIONS[self.activation][1](z)

    def _softmax(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass: propaga entrada pela rede camada a camada."""
        self._cache_z  = []   # pre-ativacoes
        self._cache_a  = [X]  # ativacoes (a[0] = entrada)

        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            self._cache_z.append(z)
            if i < len(self.weights) - 1:
                a = self._act(z)
            else:
                a = self._softmax(z)   # camada de saida: softmax
            self._cache_a.append(a)

        return self._cache_a[-1]

    def _cross_entropy(self, y_pred, y_true):
        """Cross-entropy loss para classificacao multi-classe."""
        n = y_true.shape[0]
        log_p = np.log(y_pred + 1e-15)
        return -np.sum(y_true * log_p) / n

    def backward(self, y_true):
        """
        Backpropagation: calcula gradientes de cada peso usando a regra da cadeia.
        dL/dW = dL/da * da/dz * dz/dW
        """
        n = y_true.shape[0]
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Gradiente da saida (softmax + cross-entropy = simplificado)
        delta = self._cache_a[-1] - y_true   # shape: (n, n_classes)

        for i in reversed(range(len(self.weights))):
            a_prev = self._cache_a[i]
            grads_w[i] = (a_prev.T @ delta) / n
            grads_b[i] = delta.mean(axis=0, keepdims=True)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self._act_deriv(self._cache_z[i-1])

        # SGD update
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i]  -= self.lr * grads_b[i]

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == np.argmax(y, axis=1))

    def train(self, X_train, y_train, X_val, y_val, epochs=100, snapshot_every=10,
              early_stopping=False, patience=10):
        """Treina a rede e salva historico para visualizacao."""
        self.history = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], "weights_snapshot": []}
        best_val_loss   = float("inf")
        no_improve_cnt  = 0

        for ep in range(1, epochs + 1):
            # Mini-batch shuffle
            idx = np.random.permutation(len(X_train))
            batch_size = min(32, len(X_train))
            total_loss = 0.0
            n_batches  = 0

            for start in range(0, len(X_train), batch_size):
                xb = X_train[idx[start:start+batch_size]]
                yb = y_train[idx[start:start+batch_size]]
                out = self.forward(xb)
                total_loss += self._cross_entropy(out, yb)
                self.backward(yb)
                n_batches += 1

            loss     = total_loss / n_batches
            val_out  = self.forward(X_val)
            val_loss = self._cross_entropy(val_out, y_val)
            acc      = self.accuracy(X_train, y_train)
            val_acc  = self.accuracy(X_val, y_val)

            self.history["loss"].append(float(loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["acc"].append(float(acc))
            self.history["val_acc"].append(float(val_acc))

            if ep % snapshot_every == 0 or ep == 1:
                self.history["weights_snapshot"].append({
                    "epoch": ep,
                    "weights": [W.copy() for W in self.weights],
                    "biases":  [b.copy() for b in self.biases],
                })

            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss  = val_loss
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
                if no_improve_cnt >= patience:
                    break

        return self.history


# ─────────────────────────────────────────────────────────
# DATASET IRIS
# ─────────────────────────────────────────────────────────

@st.cache_data
def load_iris_data():
    iris  = load_iris()
    X, y  = iris.data, iris.target
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # One-hot encoding
    n_classes = 3
    y_oh = np.eye(n_classes)[y]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_oh, test_size=0.2, random_state=42, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, iris.feature_names, iris.target_names, X_sc, y


# ─────────────────────────────────────────────────────────
# SVG DA REDE NEURAL
# ─────────────────────────────────────────────────────────

def network_svg(layer_sizes, weights=None, activations_val=None, highlight_path=False):
    """
    Renderiza SVG da arquitetura da rede neural.
    - Espessura das conexoes proporcional ao peso
    - Cor dos neuronios proporcional a ativacao
    - Pesos positivos = roxo, negativos = vermelho
    """
    W_px   = 780
    H_px   = 380
    margin = 60
    n_layers = len(layer_sizes)

    # posicoes X de cada camada
    x_pos = [margin + i * (W_px - 2*margin) / (n_layers-1) for i in range(n_layers)]

    # max neurons por camada para espacar Y
    max_n  = max(layer_sizes)
    node_r = min(18, (H_px - 80) / (max_n * 2.5))

    def y_pos(layer_idx, neuron_idx):
        n = layer_sizes[layer_idx]
        total_h = (n - 1) * node_r * 2.8
        start_y = H_px/2 - total_h/2
        return start_y + neuron_idx * node_r * 2.8

    svg_lines = [
        f'<svg viewBox="0 0 {W_px} {H_px}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;display:block;border-radius:8px">',
        '<defs>',
        '<style>',
        '.nlabel{font-family:Space Mono,monospace;font-size:9px;fill:#475569;}',
        '.llabel{font-family:Space Grotesk,sans-serif;font-size:11px;font-weight:600;}',
        '.pulse{animation:pulse 1.5s ease-in-out infinite;}',
        '@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}',
        '</style>',
        '<radialGradient id="ng" cx="35%" cy="35%">',
        '  <stop offset="0%" stop-color="#c4b5fd"/>',
        '  <stop offset="100%" stop-color="#7c3aed"/>',
        '</radialGradient>',
        '<radialGradient id="ni" cx="35%" cy="35%">',
        '  <stop offset="0%" stop-color="#67e8f9"/>',
        '  <stop offset="100%" stop-color="#0891b2"/>',
        '</radialGradient>',
        '<radialGradient id="no" cx="35%" cy="35%">',
        '  <stop offset="0%" stop-color="#86efac"/>',
        '  <stop offset="100%" stop-color="#16a34a"/>',
        '</radialGradient>',
        '</defs>',
        f'<rect width="{W_px}" height="{H_px}" fill="#07090f" rx="8"/>',
    ]

    # Conexoes (pesos)
    if weights is not None:
        for li in range(n_layers - 1):
            W = weights[li]
            w_abs_max = np.abs(W).max() + 1e-8
            for ni in range(layer_sizes[li]):
                for nj in range(layer_sizes[li+1]):
                    x1, y1 = x_pos[li],   y_pos(li,   ni)
                    x2, y2 = x_pos[li+1], y_pos(li+1, nj)
                    w_val  = W[ni, nj]
                    opacity = 0.08 + 0.55 * abs(w_val) / w_abs_max
                    lw      = 0.4 + 2.5 * abs(w_val) / w_abs_max
                    color   = "#a78bfa" if w_val >= 0 else "#f87171"
                    svg_lines.append(
                        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
                        f' stroke="{color}" stroke-width="{lw:.2f}" opacity="{opacity:.2f}"/>'
                    )
    else:
        # Sem pesos: linhas cinzas uniformes
        for li in range(n_layers - 1):
            for ni in range(layer_sizes[li]):
                for nj in range(layer_sizes[li+1]):
                    x1, y1 = x_pos[li],   y_pos(li,   ni)
                    x2, y2 = x_pos[li+1], y_pos(li+1, nj)
                    svg_lines.append(
                        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
                        f' stroke="#1e2a38" stroke-width="0.8" opacity="0.6"/>'
                    )

    # Neuronios
    for li, n_neurons in enumerate(layer_sizes):
        grad_id = "ni" if li == 0 else "no" if li == n_layers-1 else "ng"
        layer_label = (
            "Entrada" if li == 0
            else "Saida" if li == n_layers-1
            else f"Oculta {li}"
        )

        for ni in range(n_neurons):
            cx, cy = x_pos[li], y_pos(li, ni)

            # brilho baseado na ativacao
            if activations_val and li < len(activations_val):
                a_row = activations_val[li]
                act_v = float(np.mean(a_row[:, ni])) if ni < a_row.shape[1] else 0.0
                act_v = max(0.0, min(1.0, abs(act_v)))
                glow_r  = int(12 + act_v * 16)
                glow_op = 0.15 + act_v * 0.45
            else:
                glow_r  = 14
                glow_op = 0.2

            # halo
            svg_lines.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{node_r + glow_r}" '
                f'fill="url(#{grad_id})" opacity="{glow_op:.2f}"/>'
            )
            # corpo
            svg_lines.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{node_r:.1f}" '
                f'fill="url(#{grad_id})" stroke="#0d1117" stroke-width="1.5"/>'
            )
            # indice
            svg_lines.append(
                f'<text x="{cx:.1f}" y="{cy+3.5:.1f}" text-anchor="middle" '
                f'class="nlabel" style="font-size:{min(9,node_r*0.7):.0f}px">{ni+1}</text>'
            )

        # label da camada
        label_y = y_pos(li, n_neurons-1) + node_r + 22
        color_l = "#22d3ee" if li == 0 else "#4ade80" if li == n_layers-1 else "#a78bfa"
        svg_lines.append(
            f'<text x="{x_pos[li]:.1f}" y="{label_y:.1f}" text-anchor="middle" '
            f'class="llabel" style="fill:{color_l};font-size:10px">{layer_label}</text>'
        )
        svg_lines.append(
            f'<text x="{x_pos[li]:.1f}" y="{label_y+14:.1f}" text-anchor="middle" '
            f'class="nlabel">{n_neurons} neuronio{"s" if n_neurons!=1 else ""}</text>'
        )

    # legenda de pesos
    if weights is not None:
        svg_lines += [
            f'<line x1="{W_px-120}" y1="18" x2="{W_px-90}" y2="18" stroke="#a78bfa" stroke-width="2"/>',
            f'<text x="{W_px-84}" y="22" class="nlabel" style="fill:#a78bfa">peso +</text>',
            f'<line x1="{W_px-120}" y1="34" x2="{W_px-90}" y2="34" stroke="#f87171" stroke-width="2"/>',
            f'<text x="{W_px-84}" y="38" class="nlabel" style="fill:#f87171">peso -</text>',
        ]

    svg_lines.append('</svg>')
    return "\n".join(svg_lines)


# ─────────────────────────────────────────────────────────
# GRAFICOS PLOTLY
# ─────────────────────────────────────────────────────────

PLOTLY_DARK = dict(
    paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
    font=dict(family="Space Mono", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#1e2a38", linecolor="#243344", zerolinecolor="#1e2a38"),
    yaxis=dict(gridcolor="#1e2a38", linecolor="#243344", zerolinecolor="#1e2a38"),
)


def plot_loss_curves(history):
    eps  = list(range(1, len(history["loss"])+1))
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=history["loss"],     name="Loss (treino)",
                             line=dict(color="#a78bfa", width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(x=eps, y=history["val_loss"], name="Loss (validacao)",
                             line=dict(color="#f472b6", width=2.5, dash="dot"), mode="lines"))
    fig.update_layout(**PLOTLY_DARK, title="Curva de Loss",
                      xaxis_title="Epoca", yaxis_title="Cross-Entropy Loss",
                      legend=dict(bgcolor="#0d1117", bordercolor="#1e2a38"),
                      height=300, margin=dict(t=40,b=30,l=40,r=20))
    return fig


def plot_acc_curves(history):
    eps = list(range(1, len(history["acc"])+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=[a*100 for a in history["acc"]],
                             name="Acuracia (treino)", line=dict(color="#22d3ee", width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(x=eps, y=[a*100 for a in history["val_acc"]],
                             name="Acuracia (validacao)", line=dict(color="#4ade80", width=2.5, dash="dot"), mode="lines"))
    fig.update_layout(**PLOTLY_DARK, title="Curva de Acuracia",
                      xaxis_title="Epoca", yaxis_title="Acuracia (%)",
                      yaxis_range=[0,105],
                      legend=dict(bgcolor="#0d1117", bordercolor="#1e2a38"),
                      height=300, margin=dict(t=40,b=30,l=40,r=20))
    return fig


def plot_loss_landscape(nn, X_tr, y_tr):
    """
    Projeta 2 direcoes aleatorias no espaco de parametros e plota
    a superficie de perda 3D ao redor dos pesos finais.
    """
    np.random.seed(7)

    # Achata todos os pesos num vetor 1D
    flat_w = np.concatenate([W.flatten() for W in nn.weights] +
                             [b.flatten() for b in nn.biases])
    n_params = len(flat_w)

    # Duas direcoes ortogonais aleatorias (normalizadas)
    d1 = np.random.randn(n_params); d1 /= np.linalg.norm(d1)
    d2 = np.random.randn(n_params); d2 -= d2.dot(d1)*d1; d2 /= np.linalg.norm(d2)

    # Grade de pontos ao redor do minimo
    scale  = 2.5
    n_pts  = 30
    alphas = np.linspace(-scale, scale, n_pts)
    betas  = np.linspace(-scale, scale, n_pts)
    Z      = np.zeros((n_pts, n_pts))

    # Formas dos tensores para restaurar
    shapes_w = [W.shape for W in nn.weights]
    shapes_b = [b.shape for b in nn.biases]

    def set_weights(w_flat):
        cursor = 0
        for i, sh in enumerate(shapes_w):
            sz = int(np.prod(sh))
            nn.weights[i] = w_flat[cursor:cursor+sz].reshape(sh)
            cursor += sz
        for i, sh in enumerate(shapes_b):
            sz = int(np.prod(sh))
            nn.biases[i] = w_flat[cursor:cursor+sz].reshape(sh)
            cursor += sz

    for i, a in enumerate(alphas):
        for j, b_val in enumerate(betas):
            w_perturbed = flat_w + a*d1 + b_val*d2
            set_weights(w_perturbed)
            out  = nn.forward(X_tr)
            loss = nn._cross_entropy(out, y_tr)
            Z[j, i] = float(np.clip(loss, 0, 10))

    # Restaura pesos originais
    set_weights(flat_w)

    fig = go.Figure(data=[
        go.Surface(
            z=Z, x=alphas, y=betas,
            colorscale=[[0,"#7c3aed"],[0.3,"#a78bfa"],[0.6,"#f472b6"],[1,"#f87171"]],
            opacity=0.88,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
            showscale=True,
            colorbar=dict(title=dict(text="Loss",font=dict(color="#94a3b8")),
                          tickfont=dict(color="#94a3b8")),
        ),
        # Ponto do minimo (centro)
        go.Scatter3d(
            x=[0], y=[0], z=[float(Z[n_pts//2, n_pts//2])],
            mode="markers+text",
            marker=dict(size=8, color="#4ade80", symbol="diamond"),
            text=["Minimo atual"],
            textfont=dict(color="#4ade80", size=11),
            textposition="top center",
            name="Pesos atuais",
        )
    ])
    fig.update_layout(
        paper_bgcolor="#07090f",
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(title=dict(text="Direcao 1", font=dict(color="#94a3b8")),
                       gridcolor="#1e2a38", linecolor="#243344", tickfont=dict(color="#475569")),
            yaxis=dict(title=dict(text="Direcao 2", font=dict(color="#94a3b8")),
                       gridcolor="#1e2a38", linecolor="#243344", tickfont=dict(color="#475569")),
            zaxis=dict(title=dict(text="Loss",      font=dict(color="#94a3b8")),
                       gridcolor="#1e2a38", linecolor="#243344", tickfont=dict(color="#475569")),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        ),
        title=dict(text="Loss Landscape — Superficie de Perda 3D",
                   font=dict(family="Space Grotesk", color="#e2e8f0", size=14)),
        height=520,
        margin=dict(t=50, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(bgcolor="#0d1117", bordercolor="#1e2a38",
                    font=dict(color="#94a3b8")),
    )
    return fig


def plot_decision_boundary(nn, X_sc, y_raw, feature_names):
    """Projeta fronteira de decisao nos 2 primeiros features."""
    x_min, x_max = X_sc[:,0].min()-1, X_sc[:,0].max()+1
    y_min, y_max = X_sc[:,1].min()-1, X_sc[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel(),
                 np.zeros(xx.size), np.zeros(xx.size)]
    Z = nn.predict(grid).reshape(xx.shape)

    colors_bg   = ["rgba(124,58,237,.15)","rgba(34,211,238,.15)","rgba(74,222,128,.15)"]
    colors_pt   = ["#a78bfa","#22d3ee","#4ade80"]
    class_names = ["Setosa","Versicolor","Virginica"]

    fig = go.Figure()
    for cls in range(3):
        mask = Z == cls
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=(Z == cls).astype(int),
            showscale=False, showlegend=False,
            colorscale=[[0,"rgba(0,0,0,0)"],[1,colors_bg[cls]]],
            contours=dict(start=0.5, end=0.5),
        ))
    for cls in range(3):
        mask = y_raw == cls
        fig.add_trace(go.Scatter(
            x=X_sc[mask, 0], y=X_sc[mask, 1],
            mode="markers", name=class_names[cls],
            marker=dict(color=colors_pt[cls], size=7,
                        line=dict(color="#0d1117", width=1)),
        ))
    fig.update_layout(**PLOTLY_DARK,
                      title="Fronteira de Decisao (features 1 e 2)",
                      xaxis_title=feature_names[0],
                      yaxis_title=feature_names[1],
                      legend=dict(bgcolor="#0d1117", bordercolor="#1e2a38"),
                      height=380, margin=dict(t=40,b=30,l=40,r=20))
    return fig


def plot_weight_heatmap(weights, layer_idx):
    W = weights[layer_idx]
    fig = go.Figure(go.Heatmap(
        z=W, colorscale=[[0,"#f87171"],[0.5,"#1e2a38"],[1,"#a78bfa"]],
        zmid=0, showscale=True,
        colorbar=dict(tickfont=dict(color="#94a3b8")),
    ))
    fig.update_layout(**PLOTLY_DARK,
                      title=f"Mapa de Pesos — Camada {layer_idx+1}",
                      xaxis_title="Neuronio destino",
                      yaxis_title="Neuronio origem",
                      height=320, margin=dict(t=40,b=30,l=40,r=20))
    return fig


def plot_confusion(nn, X_te, y_te_oh, class_names):
    y_pred = nn.predict(X_te)
    y_true = np.argmax(y_te_oh, axis=1)
    cm = np.zeros((3,3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig = go.Figure(go.Heatmap(
        z=cm, x=list(class_names), y=list(class_names),
        colorscale=[[0,"#0d1117"],[1,"#7c3aed"]],
        showscale=False, text=cm, texttemplate="%{text}",
        textfont=dict(size=16, color="#e2e8f0"),
    ))
    fig.update_layout(**PLOTLY_DARK, title="Matriz de Confusao",
                      xaxis_title="Previsto", yaxis_title="Real",
                      height=320, margin=dict(t=40,b=30,l=50,r=20))
    return fig, cm


def class_metrics_table(cm, class_names):
    """Retorna DataFrame com Precisao, Recall e F1 por classe."""
    rows = []
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        rows.append({
            "Classe":    cls.capitalize(),
            "Precisao":  f"{prec*100:.1f}%",
            "Recall":    f"{recall*100:.1f}%",
            "F1-Score":  f"{f1*100:.1f}%",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────

def init():
    if "nn"         not in st.session_state: st.session_state.nn         = None
    if "history"    not in st.session_state: st.session_state.history    = None
    if "trained"    not in st.session_state: st.session_state.trained    = False
    if "snap_idx"   not in st.session_state: st.session_state.snap_idx   = 0
    if "n2"         not in st.session_state: st.session_state["n2"]      = 7
    if "active_tab" not in st.session_state: st.session_state.active_tab = 0


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    init()
    X_tr, X_te, y_tr, y_te, feat_names, class_names, X_sc, y_raw = load_iris_data()

    # ── SIDEBAR ─────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-family:Space Grotesk,sans-serif;font-size:1.3rem;font-weight:700;'
            'background:linear-gradient(90deg,#a78bfa,#22d3ee);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;letter-spacing:.05em">&#129504; NeuralMind</div>'
            '<div style="font-family:Space Mono,monospace;font-size:.6rem;'
            'color:#475569;letter-spacing:.15em;margin-bottom:1rem">REDE NEURAL DO ZERO</div>',
            unsafe_allow_html=True
        )

        st.markdown('<p class="sl">Arquitetura da rede</p>', unsafe_allow_html=True)
        hidden1 = st.slider("Neuronios — camada oculta 1", 2, 16, 8)
        add_layer2 = st.checkbox("Adicionar 2a camada oculta", value=False)
        hidden2 = 0
        if add_layer2:
            hidden2 = st.slider("Neuronios — camada oculta 2", 2, 16,
                                 st.session_state["n2"], key="n2")

        st.markdown('<p class="sl">Hiperparametros</p>', unsafe_allow_html=True)
        lr        = st.select_slider("Learning rate", [0.001,0.005,0.01,0.05,0.1,0.3,0.5], value=0.01)
        if lr >= 0.3:
            st.warning("Taxa alta pode causar instabilidade no treino.")
        elif lr <= 0.005:
            st.info("Taxa baixa pode deixar o treino muito lento.")
        epochs    = st.slider("Epocas de treino", 50, 500, 200, 50)
        activation = st.radio("Funcao de ativacao", ["relu","sigmoid","tanh"], horizontal=True)

        st.markdown('<p class="sl">Early Stopping</p>', unsafe_allow_html=True)
        use_early_stop = st.checkbox("Ativar Early Stopping", value=False)
        es_patience = 10
        if use_early_stop:
            es_patience = st.slider("Paciencia (epocas sem melhora)", 3, 50, 10)

        st.markdown('<hr class="dv">', unsafe_allow_html=True)
        train_btn = st.button("Treinar rede agora", use_container_width=True)

        if st.session_state.trained and st.session_state.history:
            h = st.session_state.history
            st.markdown('<p class="sl">Resultado do treino</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="card">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:.4rem">'
                f'<span style="font-family:Space Mono,monospace;font-size:.62rem;color:var(--muted)">Perda Final</span>'
                f'<span style="font-family:Space Mono,monospace;color:var(--purple)">{h["loss"][-1]:.4f}</span></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:.4rem">'
                f'<span style="font-family:Space Mono,monospace;font-size:.62rem;color:var(--muted)">Val loss</span>'
                f'<span style="font-family:Space Mono,monospace;color:var(--pink)">{h["val_loss"][-1]:.4f}</span></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:.4rem">'
                f'<span style="font-family:Space Mono,monospace;font-size:.62rem;color:var(--muted)">Acuracia treino</span>'
                f'<span style="font-family:Space Mono,monospace;color:var(--cyan)">{h["acc"][-1]*100:.1f}%</span></div>'
                f'<div style="display:flex;justify-content:space-between">'
                f'<span style="font-family:Space Mono,monospace;font-size:.62rem;color:var(--muted)">Acuracia val</span>'
                f'<span style="font-family:Space Mono,monospace;color:var(--green)">{h["val_acc"][-1]*100:.1f}%</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Botao exportar modelo
            st.markdown('<p class="sl">Exportar modelo</p>', unsafe_allow_html=True)
            _nn_exp = st.session_state.nn
            _buf = io.BytesIO()
            np.savez(
                _buf,
                **{f"W{i}": W for i, W in enumerate(_nn_exp.weights)},
                **{f"b{i}": b for i, b in enumerate(_nn_exp.biases)},
            )
            st.download_button(
                "Baixar pesos (.npz)",
                data=_buf.getvalue(),
                file_name="neuralmind_weights.npz",
                mime="application/octet-stream",
                use_container_width=True,
            )

        st.markdown('<hr class="dv">', unsafe_allow_html=True)
        st.markdown('<p class="sl">Conceitos</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="cp" style="margin-bottom:.5rem"><b style="color:#a78bfa">Forward Pass</b><br>'
            '<span style="font-size:.8rem">Entrada flui camada a camada: z = Wx + b, a = f(z)</span></div>'
            '<div class="cc" style="margin-bottom:.5rem"><b style="color:#22d3ee">Backpropagation</b><br>'
            '<span style="font-size:.8rem">Gradiente do erro flui de tras pra frente pela regra da cadeia</span></div>'
            '<div class="ca" style="margin-bottom:.5rem"><b style="color:#fbbf24">Gradient Descent</b><br>'
            '<span style="font-size:.8rem">W = W - lr x dL/dW — pesos se ajustam para reduzir o erro</span></div>'
            '<div class="cg"><b style="color:#4ade80">Loss Landscape</b><br>'
            '<span style="font-size:.8rem">Superficie 3D do erro em funcao dos pesos — o gradiente desce o vale</span></div>',
            unsafe_allow_html=True
        )

    # Monta layer sizes
    layer_sizes = [4, hidden1] + ([hidden2] if add_layer2 and hidden2 > 0 else []) + [3]

    # Treina se botao clicado
    if train_btn:
        st.session_state.active_tab = 0
        with st.spinner("Treinando rede neural..."):
            nn = NeuralNetwork(layer_sizes, activation=activation, lr=lr)
            history = nn.train(
                X_tr, y_tr, X_te, y_te,
                epochs=epochs,
                snapshot_every=max(1, epochs // 10),
                early_stopping=use_early_stop,
                patience=es_patience,
            )
        st.session_state.nn       = nn
        st.session_state.history  = history
        st.session_state.trained  = True
        st.session_state.snap_idx = len(history["weights_snapshot"]) - 1
        st.rerun()

    # ── HEADER ──────────────────────────────────────────
    st.markdown(
        '<h1 style="font-size:2rem;margin-bottom:0">'
        '<span style="background:linear-gradient(90deg,#a78bfa,#22d3ee);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent">NeuralMind</span>'
        ' <span style="color:#475569;font-size:1rem;font-family:Space Mono,monospace">/ Rede Neural do Zero</span></h1>'
        '<p style="color:#475569;font-family:Space Mono,monospace;font-size:.68rem;'
        'letter-spacing:.14em;margin-top:.2rem;margin-bottom:1rem">'
        'NUMPY PURO — FORWARD PASS — BACKPROPAGATION — GRADIENT DESCENT — IRIS DATASET</p>',
        unsafe_allow_html=True
    )

    # ── METRICAS ────────────────────────────────────────
    h = st.session_state.history
    n_params = sum(W.size + b.size for W, b in zip(
        st.session_state.nn.weights, st.session_state.nn.biases
    )) if st.session_state.trained else sum(
        layer_sizes[i]*layer_sizes[i+1] + layer_sizes[i+1]
        for i in range(len(layer_sizes)-1)
    )
    vals = [
        ("Arquitetura",  " x ".join(map(str,layer_sizes)), "pv"),
        ("Parametros",   n_params,                          "cv"),
        ("Acur. treino", f"{h['acc'][-1]*100:.1f}%" if h else "--", "gv"),
        ("Acur. val",    f"{h['val_acc'][-1]*100:.1f}%" if h else "--", "av"),
        ("Perda Final",  f"{h['loss'][-1]:.4f}"       if h else "--", "pkv"),
    ]
    for col_w, (lbl, val, cls) in zip(st.columns(5), vals):
        with col_w:
            st.markdown(f'<div class="mb"><div class="ml">{lbl}</div><div class="mv {cls}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.trained:
        st.markdown(
            '<div class="cp" style="text-align:center;padding:2rem">'
            '<div style="font-size:3rem">&#129504;</div>'
            '<div style="font-family:Space Grotesk,sans-serif;font-size:1.1rem;font-weight:600;'
            'color:#a78bfa;margin:.5rem 0">Configure e treine a rede no painel lateral</div>'
            '<div style="font-size:.9rem;color:#475569">Ajuste a arquitetura, o learning rate e as epocas<br>'
            'depois clique em <b style="color:#a78bfa">Treinar rede agora</b></div></div>',
            unsafe_allow_html=True
        )

        # Mostra SVG sem pesos como preview
        st.markdown('<p class="sl" style="margin-top:1.5rem">Preview da arquitetura atual</p>', unsafe_allow_html=True)
        st.markdown(network_svg(layer_sizes), unsafe_allow_html=True)
        return

    nn = st.session_state.nn
    h  = st.session_state.history

    # ── TABS ────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "TREINO E CURVAS", "ARQUITETURA E PESOS", "LOSS LANDSCAPE", "COMO FUNCIONA"
    ])

    # ════════════════════════════════════════════════════
    # TAB 1: TREINO E CURVAS
    # ════════════════════════════════════════════════════
    with tab1:
        c1a, c1b = st.columns(2, gap="large")
        with c1a:
            st.plotly_chart(plot_loss_curves(h), use_container_width=True)
        with c1b:
            st.plotly_chart(plot_acc_curves(h), use_container_width=True)

        c1c, c1d = st.columns(2, gap="large")
        with c1c:
            fig_cm, cm_arr = plot_confusion(nn, X_te, y_te, class_names)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown('<p class="sl" style="margin-top:.2rem">Metricas por classe</p>', unsafe_allow_html=True)
            st.dataframe(class_metrics_table(cm_arr, class_names), hide_index=True, use_container_width=True)
        with c1d:
            st.plotly_chart(plot_decision_boundary(nn, X_sc, y_raw, feat_names), use_container_width=True)

        # Tabela de historico por epoca
        st.markdown('<p class="sl" style="margin-top:.5rem">Log de treinamento por epoca</p>', unsafe_allow_html=True)
        loss_max = max(h["loss"]) + 1e-8
        log_rows = []
        for ep in range(len(h["loss"])):
            bar_w = int(200 * h["loss"][ep] / loss_max)
            color = "#a78bfa" if ep < len(h["loss"])*0.5 else "#22d3ee" if ep < len(h["loss"])*0.85 else "#4ade80"
            log_rows.append(
                f'<div class="epoch-row">'
                f'<span style="color:#475569;min-width:55px">ep {ep+1:03d}</span>'
                f'<div style="flex:1">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:2px">'
                f'<span style="color:#a78bfa">Perda {h["loss"][ep]:.4f}</span>'
                f'<span style="color:#4ade80">Acur. {h["acc"][ep]*100:.1f}%</span>'
                f'<span style="color:#f472b6">Val {h["val_acc"][ep]*100:.1f}%</span>'
                f'</div>'
                f'<div style="background:#1e2a38;border-radius:3px;height:5px">'
                f'<div class="loss-bar" style="width:{bar_w}px;background:{color}"></div>'
                f'</div></div></div>'
            )

        with st.expander(f"Ver todas as {len(h['loss'])} epocas", expanded=False):
            st.markdown("".join(log_rows), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # TAB 2: ARQUITETURA E PESOS
    # ════════════════════════════════════════════════════
    with tab2:
        snaps = h["weights_snapshot"]
        if len(snaps) > 1:
            snap_ep = [s["epoch"] for s in snaps]
            st.markdown('<p class="sl">Navegar pela historia de pesos</p>', unsafe_allow_html=True)
            chosen_ep = st.select_slider(
                "Epoca", options=snap_ep, value=snap_ep[-1], label_visibility="collapsed"
            )
            st.write(f"Visualizando pesos da época: **{chosen_ep}** / {snap_ep[-1]}")
            snap = snaps[snap_ep.index(chosen_ep)]
        else:
            snap = snaps[-1]

        weights_now = snap["weights"]

        # Calcula ativacoes de uma amostra para colorir neuronios
        nn_temp = NeuralNetwork(layer_sizes, activation=activation, lr=lr)
        nn_temp.weights = weights_now
        nn_temp.biases  = snap["biases"]
        sample_x = X_tr[:8]
        nn_temp.forward(sample_x)
        act_vals = [a for a in nn_temp._cache_a]  # lista de ativacoes por camada

        st.markdown('<p class="sl">Visualizacao da rede</p>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:.7rem;color:#475569;margin-bottom:.5rem">'
            'Espessura da conexao = magnitude do peso &nbsp;|&nbsp; '
            'Roxo = peso positivo &nbsp;|&nbsp; Vermelho = peso negativo &nbsp;|&nbsp; '
            'Brilho do neuronio = ativacao media</div>',
            unsafe_allow_html=True
        )
        st.markdown(network_svg(layer_sizes, weights_now, act_vals), unsafe_allow_html=True)

        st.markdown('<p class="sl" style="margin-top:1.2rem">Mapas de calor dos pesos</p>', unsafe_allow_html=True)
        hm_cols = st.columns(len(weights_now))
        for i, col_w in enumerate(hm_cols):
            with col_w:
                st.plotly_chart(plot_weight_heatmap(weights_now, i), use_container_width=True)

        # Estatisticas dos pesos
        st.markdown('<p class="sl">Estatisticas dos pesos por camada</p>', unsafe_allow_html=True)
        stats_rows = []
        for i, W in enumerate(weights_now):
            stats_rows.append({
                "Camada": f"W{i+1} ({W.shape[0]}x{W.shape[1]})",
                "Media":  f"{W.mean():.4f}",
                "Desvio": f"{W.std():.4f}",
                "Min":    f"{W.min():.4f}",
                "Max":    f"{W.max():.4f}",
                "Norma":  f"{np.linalg.norm(W):.4f}",
            })
        st.dataframe(pd.DataFrame(stats_rows), hide_index=True, use_container_width=True)

        # ── Passo a passo do forward pass ──────────────
        st.markdown('<p class="sl" style="margin-top:1.2rem">Passo a passo — Forward Pass</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="cc" style="margin-bottom:.8rem;font-size:.84rem">'
            'Insira valores para as 4 features do Iris e veja as ativacoes camada a camada.</div>',
            unsafe_allow_html=True,
        )
        feat_labels = ["Comp. Sepala (cm)", "Larg. Sepala (cm)", "Comp. Petala (cm)", "Larg. Petala (cm)"]
        fp_cols = st.columns(4)
        fp_vals = []
        for fi, (fc, fl) in enumerate(zip(fp_cols, feat_labels)):
            with fc:
                fp_vals.append(st.number_input(fl, value=0.0, step=0.1, format="%.2f", key=f"fp_{fi}"))
        sample_input = np.array(fp_vals).reshape(1, -1)
        nn.forward(sample_input)
        act_layers = nn._cache_a  # lista: entrada + saidas de cada camada
        act_names  = (["Entrada"] +
                      [f"Oculta {i+1}" for i in range(len(layer_sizes) - 2)] +
                      ["Saida (Softmax)"])
        class_labels = ["setosa", "versicolor", "virginica"]
        for act, aname in zip(act_layers, act_names):
            vals = act[0]
            chips = " ".join(
                f'<span class="weight-chip">{v:+.3f}</span>' for v in vals
            )
            st.markdown(
                f'<div class="epoch-row" style="flex-wrap:wrap;gap:.4rem">'
                f'<span style="color:#475569;min-width:90px;font-size:.68rem">{aname}</span>'
                f'{chips}</div>',
                unsafe_allow_html=True,
            )
        pred_idx  = int(np.argmax(act_layers[-1][0]))
        pred_prob = float(act_layers[-1][0][pred_idx]) * 100
        st.markdown(
            f'<div class="cg" style="margin-top:.6rem">'
            f'Predicao: <b style="color:#4ade80">{class_labels[pred_idx]}</b> '
            f'— confianca <b style="color:#4ade80">{pred_prob:.1f}%</b></div>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════
    # TAB 3: LOSS LANDSCAPE
    # ════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            '<div class="ca" style="margin-bottom:1rem">'
            '<b style="color:#fbbf24">O que e o Loss Landscape?</b><br>'
            '<span style="font-size:.88rem">Cada ponto da superficie representa os pesos da rede em uma posicao diferente '
            'do espaco de parametros. O vale mais fundo e o minimo — onde o erro e menor. '
            'O treinamento move os pesos descendo esse terreno pelo gradiente. '
            'O ponto verde marca onde os pesos atuais estao posicionados.</span></div>',
            unsafe_allow_html=True
        )

        with st.spinner("Calculando superficie de perda 3D..."):
            fig_ls = plot_loss_landscape(nn, X_tr, y_tr)
        st.plotly_chart(fig_ls, use_container_width=True)

        c3a, c3b = st.columns(2, gap="large")
        with c3a:
            st.markdown('<p class="sl">Trajetoria do gradiente</p>', unsafe_allow_html=True)
            epochs_range = list(range(1, len(h["loss"])+1))
            fig_grad = go.Figure()
            fig_grad.add_trace(go.Scatter(
                x=epochs_range, y=h["loss"],
                fill="tozeroy",
                fillcolor="rgba(124,58,237,.15)",
                line=dict(color="#a78bfa", width=2.5),
                name="Loss",
            ))
            # Anota inicio e fim
            fig_grad.add_annotation(x=1, y=h["loss"][0],
                text="Inicio", showarrow=True, arrowhead=2,
                font=dict(color="#f87171"), arrowcolor="#f87171")
            fig_grad.add_annotation(x=len(h["loss"]), y=h["loss"][-1],
                text="Minimo", showarrow=True, arrowhead=2,
                font=dict(color="#4ade80"), arrowcolor="#4ade80")
            fig_grad.update_layout(**PLOTLY_DARK,
                title="Descida do Gradiente — Loss ao longo do treino",
                xaxis_title="Epoca", yaxis_title="Loss",
                height=320, margin=dict(t=40,b=30,l=40,r=20))
            st.plotly_chart(fig_grad, use_container_width=True)

        with c3b:
            st.markdown('<p class="sl">Norma do gradiente por camada</p>', unsafe_allow_html=True)
            # Calcula norma dos pesos ao longo dos snapshots
            snap_epochs = [s["epoch"] for s in snaps]
            norms = {f"Camada {i+1}": [] for i in range(len(nn.weights))}
            for s in snaps:
                for i, W in enumerate(s["weights"]):
                    norms[f"Camada {i+1}"].append(float(np.linalg.norm(W)))

            colors_norm = ["#a78bfa","#22d3ee","#f472b6","#4ade80"]
            fig_norm = go.Figure()
            for i, (k, v) in enumerate(norms.items()):
                fig_norm.add_trace(go.Scatter(
                    x=snap_epochs, y=v, name=k,
                    line=dict(color=colors_norm[i % len(colors_norm)], width=2),
                    mode="lines+markers",
                    marker=dict(size=5),
                ))
            fig_norm.update_layout(**PLOTLY_DARK,
                title="Norma dos Pesos por Camada (snapshots)",
                xaxis_title="Epoca", yaxis_title="||W||",
                legend=dict(bgcolor="#0d1117", bordercolor="#1e2a38"),
                height=320, margin=dict(t=40,b=30,l=40,r=20))
            st.plotly_chart(fig_norm, use_container_width=True)

    # ════════════════════════════════════════════════════
    # TAB 4: COMO FUNCIONA
    # ════════════════════════════════════════════════════
    with tab4:
        c4a, c4b = st.columns(2, gap="large")

        with c4a:
            st.markdown("### Forward Pass")
            st.markdown(
                "A entrada flui da esquerda para a direita, camada por camada. "
                "Cada neuronio recebe os sinais dos anteriores, pondera pelos pesos e aplica uma funcao de ativacao."
            )
            st.markdown(
                '<div class="formula">'
                'z = W * x + b<br>'
                'a = f(z)  # funcao de ativacao<br><br>'
                '# Saida: Softmax para probabilidades<br>'
                'P(classe_k) = exp(z_k) / sum(exp(z_j))'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("### Funcoes de Ativacao")
            st.markdown(
                '<div class="formula">'
                'ReLU(z)    = max(0, z)<br>'
                'Sigmoid(z) = 1 / (1 + exp(-z))<br>'
                'Tanh(z)    = (exp(z) - exp(-z)) / (exp(z) + exp(-z))'
                '</div>',
                unsafe_allow_html=True
            )

            # Grafico das ativacoes
            z_range = np.linspace(-4, 4, 200)
            fig_act = go.Figure()
            for fn, col, name in [
                (lambda z: np.maximum(0,z),                "#a78bfa","ReLU"),
                (lambda z: 1/(1+np.exp(-z)),               "#22d3ee","Sigmoid"),
                (lambda z: np.tanh(z),                     "#4ade80","Tanh"),
            ]:
                fig_act.add_trace(go.Scatter(x=z_range, y=fn(z_range), name=name,
                                             line=dict(color=col, width=2.5)))
            fig_act.add_hline(y=0, line=dict(color="#1e2a38", width=1))
            fig_act.add_vline(x=0, line=dict(color="#1e2a38", width=1))
            # Destaca a funcao de ativacao atualmente em uso
            _act_colors = {"relu": "#a78bfa", "sigmoid": "#22d3ee", "tanh": "#4ade80"}
            _act_cur    = activation   # vem do sidebar
            fig_act.add_annotation(
                x=2.5, y={"relu": 2.5, "sigmoid": 0.9, "tanh": 0.85}[_act_cur],
                text=f"← em uso: {_act_cur}",
                showarrow=False,
                font=dict(color=_act_colors[_act_cur], size=11),
            )
            fig_act.update_layout(**PLOTLY_DARK, title="Funcoes de Ativacao (ativa destacada)",
                                  xaxis_title="z", yaxis_title="f(z)",
                                  legend=dict(bgcolor="#0d1117", bordercolor="#1e2a38"),
                                  height=280, margin=dict(t=40,b=30,l=40,r=20))
            st.plotly_chart(fig_act, use_container_width=True)

            # Comparacao de propriedades das ativacoes
            st.markdown('<p class="sl">Comparacao de propriedades</p>', unsafe_allow_html=True)
            _act_rows = [
                {"Funcao": "ReLU",    "Faixa": "[0, +inf)", "Gradiente saturado": "Nao (z>0)", "Uso tipico": "Camadas ocultas profundas", "Em uso": "✓" if activation == "relu"    else ""},
                {"Funcao": "Sigmoid", "Faixa": "(0, 1)",    "Gradiente saturado": "Sim",       "Uso tipico": "Classificacao binaria",    "Em uso": "✓" if activation == "sigmoid" else ""},
                {"Funcao": "Tanh",    "Faixa": "(-1, 1)",   "Gradiente saturado": "Sim",       "Uso tipico": "Camadas ocultas (centrado)","Em uso": "✓" if activation == "tanh"    else ""},
            ]
            st.dataframe(pd.DataFrame(_act_rows), hide_index=True, use_container_width=True)

        with c4b:
            st.markdown("### Backpropagation")
            st.markdown(
                "Depois do forward pass, calculamos o erro (loss). "
                "O backprop usa a **regra da cadeia** para calcular quanto cada peso "
                "contribuiu para o erro, propagando o gradiente de tras para frente."
            )
            st.markdown(
                '<div class="formula">'
                '# Erro na saida<br>'
                'delta_out = y_pred - y_true<br><br>'
                '# Gradiente dos pesos<br>'
                'dL/dW = a_prev.T @ delta / n<br><br>'
                '# Propaga para camada anterior<br>'
                'delta_prev = (delta @ W.T) * f_prima(z)'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("### Gradient Descent")
            st.markdown(
                "Com os gradientes calculados, atualizamos cada peso na direcao "
                "que reduz o erro. O **learning rate** controla o tamanho do passo."
            )
            st.markdown(
                '<div class="formula">'
                'W = W - lr * dL/dW<br>'
                'b = b - lr * dL/db<br><br>'
                '# lr muito alto: oscila, nao converge<br>'
                '# lr muito baixo: converge lento demais'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("### Cross-Entropy Loss")
            st.markdown(
                '<div class="formula">'
                'L = -1/n * sum( y_true * log(y_pred) )<br><br>'
                '# Penaliza quando a rede esta<br>'
                '# confiante e errada'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("### Parametros vs Hiperparametros")
            st.markdown("""
| Item | Tipo | Quem ajusta |
|---|---|---|
| Pesos W, bias b | Parametros | A rede (backprop) |
| Learning rate | Hiperparametro | Voce |
| N. de camadas | Hiperparametro | Voce |
| N. de neuronios | Hiperparametro | Voce |
| Funcao de ativacao | Hiperparametro | Voce |
| N. de epocas | Hiperparametro | Voce |
            """)

        st.markdown('<hr class="dv">', unsafe_allow_html=True)
        st.markdown("### Codigo da rede — NumPy puro (trecho essencial)")
        st.code("""
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.01):
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
        self.lr = lr

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        a = X
        self._cache = [X]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            a = self.relu(z) if i < len(self.weights)-1 else self.softmax(z)
            self._cache.append(a)
        return a

    def backward(self, y_true):
        n     = y_true.shape[0]
        delta = self._cache[-1] - y_true      # gradiente da saida
        for i in reversed(range(len(self.weights))):
            dW = self._cache[i].T @ delta / n
            db = delta.mean(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (self._cache[i] > 0)  # ReLU deriv
            self.weights[i] -= self.lr * dW
            self.biases[i]  -= self.lr * db
        """, language="python")

        st.markdown('<p class="sl">Dataset Iris</p>', unsafe_allow_html=True)
        iris = load_iris()
        df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
        df_iris["classe"] = [iris.target_names[t] for t in iris.target]
        st.dataframe(df_iris.sample(20, random_state=42), hide_index=True, use_container_width=True)
        st.caption(f"150 amostras | 4 features | 3 classes | 80% treino / 20% validacao")


if __name__ == "__main__":
    main()
