
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Q-Zeno: DGFT × Quasar Controller for SQL Planning", layout="centered")

# --- MODEL DEFINITIONS ---
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, layers=3):
        super().__init__()
        layers_list = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 2):
            layers_list += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers_list += [nn.Linear(hidden, out_dim)]
        self.model = nn.Sequential(*layers_list)
    def forward(self, x): return self.model(x)

class QZenoGate(nn.Module):
    def __init__(self, dim): super().__init__(); self.net = MLP(dim, 1)
    def forward(self, x): return torch.sigmoid(self.net(x))  # μ ∈ [0,1]

class SketchPredictor(nn.Module):
    def __init__(self, dim): super().__init__(); self.gen = MLP(dim, 1)
    def forward(self, x, mu): return self.gen(x * (mu < 0.5).float())

# --- SIMULATION ---
def encode_nlq(text, dim=8, noise=0.0):
    torch.manual_seed(len(text))
    base = torch.randn(dim)
    noise_tensor = noise * torch.randn(dim)
    return (base + noise_tensor).unsqueeze(0)

# --- INIT MODELS ---
dim = 8
zeno = QZenoGate(dim)
sketch = SketchPredictor(dim)

# --- UI ---
st.title("Q-Zeno Controller for Distyl AI")
st.markdown("**Adaptive DGFT Controller tuned with Quasar Alpha-style signal modulation**")

user_input = st.text_input("Enter your natural language question:", "List top customers who placed orders in the last 90 days.")
noise_level = st.slider("Query Ambiguity (Simulated Noise)", 0.0, 1.0, 0.25, 0.01)

if user_input:
    with torch.no_grad():
        emb = encode_nlq(user_input, dim=dim, noise=noise_level)
        mu = zeno(emb)
        sketch_score = sketch(emb, mu)

    st.subheader("Zeno μ Control Field")
    mu_np = mu.squeeze().numpy()
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.bar(np.arange(len(mu_np)), mu_np, color="tomato")
    ax.set_ylim([0, 1])
    ax.set_ylabel("μ Value")
    ax.set_title("Dimensional Stability Map")
    st.pyplot(fig)

    st.subheader("Sketch Confidence")
    st.metric(label="Predicted Sketch Reliability", value=f"{sketch_score.item():.4f}")

    st.subheader("System Insight")
    μ_mean = mu.mean().item()
    if μ_mean > 0.75:
        st.warning("Query is unstable. Suggest decomposing into sub-steps or requesting clarification.")
    elif μ_mean < 0.3:
        st.success("Stable semantic structure detected. Proceed to SQL generation.")
    else:
        st.info("Moderate stability detected. Recommend reviewing intermediate sketch.")

    st.caption("Prototype inspired by Distyl AI × Quasar Alpha signal behavior + DGFT modulation.")
