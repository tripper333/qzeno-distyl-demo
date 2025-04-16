
import streamlit as st
import openai
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set API key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Compute OpenAI embedding from prompt text
def get_embedding(text, model="text-embedding-ada-002"):
    result = openai.Embedding.create(input=[text], model=model)
    return torch.tensor(result["data"][0]["embedding"])

# DGFT: Compute Zeno μ (sigmoid of norm modulo 1)
def compute_zeno_mu(embedding):
    return torch.sigmoid(torch.tensor([torch.norm(embedding) % 1]))

# DGFT Metrics
def compute_dgft_metrics(embedding, mu):
    norm = torch.norm(embedding).item()
    entropy = (-mu * torch.log(mu + 1e-8)).item()
    variance = embedding.var().item()
    stability_idx = (variance / (mu + 1e-6)).item()
    return norm, entropy, stability_idx

# GPT-4 query with μ-gating
def generate_sql_with_zeno(prompt, embedding, model="gpt-4", temperature=0.3):
    mu = compute_zeno_mu(embedding).item()

    if mu > 0.7:
        return mu, "rejected", "Zeno μ too high. Please simplify the query."

    refinement_hint = ""
    if 0.3 < mu <= 0.7:
        refinement_hint = (
            "\n\nNote: The user's intent appears moderately ambiguous. "
            "You may provide follow-up questions before SQL generation."
        )

    messages = [
        {"role": "system", "content": "You are a precise SQL generator for telecom queries. Respond only with SQL."},
        {"role": "user", "content": prompt + refinement_hint}
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return mu, "ok", response.choices[0].message["content"].strip()
    except Exception as e:
        return mu, "error", str(e)

# Streamlit app UI
st.title("🧠 Q-Zeno GPT-4 Field Visualizer")
st.markdown("This demo uses real OpenAI embeddings and a DGFT-inspired μ-controller for SQL safety.")

prompt = st.text_input("Enter natural language query:")

if prompt:
    embedding = get_embedding(prompt)
    mu = compute_zeno_mu(embedding)
    norm, entropy, stability = compute_dgft_metrics(embedding, mu)

    mu_val, status, result = generate_sql_with_zeno(prompt, embedding)

    st.subheader("Zeno μ Gauge")
    st.metric(label="μ Value", value=round(mu_val, 4))
    st.progress(min(mu_val, 1.0))

    st.subheader("DGFT Field Metrics")
    st.write(f"**Norm:** {round(norm, 4)}")
    st.write(f"**Entropy:** {round(entropy, 6)}")
    st.write(f"**Topological Stability Index:** {round(stability, 4)}")

    st.subheader("Response Status")
    st.write(f"**Status:** `{status}`")

    if status == "ok":
        st.code(result, language="sql")
    else:
        st.warning(result)

    # Optional visual
    st.subheader("Embedding Distribution")
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(embedding.numpy(), alpha=0.7)
    ax.set_title("Latent Embedding Vector")
    st.pyplot(fig)
