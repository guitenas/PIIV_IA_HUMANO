"""
app.py - Plataforma de análise de autoria textual (humano vs. IA)

Projeto: Humano ou IA? Identificação de autoria textual em português (FATEC Jundiaí)
Autores: Allan Vytor Francisco Oliveira, Gabriel Passos, Guilherme Tenório Cavalcante

Execução:  python app.py   ->   http://127.0.0.1:7860
Requer, na mesma pasta: stylometry.py e a pasta models/ exportada pelo notebook.
"""
import json
import os

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from stylometry import FEATURE_COLS, FEATURE_LABELS, TRUNC_WORDS, build_features, n_words, preprocess, windows

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")

META = json.load(open(os.path.join(MODELS, "meta.json"), encoding="utf-8"))
REF = json.load(open(os.path.join(MODELS, "reference_stats.json"), encoding="utf-8"))
CLS = joblib.load(os.path.join(MODELS, "classico.joblib"))
TOK = AutoTokenizer.from_pretrained(os.path.join(MODELS, "bertimbau"))
BERT = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODELS, "bertimbau")).eval()
MAX_LEN = META.get("bert_max_len", 192)
T_HUMAN, T_AI = META["thresholds"]["human_max"], META["thresholds"]["ai_min"]
MIN_WORDS_WARN = 50

AVISO = ("**Estimativa experimental, não prova de autoria.** Os modelos foram treinados com trechos de cerca de "
         f"{TRUNC_WORDS} palavras em português (Wikipedia e notícias) e com textos de dois modelos de linguagem abertos. "
         "O resultado é uma probabilidade estatística que pode errar, inclusive marcando como IA um texto escrito por "
         "uma pessoa. Não use este resultado para acusar ou punir ninguém.")


def prob_classico(chunks):
    if CLS["kind"] == "text":
        p = CLS["model"].predict_proba(chunks)[:, 1]
    else:
        p = CLS["model"].predict_proba(build_features(chunks)[CLS["cols"]])[:, 1]
    return float(np.mean(p))


@torch.no_grad()
def prob_bert(chunks):
    enc = TOK(chunks, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    p = torch.softmax(BERT(**enc).logits, -1)[:, 1].numpy()
    return float(np.mean(p))


def veredito(p):
    if p >= T_AI:
        return "Provavelmente gerado por IA"
    if p <= T_HUMAN:
        return "Provavelmente escrito por humano"
    return "Inconclusivo"


def indicadores(chunks):
    f = build_features(chunks).mean()            # média dos trechos de ~100 palavras
    rows = []
    for c in FEATURE_COLS:
        r, v = REF[c], float(f[c])
        d_h = abs(v - r["human_mean"]) / (r["human_std"] + 1e-9)
        d_a = abs(v - r["ai_mean"]) / (r["ai_std"] + 1e-9)
        rows.append({"Indicador": FEATURE_LABELS[c], "Este texto": v, "Média humanos": r["human_mean"],
                     "Média IA": r["ai_mean"], "Mais próximo de": "IA" if d_a < d_h else "Humano", "effect": r["effect"]})
    return pd.DataFrame(rows)


def grafico(tab, k=10):
    top = tab.reindex(tab["effect"].abs().sort_values(ascending=False).index).head(k)
    pos = [(r["Este texto"] - r["Média humanos"]) / (r["Média IA"] - r["Média humanos"])
           if abs(r["Média IA"] - r["Média humanos"]) > 1e-9 else 0.5 for _, r in top.iterrows()]
    pos = np.clip(pos, -1, 2)
    y = np.arange(len(top))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axvspan(-1.05, 0.5, color="#2471a3", alpha=0.06)
    ax.axvspan(0.5, 2.05, color="#c0392b", alpha=0.06)
    ax.axvline(0, color="#2471a3", ls="--", lw=1.2, label="média dos humanos")
    ax.axvline(1, color="#c0392b", ls="--", lw=1.2, label="média da IA")
    ax.scatter(pos, y, s=70, color="#222", zorder=3, label="este texto")
    ax.set_yticks(y, top["Indicador"], fontsize=8)
    ax.set_xlim(-1.05, 2.05)
    ax.set_xlabel("Posição relativa (0 = média humana · 1 = média da IA)")
    ax.set_title("Os 10 indicadores que mais separam humano e IA")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    return fig


def analisar(texto):
    t = preprocess(texto or "")
    nw = n_words(t)
    if nw < 10:
        return "Cole um texto em português com pelo menos algumas frases.", None, None, None, None
    chunks = windows(t, TRUNC_WORDS, min_last=50)
    pc, pb = prob_classico(chunks), prob_bert(chunks)
    pm = (pc + pb) / 2

    avisos = []
    if nw < MIN_WORDS_WARN:
        avisos.append(f"Texto curto ({nw} palavras): a confiabilidade é baixa. O ideal é ter {TRUNC_WORDS} palavras ou mais.")
    if len(chunks) > 1:
        avisos.append(f"Texto dividido em {len(chunks)} trechos de ~{TRUNC_WORDS} palavras; as probabilidades são a média dos trechos.")
    if abs(pc - pb) > 0.4:
        avisos.append("Os dois modelos discordam fortemente: trate o resultado como inconclusivo.")

    resumo = (f"## {veredito(pm)}\n**Probabilidade estimada de IA (média dos dois modelos): {pm:.0%}**\n\n"
              + "".join(f"- {a}\n" for a in avisos) + "\n" + AVISO)
    modelos = pd.DataFrame([
        {"Modelo": f"Clássico: {CLS['name']}", "P(IA)": f"{pc:.1%}", "Classificação": veredito(pc)},
        {"Modelo": "Transformer: BERTimbau (fine-tuning)", "P(IA)": f"{pb:.1%}", "Classificação": veredito(pb)},
    ])
    tab = indicadores(chunks)
    fig = grafico(tab)
    tab_show = tab.drop(columns="effect").round(3)
    return resumo, {"IA": pm, "Humano": 1 - pm}, modelos, tab_show, fig


def rodape():
    linhas = []
    for nome, m in META.get("test_metrics", {}).items():
        linhas.append(f"{nome}: AUROC {m['auroc']:.3f} · F1 {m['f1']:.3f} · FPR {m['fpr']:.1%}")
    return ("**Desempenho no conjunto de teste (mesma distribuição do treino):** " + " | ".join(linhas)
            + f"  \nModelos treinados em {META.get('criado_em', '?')}. Fora dessas condições (outros geradores, "
              "outros gêneros, textos editados por humanos) o desempenho tende a cair; veja a Etapa 11 do notebook.")


with gr.Blocks(title="Humano ou IA? · Análise de autoria") as demo:
    gr.Markdown("# Humano ou IA? Análise de autoria textual\n"
                "Projeto Integrador · FATEC Jundiaí · Allan Vytor Francisco Oliveira, Gabriel Passos e "
                "Guilherme Tenório Cavalcante\n\n" + AVISO)
    with gr.Row():
        with gr.Column(scale=3):
            entrada = gr.Textbox(lines=12, label="Texto em português", placeholder="Cole aqui o texto (ideal: 100 palavras ou mais)")
            botao = gr.Button("Analisar", variant="primary")
        with gr.Column(scale=2):
            out_resumo = gr.Markdown()
            out_prob = gr.Label(num_top_classes=2, label="Probabilidade estimada")
    out_modelos = gr.Dataframe(label="Resultado por modelo", interactive=False)
    with gr.Row():
        out_graf = gr.Plot(label="Visualização dos indicadores")
        out_tab = gr.Dataframe(label="Indicadores estilométricos", interactive=False)
    gr.Markdown(rodape())
    botao.click(analisar, inputs=entrada, outputs=[out_resumo, out_prob, out_modelos, out_tab, out_graf])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
