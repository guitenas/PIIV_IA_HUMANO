"""
core.py - lógica de inferência da plataforma (sem interface).

Projeto: Humano ou IA? Identificação de autoria textual em português (FATEC Jundiaí)
Autores: Allan Vytor Francisco Oliveira, Gabriel Passos, Guilherme Tenório Cavalcante

Separado da interface (app.py) para poder ser testado sozinho:
    python core.py            -> roda uma análise de exemplo no terminal
"""
import json
import os
import re
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from stylometry import (FEATURE_COLS, FEATURE_LABELS, STOP_PT, TRUNC_WORDS, build_features, n_words,
                        preprocess, windows)

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
DATA = os.path.join(BASE, "data")

META = json.load(open(os.path.join(MODELS, "meta.json"), encoding="utf-8"))
REF = json.load(open(os.path.join(MODELS, "reference_stats.json"), encoding="utf-8"))

T_HUMAN = float(META["thresholds"]["human_max"])   # P(IA) <= 0,35 -> provavelmente humano
T_AI = float(META["thresholds"]["ai_min"])         # P(IA) >= 0,661 -> provavelmente IA (escolhido na validação)
MAX_LEN = int(META.get("bert_max_len", 192))

MIN_WORDS = 10          # abaixo disso não analisamos
MIN_WORDS_WARN = 50     # abaixo disso avisamos que a confiabilidade é baixa
DISAGREE = 0.40         # diferença entre os modelos que gera aviso
STOP_RATIO_MIN = 0.15   # proporção de stopwords PT muito baixa -> talvez não seja português

NOME_CLASSICO = "Clássico: " + META.get("classico", "TF-IDF caracteres + LR")
NOME_BERT = "Transformer: BERTimbau (fine-tuning)"

# ----------------------------------------------------------------------
# Carregamento (uma vez, na importação)
# ----------------------------------------------------------------------
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
CLS = joblib.load(os.path.join(MODELS, "classico.joblib"))
TOK = AutoTokenizer.from_pretrained(os.path.join(MODELS, "bertimbau"))
BERT = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODELS, "bertimbau")).eval()
IDX_IA = 1   # convenção do notebook: label 1 = IA (o config.json não traz id2label)

# Peças do pipeline clássico para a explicação por n-gramas (só se for TF-IDF + LR)
_VEC = _LR = _VOCAB = _COEF = None
if CLS.get("kind") == "text" and hasattr(CLS["model"], "steps"):
    _VEC, _LR = CLS["model"].steps[0][1], CLS["model"].steps[-1][1]
    if hasattr(_LR, "coef_") and list(_LR.classes_) == [0, 1]:
        _VOCAB, _COEF = _VEC.vocabulary_, _LR.coef_[0]
        _INV_VOCAB = {j: g for g, j in _VOCAB.items()}


# ----------------------------------------------------------------------
# Probabilidades
# ----------------------------------------------------------------------
def probs_classico(chunks):
    if CLS["kind"] == "text":
        return CLS["model"].predict_proba(chunks)[:, 1]
    return CLS["model"].predict_proba(build_features(chunks)[CLS["cols"]])[:, 1]


@torch.no_grad()
def probs_bert(chunks):
    out = []
    for i in range(0, len(chunks), 8):
        enc = TOK(chunks[i:i + 8], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        out.append(torch.softmax(BERT(**enc).logits, -1)[:, IDX_IA].numpy())
    return np.concatenate(out)


def faixa(p):
    """Três faixas: 'ia', 'humano' ou 'inconclusivo'."""
    if p >= T_AI:
        return "ia"
    if p <= T_HUMAN:
        return "humano"
    return "inconclusivo"


def pct(p, casas=0):
    """Formata probabilidade sem sugerir certeza absoluta (nunca mostra 0% nem 100%)."""
    lim = 0.5 * 10 ** -(casas + 2)
    if p >= 1 - lim:
        return f">{100 - 10 ** -casas:.{casas}f}%"
    if p < lim:
        return f"<{10 ** -casas:.{casas}f}%"
    return f"{p * 100:.{casas}f}%"


T_MODELO = 0.5   # corte de cada modelo isolado (o mesmo usado nas métricas por modelo do notebook)


def faixa_modelo(p):
    """Conclusão binária de UM modelo: 'ia' se P(IA) >= 50%, senão 'humano'."""
    return "ia" if p >= T_MODELO else "humano"


ROTULO_MODELO = {"ia": "IA", "humano": "Humano"}

ROTULO = {"ia": "Provavelmente gerado por IA", "humano": "Provavelmente escrito por humano",
          "inconclusivo": "Inconclusivo"}


# ----------------------------------------------------------------------
# Indicadores estilométricos (descritivos: NÃO são a entrada do modelo clássico exportado)
# ----------------------------------------------------------------------
def indicadores(chunks):
    f = build_features(chunks).mean()
    rows = []
    for c in FEATURE_COLS:
        r, v = REF[c], float(f[c])
        d_h = abs(v - r["human_mean"]) / (r["human_std"] + 1e-9)
        d_a = abs(v - r["ai_mean"]) / (r["ai_std"] + 1e-9)
        gap = r["ai_mean"] - r["human_mean"]
        pos = (v - r["human_mean"]) / gap if abs(gap) > 1e-9 else 0.5
        rows.append({"col": c, "Indicador": FEATURE_LABELS[c], "Este texto": v, "Média humanos": r["human_mean"],
                     "Média IA": r["ai_mean"], "Mais próximo de": "IA" if d_a < d_h else "Humano",
                     "pos": float(pos), "effect": float(r["effect"])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Explicação do modelo clássico: contribuição de cada n-grama de caracteres
# ----------------------------------------------------------------------
def _char_wb_spans(text, lo, hi):
    """Reproduz o analyzer 'char_wb' do scikit-learn, devolvendo (n-grama, início, fim) no texto original.
    As posições da palavra acolchoada ' palavra ' mapeiam para [start-1, end] no texto."""
    out = []
    for m in re.finditer(r"\S+", text):
        w = " " + m.group().lower() + " "
        if len(w) != len(m.group()) + 2:      # lower() mudou o tamanho (raro): não dá para mapear posições
            continue
        base = m.start() - 1
        for n in range(lo, hi + 1):
            off = 0
            out.append((w[off:off + n], base + off, base + off + n))
            while off + n < len(w):
                off += 1
                out.append((w[off:off + n], base + off, base + off + n))
            if off == 0:
                break
    return out


def contribuicoes(chunk):
    """Para um trecho, devolve (score por caractere, dict n-grama -> contribuição no logit).
    contribuição(j) = coef_j * tfidf_j. Somando tudo + intercepto = logit do modelo clássico."""
    if _COEF is None:
        return None, {}
    x = _VEC.transform([chunk]).tocoo()
    contrib = {int(j): float(v) * float(_COEF[j]) for j, v in zip(x.col, x.data)}
    lo, hi = _VEC.ngram_range
    spans = [(g, a, b) for g, a, b in _char_wb_spans(chunk, lo, hi) if g in _VOCAB]
    count = {}
    for g, _, _ in spans:
        count[g] = count.get(g, 0) + 1
    score = np.zeros(len(chunk))
    for g, a, b in spans:
        j = _VOCAB[g]
        if j not in contrib:
            continue
        a2, b2 = max(a, 0), min(b, len(chunk))
        if b2 > a2:
            score[a2:b2] += contrib[j] / count[g] / (b - a)
    return score, {_INV_VOCAB[j]: c for j, c in contrib.items()}


def palavras_com_score(chunk, score):
    """Agrupa o score por palavra: lista de (palavra, score)."""
    return [(m.group(), float(score[m.start():m.end()].sum())) for m in re.finditer(r"\S+", chunk)]


# ----------------------------------------------------------------------
# Análise completa
# ----------------------------------------------------------------------
@dataclass
class Resultado:
    ok: bool
    mensagem: str = ""
    n_palavras: int = 0
    n_palavras_analisadas: int = 0
    chunks: list = field(default_factory=list)
    resto: str = ""
    p_classico: float = 0.0
    p_bert: float = 0.0
    p_final: float = 0.0
    p_classico_trechos: list = field(default_factory=list)
    p_bert_trechos: list = field(default_factory=list)
    avisos: list = field(default_factory=list)
    indicadores: pd.DataFrame = None
    palavras_trechos: list = field(default_factory=list)   # por trecho: [(palavra, score), ...]
    ngramas: pd.DataFrame = None                            # n-grama, contribuição média

    @property
    def discordam(self):
        return abs(self.p_classico - self.p_bert) > DISAGREE

    @property
    def faixa(self):
        """Veredito final. Se os modelos discordam fortemente, é sempre 'inconclusivo'
        (decisão do grupo: a média de duas opiniões opostas não é evidência)."""
        if self.discordam:
            return "inconclusivo"
        return faixa(self.p_final)


def analisar(texto):
    t = preprocess(texto or "")
    nw = n_words(t)
    if nw < MIN_WORDS:
        return Resultado(ok=False, n_palavras=nw,
                         mensagem=f"O texto tem {nw} palavra(s) depois da limpeza. Cole um texto em português com "
                                  f"pelo menos {MIN_WORDS} palavras (o ideal é {TRUNC_WORDS} ou mais).")
    chunks = windows(t, TRUNC_WORDS, min_last=50)
    analisadas = sum(n_words(c) for c in chunks)
    resto = " ".join(t.split()[analisadas:])

    pc_t, pb_t = probs_classico(chunks), probs_bert(chunks)
    pc, pb = float(np.mean(pc_t)), float(np.mean(pb_t))
    pm = (pc + pb) / 2

    avisos = []
    if nw < MIN_WORDS_WARN:
        avisos.append(f"Texto curto ({nw} palavras): a confiabilidade é baixa. Os modelos foram treinados com "
                      f"trechos de {TRUNC_WORDS} palavras.")
    if len(chunks) > 1:
        avisos.append(f"Texto dividido em {len(chunks)} trechos de ~{TRUNC_WORDS} palavras; cada probabilidade é a "
                      "média dos trechos.")
    if resto:
        avisos.append(f"As últimas {n_words(resto)} palavras não foram analisadas (trecho final com menos de 50 "
                      "palavras).")
    if abs(pc - pb) > DISAGREE:
        avisos.append(f"Os dois modelos discordam fortemente (diferença de {abs(pc - pb) * 100:.0f} pontos, limite "
                      f"{DISAGREE * 100:.0f}). Por isso o resultado foi classificado como inconclusivo, mesmo que "
                      "a média caia em outra faixa.")
    words = re.findall(r"\w+", t.lower())
    if words and sum(w in STOP_PT for w in words) / len(words) < STOP_RATIO_MIN:
        avisos.append("O texto parece não estar em português (poucas palavras funcionais do português). "
                      "Os modelos só foram treinados com português.")

    palavras_trechos, agg = [], {}
    for c in chunks:
        score, contrib = contribuicoes(c)
        if score is None:
            break
        palavras_trechos.append(palavras_com_score(c, score))
        for g, v in contrib.items():
            agg[g] = agg.get(g, 0.0) + v / len(chunks)
    ngr = (pd.DataFrame({"ngrama": list(agg), "contrib": list(agg.values())}) if agg else None)

    return Resultado(ok=True, n_palavras=nw, n_palavras_analisadas=analisadas, chunks=chunks, resto=resto,
                     p_classico=pc, p_bert=pb, p_final=pm,
                     p_classico_trechos=[float(x) for x in pc_t], p_bert_trechos=[float(x) for x in pb_t],
                     avisos=avisos, indicadores=indicadores(chunks), palavras_trechos=palavras_trechos,
                     ngramas=ngr)


def metricas_teste():
    rows = []
    for nome, m in META.get("test_metrics", {}).items():
        rows.append({"Modelo": nome, "Acurácia": f"{m['acc']:.1%}", "Precisão": f"{m['precision']:.1%}",
                     "Recall": f"{m['recall']:.1%}", "F1": f"{m['f1']:.3f}", "AUROC": f"{m['auroc']:.3f}",
                     "Humanos marcados como IA (FPR)": f"{m['fpr']:.1%}",
                     "IA detectada com FPR de 5%": f"{m['tpr@fpr5%']:.1%}"})
    return pd.DataFrame(rows)


def robustez():
    p = os.path.join(DATA, "resultados_robustez.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    return d[d.detector.isin(META.get("test_metrics", {}).keys())]


if __name__ == "__main__":
    ex = ("A fotossíntese é um processo fundamental para a vida na Terra. Por meio dela, plantas, algas e algumas "
          "bactérias convertem energia luminosa em energia química, produzindo glicose e liberando oxigênio. "
          "Além disso, esse processo desempenha um papel importante na regulação do clima.")
    r = analisar(ex)
    print(f"Clássico {r.p_classico:.3f} | BERT {r.p_bert:.3f} | final {r.p_final:.3f} -> {ROTULO[r.faixa]}")
    print("Avisos:", r.avisos)
    print(r.ngramas.sort_values("contrib").tail(5) if r.ngramas is not None else "sem n-gramas")
