"""
stylometry.py - pré-processamento e atributos estilométricos para PT-BR.

Projeto: Humano ou IA? Identificação de autoria textual em português (FATEC Jundiaí)
Autores: Allan Vytor Francisco Oliveira, Gabriel Passos, Guilherme Tenório Cavalcante

Este módulo é usado tanto no notebook (treino) quanto na plataforma (app.py).
Manter UM único código garante que o texto analisado no app seja tratado
exatamente como foi tratado no treino.
"""
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

TRUNC_WORDS = 100            # tamanho (em palavras) dos textos usados no treino
NORMALIZE_TYPOGRAPHY = True  # aspas, reticências e travessões tipográficos -> ASCII (evita "atalhos" de formatação)

# ----------------------------------------------------------------------
# Utilidades básicas
# ----------------------------------------------------------------------
def normalize_ws(t):
    return re.sub(r"\s+", " ", str(t)).strip()

def split_sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]

def n_words(t):
    return len(str(t).split())

def first_n_words(t, n):
    return " ".join(str(t).split()[:n])

def windows(text, size=TRUNC_WORDS, min_last=50):
    """Divide um texto em janelas de `size` palavras (a última só entra se tiver >= min_last)."""
    words = str(text).split()
    if len(words) <= size:
        return [" ".join(words)]
    chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    if len(chunks) > 1 and len(chunks[-1].split()) < min_last:
        chunks = chunks[:-1]
    return chunks

# ----------------------------------------------------------------------
# Pré-processamento SIMÉTRICO (aplicado igualmente a textos humanos e de IA)
# ----------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_CITE_RE = re.compile(r"\[\s*(\d+|nota \d+|carece de fontes|citation needed)\s*\]", re.I)
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]+")
_INVIS_RE = re.compile("[​‌‍﻿­]")
_MD_RE = re.compile(r"[*#`_>]+")
_TYPO = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'",
                       "«": '"', "»": '"', "…": "...",
                       "—": "-", "–": "-"})

def preprocess(t):
    """Normaliza Unicode, remove URLs/e-mails/citações/emojis/markdown e padroniza aspas e travessões."""
    t = unicodedata.normalize("NFC", str(t))
    t = _INVIS_RE.sub("", t)
    for rx in (_URL_RE, _EMAIL_RE, _CITE_RE, _EMOJI_RE):
        t = rx.sub(" ", t)
    if NORMALIZE_TYPOGRAPHY:
        t = t.translate(_TYPO)
    t = _MD_RE.sub(" ", t)
    return normalize_ws(t)

# ----------------------------------------------------------------------
# Léxicos
# ----------------------------------------------------------------------
STOP_PT = set("a o as os de da do das dos em no na nos nas um uma uns umas e ou mas que se com por para como mais "
              "ao à aos às é são foi ser sua seu suas seus não já também entre sobre até mesmo ela ele eles elas "
              "isso este esta esse essa".split())

CONNECTIVES = ["portanto", "além disso", "entretanto", "no entanto", "contudo", "todavia", "por outro lado",
               "ademais", "dessa forma", "desse modo", "dessa maneira", "assim", "ou seja", "em suma", "por fim",
               "em resumo", "em conclusão", "por exemplo", "porém", "logo", "de fato", "com isso",
               "consequentemente", "sobretudo", "por isso", "em contrapartida"]
# Hipótese testável: LLMs usam mais marcadores de ênfase/valoração
EMPHASIS = ["é importante", "é fundamental", "vale destacar", "vale ressaltar", "é essencial",
            "desempenha um papel", "papel fundamental", "papel importante", "diversos aspectos"]

def _phrase_re(items):
    return re.compile(r"\b(" + "|".join(re.escape(x) for x in sorted(items, key=len, reverse=True)) + r")\b")

_CONN_RE = _phrase_re(CONNECTIVES)
_EMPH_RE = _phrase_re(EMPHASIS)

# ----------------------------------------------------------------------
# Silabação aproximada e legibilidade (Flesch adaptado ao português)
# ----------------------------------------------------------------------
_VOW_GROUP = re.compile(r"[aeiouáéíóúâêôãõàü]+")
_STRONG = re.compile(r"[aeoáéóâêôãõ]")

def count_syllables_pt(word):
    """Aproximação: cada grupo vocálico = 1 sílaba; vogais fortes consecutivas (a/e/o) formam hiato."""
    n = 0
    for g in _VOW_GROUP.findall(word.lower()):
        n += max(1, len(_STRONG.findall(g)))
    return max(n, 1)

def flesch_pt(n_words_, n_sents, n_syll):
    """Índice de Facilidade de Leitura Flesch adaptado ao português (Martins et al., 1996)."""
    if n_words_ == 0 or n_sents == 0:
        return 0.0
    return 248.835 - 1.015 * (n_words_ / n_sents) - 84.6 * (n_syll / n_words_)

# ----------------------------------------------------------------------
# Atributos estilométricos
# ----------------------------------------------------------------------
def stylo_features(t):
    t = str(t)
    words = re.findall(r"\w+", t.lower())
    nw = max(len(words), 1)
    sents = split_sentences(t) or [t]
    sl = np.array([len(re.findall(r"\w+", s)) for s in sents], dtype=float)
    bi = list(zip(words, words[1:]))
    tri = list(zip(words, words[1:], words[2:]))
    cnt = Counter(words)
    syl = sum(count_syllables_pt(w) for w in words)
    low = t.lower()
    L = max(len(t), 1)
    return {
        "st_n_sents": len(sents),
        "st_avg_sent_len": sl.mean(),
        "st_std_sent_len": sl.std(),
        "st_cv_sent_len": sl.std() / (sl.mean() + 1e-9),
        "st_max_sent_len": sl.max(),
        "st_avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "st_long_word_ratio": float(np.mean([len(w) >= 9 for w in words])) if words else 0.0,
        "st_ttr": len(cnt) / nw,
        "st_hapax_ratio": sum(1 for c in cnt.values() if c == 1) / nw,
        "st_stop_ratio": sum(w in STOP_PT for w in words) / nw,
        "st_connectives": 100 * len(_CONN_RE.findall(low)) / nw,
        "st_emphasis": 100 * len(_EMPH_RE.findall(low)) / nw,
        "st_flesch": flesch_pt(len(words), len(sents), syl),
        "st_syll_per_word": syl / nw,
        "st_comma": t.count(",") / nw,
        "st_period": t.count(".") / nw,
        "st_colon_semicolon": (t.count(":") + t.count(";")) / nw,
        "st_excl_quest": (t.count("!") + t.count("?")) / nw,
        "st_quotes_parens": sum(t.count(c) for c in '"()') / nw,
        "st_dash": t.count(" - ") / nw,
        "st_digit_ratio": sum(c.isdigit() for c in t) / L,
        "st_upper_ratio": sum(c.isupper() for c in t) / L,
        "st_rep_bigram": 1 - len(set(bi)) / max(len(bi), 1),
        "st_rep_trigram": 1 - len(set(tri)) / max(len(tri), 1),
    }

FEATURE_COLS = list(stylo_features("Exemplo de texto. Outra frase aqui, com vírgula.").keys())

FEATURE_LABELS = {
    "st_n_sents": "Nº de frases",
    "st_avg_sent_len": "Tamanho médio da frase (palavras)",
    "st_std_sent_len": "Desvio do tamanho das frases",
    "st_cv_sent_len": "Variação relativa do tamanho das frases",
    "st_max_sent_len": "Maior frase (palavras)",
    "st_avg_word_len": "Tamanho médio da palavra (letras)",
    "st_long_word_ratio": "Proporção de palavras longas (>=9 letras)",
    "st_ttr": "Riqueza vocabular (type-token ratio)",
    "st_hapax_ratio": "Palavras que aparecem uma só vez",
    "st_stop_ratio": "Proporção de stopwords",
    "st_connectives": "Conectivos (por 100 palavras)",
    "st_emphasis": "Marcadores de ênfase (por 100 palavras)",
    "st_flesch": "Legibilidade (Flesch-PT; maior = mais fácil)",
    "st_syll_per_word": "Sílabas por palavra (aprox.)",
    "st_comma": "Vírgulas por palavra",
    "st_period": "Pontos por palavra",
    "st_colon_semicolon": "Dois-pontos e ponto e vírgula por palavra",
    "st_excl_quest": "Exclamações e interrogações por palavra",
    "st_quotes_parens": "Aspas e parênteses por palavra",
    "st_dash": "Travessões ( - ) por palavra",
    "st_digit_ratio": "Proporção de dígitos",
    "st_upper_ratio": "Proporção de maiúsculas",
    "st_rep_bigram": "Repetição de bigramas",
    "st_rep_trigram": "Repetição de trigramas",
}

def build_features(texts):
    """DataFrame (uma linha por texto) com os atributos estilométricos, na ordem de FEATURE_COLS."""
    return pd.DataFrame([stylo_features(t) for t in texts])[FEATURE_COLS]
