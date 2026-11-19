"""
validar_modelos.py - confere se os modelos exportados, rodando nesta máquina, reproduzem o notebook.

1. Recria a divisão treino/validação/teste do notebook (Etapa 6.1, por grupo, seed 42).
2. Calcula o AUROC dos dois modelos no teste e compara com o meta.json.
3. Salva alguns textos do TESTE (nunca vistos no treino) e do Gemini em data/exemplos.json,
   para servirem de exemplos na interface.

Uso (na pasta Plataforma):  .venv\\Scripts\\python ferramentas\\validar_modelos.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AQUI = os.path.dirname(os.path.abspath(__file__))
PLAT = os.path.dirname(AQUI)
sys.path.insert(0, PLAT)
import core  # noqa: E402
from stylometry import first_n_words, preprocess  # noqa: E402

DADOS = os.path.join(os.path.dirname(PLAT), "Resultados", "data")
SEED = 42


def assign_split(d, seed=SEED, frac=(0.70, 0.15, 0.15)):
    rng = np.random.RandomState(seed)
    groups = np.array(d.group.unique()); rng.shuffle(groups); n = len(groups)
    tr = set(groups[:int(frac[0] * n)]); va = set(groups[int(frac[0] * n):int((frac[0] + frac[1]) * n)])
    d = d.copy(); d["split"] = d.group.map(lambda g: "train" if g in tr else ("val" if g in va else "test")); return d


def main():
    df = assign_split(pd.read_csv(os.path.join(DADOS, "dataset_limpo.csv")))
    test = df[df.split == "test"].reset_index(drop=True)
    print("Teste:", len(test), "textos |", test.label.value_counts().to_dict())

    pc = core.probs_classico(test.text.tolist())
    print("Calculando BERTimbau no teste (CPU, pode levar ~1 min)...")
    pb = core.probs_bert(test.text.tolist())
    esperado = core.META["test_metrics"]
    for nome, p in [(core.META["classico"], pc), ("BERTimbau (fine-tuning)", pb)]:
        auc = roc_auc_score(test.label, p)
        ref = esperado[nome]["auroc"]
        print(f"  {nome:28s} AUROC aqui = {auc:.4f} | notebook = {ref:.4f} | {'OK' if abs(auc - ref) < 0.005 else 'DIFERENTE'}")
    pm = (pc + pb) / 2
    h, a = pm[test.label == 0], pm[test.label == 1]
    print(f"  Combinado: humanos >= limiar IA = {(h >= core.T_AI).mean():.1%} (notebook: 8,9%) | IA >= limiar = {(a >= core.T_AI).mean():.1%} (notebook: 94,4%)")

    # Regra final da plataforma: discordância > DISAGREE força 'inconclusivo'
    disc = np.abs(pc - pb) > core.DISAGREE
    y = test.label.values
    final = np.where(disc, "inconclusivo", np.where(pm >= core.T_AI, "ia", np.where(pm <= core.T_HUMAN, "humano", "inconclusivo")))
    print(f"  Regra da plataforma (com discordância > {core.DISAGREE:.2f} -> inconclusivo): "
          f"{disc.mean():.1%} dos textos de teste caem na regra de discordância")
    for nome, m in [("Humanos", y == 0), ("IA", y == 1)]:
        vc = pd.Series(final[m]).value_counts(normalize=True)
        print(f"    {nome:7s}: " + " | ".join(f"{k} {vc.get(k, 0):.1%}" for k in ("humano", "inconclusivo", "ia")))

    # Exemplos: 1 por (domínio, fonte) do teste + 1 Gemini
    test["pm"] = pm
    ex = []
    nomes = {"human": "Humano", "qwen": "IA (Qwen 2.5 1.5B)", "phi": "IA (Phi-3.5 mini)"}
    for (dom, src), g in test.groupby(["domain", "source"]):
        r = g.sample(1, random_state=SEED).iloc[0]
        ex.append({"titulo": f"{nomes.get(src, src)} · {'Wikipédia' if dom == 'wiki' else 'Notícias'}",
                   "origem": "conjunto de teste", "texto": r.text})
    gem_path = os.path.join(DADOS, "geracao_gemini_v4_n300.csv")
    if os.path.exists(gem_path):
        gem = pd.read_csv(gem_path)
        col = next(c for c in ("text", "raw_text") if c in gem.columns)
        gem = gem[gem[col].astype(str).str.split().str.len() >= 100]
        if len(gem):
            r = gem.sample(1, random_state=SEED).iloc[0]
            ex.append({"titulo": "IA (Gemini: gerador externo, nunca visto no treino)", "origem": "teste de robustez",
                       "texto": first_n_words(preprocess(r[col]), 100)})
    out = os.path.join(PLAT, "data", "exemplos.json")
    json.dump(ex, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("Exemplos salvos em", out, "->", [e["titulo"] for e in ex])


if __name__ == "__main__":
    main()
