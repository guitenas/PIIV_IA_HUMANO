"""
charts.py - visualizações (Plotly) e trechos de HTML da plataforma.

Cores: azul = humano, vermelho = IA, cinza = inconclusivo (par divergente azul<->vermelho).
Os dois MODELOS usam violeta e água, para não confundir "modelo" com "classe".
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core import (META, NOME_BERT, NOME_CLASSICO, ROTULO, ROTULO_MODELO, T_AI, T_HUMAN, T_MODELO, faixa,
                  faixa_modelo, pct)

HUM, IA, INC = "#2a78d6", "#e34948", "#898781"
HUM_T, IA_T, INC_T = "rgba(42,120,214,0.12)", "rgba(227,73,72,0.12)", "rgba(137,135,129,0.10)"
M_CLS, M_BERT = "#6250d6", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COR_FAIXA = {"humano": HUM, "ia": IA, "inconclusivo": INC}
FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK2)


def _base(fig, height, **kw):
    layout = dict(height=height, margin=dict(l=10, r=20, t=40, b=40), paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", font=FONT, hoverlabel=dict(font=FONT, bgcolor="#fcfcfb"))
    layout.update(kw)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#c3c2b7", linecolor="#c3c2b7", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#c3c2b7", linecolor="#c3c2b7", tickfont=dict(color=INK2))
    return fig


# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------
def medidor(p, f=None):
    f = f or faixa(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=min(max(round(p * 100, 1), 0.1), 99.9),   # nunca 0% nem 100%
        number=dict(suffix="%", font=dict(size=40, color=INK)),
        title=dict(text="Probabilidade estimada de IA<br><span style='font-size:12px;color:#52514e'>média dos dois modelos</span>",
                   font=dict(size=14, color=INK2)),
        gauge=dict(axis=dict(range=[0, 100], ticksuffix="%", tickcolor=MUTED, tickfont=dict(color=MUTED)),
                   bar=dict(color=COR_FAIXA[f], thickness=0.3), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, T_HUMAN * 100], color="rgba(42,120,214,0.22)"),
                          dict(range=[T_HUMAN * 100, T_AI * 100], color="rgba(137,135,129,0.20)"),
                          dict(range=[T_AI * 100, 100], color="rgba(227,73,72,0.22)")])))
    return _base(fig, 250, margin=dict(l=30, r=30, t=70, b=10))


def _faixas_linha(fig, y, x1=100, tres=True):
    """Fundo de UMA linha do gráfico (categoria na posição y): 3 faixas (combinada) ou corte em 50% (modelo)."""
    cortes = [(0, T_HUMAN * 100, HUM_T), (T_HUMAN * 100, T_AI * 100, INC_T), (T_AI * 100, x1, IA_T)] if tres \
        else [(0, T_MODELO * 100, HUM_T), (T_MODELO * 100, x1, IA_T)]
    for a, b, c in cortes:
        fig.add_shape(type="rect", x0=a, x1=b, y0=y - 0.42, y1=y + 0.42, fillcolor=c, line_width=0, layer="below")
    linhas = [(T_HUMAN * 100, HUM), (T_AI * 100, IA)] if tres else [(T_MODELO * 100, INC)]
    for v, c in linhas:
        fig.add_shape(type="line", x0=v, x1=v, y0=y - 0.42, y1=y + 0.42, line=dict(color=c, width=1, dash="dot"))


def modelos(r):
    # De cima para baixo, na mesma ordem da tabela: clássico (y=2), BERTimbau (y=1), combinada (y=0)
    nomes = [NOME_CLASSICO, "Transformer: BERTimbau", "Combinada (média)"]
    ys = [2, 1, 0]
    ps = [r.p_classico, r.p_bert, r.p_final]
    fs = [faixa_modelo(r.p_classico), faixa_modelo(r.p_bert), r.faixa]
    rot = [f"{ROTULO_MODELO[fs[0]]} (corte 50%)", f"{ROTULO_MODELO[fs[1]]} (corte 50%)",
           ROTULO[fs[2]] + (" (modelos discordam)" if r.discordam else "")]
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in ps], y=ys, orientation="h", marker=dict(color=[COR_FAIXA[f] for f in fs],
                                                                    cornerradius=4),
        width=0.45, text=[pct(p) for p in ps], textposition="outside", textfont=dict(color=INK),
        customdata=list(zip(nomes, rot)),
        hovertemplate="<b>%{customdata[0]}</b><br>P(IA) = %{x:.1f}%<br>%{customdata[1]}<extra></extra>"))
    _faixas_linha(fig, 2, tres=False)
    _faixas_linha(fig, 1, tres=False)
    _faixas_linha(fig, 0, tres=True)
    fig.update_yaxes(tickvals=ys, ticktext=nomes, range=[-0.6, 2.6], showgrid=False)
    fig.update_xaxes(range=[0, 112], ticksuffix="%", tickvals=[0, 20, 40, 60, 80, 100])
    fig.add_annotation(x=0, xref="paper", y=-0.2, yref="paper", xanchor="left", yanchor="top", showarrow=False,
                       font=dict(color=MUTED, size=11),
                       text="P(IA) · Modelos: corte em 50%  ·  Combinada: humano ≤ 35% · inconclusivo · IA ≥ 66%")
    return _base(fig, 260, showlegend=False, margin=dict(l=10, r=20, t=20, b=60))


def trechos(r):
    n = len(r.chunks)
    x = [f"Trecho {i + 1}" for i in range(n)]
    fig = go.Figure()
    for nome, ps, cor, sym in [(NOME_CLASSICO, r.p_classico_trechos, M_CLS, "circle"),
                               ("Transformer: BERTimbau", r.p_bert_trechos, M_BERT, "diamond")]:
        fig.add_trace(go.Scatter(x=[p * 100 for p in ps], y=x, mode="markers", name=nome,
                                 marker=dict(color=cor, size=12, symbol=sym, line=dict(color="#fcfcfb", width=2)),
                                 hovertemplate="<b>%{y}</b><br>" + nome + ": %{x:.1f}%<extra></extra>"))
    fig.add_vrect(x0=0, x1=T_MODELO * 100, fillcolor=HUM_T, line_width=0, layer="below")
    fig.add_vrect(x0=T_MODELO * 100, x1=100, fillcolor=IA_T, line_width=0, layer="below")
    fig.add_vline(x=T_MODELO * 100, line=dict(color=INC, width=1, dash="dot"))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="P(IA) de cada modelo (corte em 50%)")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 150 + 40 * n, margin=dict(l=10, r=20, t=50, b=40),
                 legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


def indicadores(tab, k=10):
    top = tab.reindex(tab["effect"].abs().sort_values(ascending=False).index).head(k).iloc[::-1]
    bruto = top["pos"].values
    pos = np.clip(bruto, -1, 2)
    simb = ["triangle-right" if v > 2 else ("triangle-left" if v < -1 else "circle") for v in bruto]
    cores = [IA if c == "IA" else HUM for c in top["Mais próximo de"]]
    fmt = lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    hover = [f"<b>{i}</b><br>Este texto: {fmt(v)}<br>Média humanos: {fmt(h)}<br>Média IA: {fmt(a)}"
             f"<br>Mais próximo de: {c}" for i, v, h, a, c in
             zip(top["Indicador"], top["Este texto"], top["Média humanos"], top["Média IA"], top["Mais próximo de"])]
    fig = go.Figure()
    fig.add_vrect(x0=-1.05, x1=0.5, fillcolor=HUM_T, line_width=0, layer="below")
    fig.add_vrect(x0=0.5, x1=2.05, fillcolor=IA_T, line_width=0, layer="below")
    fig.add_vline(x=0, line=dict(color=HUM, width=1.5, dash="dash"))
    fig.add_vline(x=1, line=dict(color=IA, width=1.5, dash="dash"))
    fig.add_annotation(x=0, y=1.06, yref="paper", text="média dos humanos", showarrow=False, font=dict(color=HUM))
    fig.add_annotation(x=1, y=1.06, yref="paper", text="média da IA", showarrow=False, font=dict(color=IA))
    fig.add_trace(go.Scatter(x=pos, y=top["Indicador"], mode="markers", hovertext=hover, hoverinfo="text",
                             marker=dict(size=13, color=cores, symbol=simb, line=dict(color="#fcfcfb", width=2)),
                             name="este texto", showlegend=False))
    if any(s != "circle" for s in simb):
        fig.add_annotation(x=1, xref="paper", y=-0.16, yref="paper", xanchor="right", showarrow=False,
                           text="▶ / ◀ = valor além do limite do gráfico", font=dict(color=MUTED, size=11))
    fig.update_xaxes(range=[-1.05, 2.05], tickvals=[-1, 0, 0.5, 1, 2],
                     ticktext=["mais humano", "0 (humanos)", "meio", "1 (IA)", "mais IA"], zeroline=False)
    return _base(fig, 450, margin=dict(l=10, r=20, t=40, b=75))


def ngramas(ngr, k=10):
    if ngr is None or ngr.empty:
        return None
    d = ngr.sort_values("contrib")
    neg, pos = d.head(k), d.tail(k)
    sel = pd.concat([neg[neg.contrib < 0], pos[pos.contrib > 0]]).drop_duplicates("ngrama").sort_values("contrib")
    rot = ["“" + g.replace(" ", "␣") + "”" for g in sel.ngrama]
    cores = [IA if c > 0 else HUM for c in sel.contrib]
    fig = go.Figure(go.Bar(x=sel.contrib, y=rot, orientation="h", marker=dict(color=cores, cornerradius=4),
                           hovertemplate="<b>%{y}</b><br>contribuição: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
    fig.update_xaxes(title="← puxa para humano · contribuição no modelo clássico · puxa para IA →")
    fig.update_yaxes(tickfont=dict(size=13, color=INK, family="Consolas, monospace"))
    return _base(fig, 120 + 24 * len(sel), showlegend=False)


def robustez(d):
    if d is None or d.empty:
        return None
    d = d.copy()
    d["exp"] = d.experimento.str.replace("→", "->")
    ordem = list(dict.fromkeys(d["exp"]))
    fig = go.Figure()
    for nome, cor in [(d.detector.unique()[0], M_CLS), ("BERTimbau (fine-tuning)", M_BERT)]:
        s = d[d.detector == nome].set_index("exp").reindex(ordem)
        fig.add_trace(go.Bar(y=ordem, x=s.fpr * 100, orientation="h", name=nome,
                             marker=dict(color=cor, cornerradius=3, line=dict(color="#fcfcfb", width=1)),
                             hovertemplate="<b>%{y}</b><br>" + nome + "<br>Humanos marcados como IA: %{x:.1f}%<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
    fig.update_xaxes(ticksuffix="%", title="FPR: % de textos HUMANOS classificados como IA (quanto menor, melhor)")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 540, margin=dict(l=10, r=20, t=50, b=50),
                 legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
AVISO_CURTO = ("<b>Estimativa experimental, não prova de autoria.</b> O resultado é uma probabilidade estatística "
               "que pode errar, inclusive marcando como IA um texto escrito por uma pessoa. "
               "Não use este resultado para acusar ou punir ninguém.")


def cartao(r):
    f = r.faixa
    cor = COR_FAIXA[f]
    icone = {"ia": "▲", "humano": "▼", "inconclusivo": "◆"}[f]
    avisos = "".join(f"<li>{html.escape(a)}</li>" for a in r.avisos)
    return f"""
<div class="cartao" style="border-left:6px solid {cor}">
  <div class="cm-tipo">Veredito combinado (média dos dois modelos)</div>
  <div class="cartao-rotulo" style="color:{cor}">{icone} {ROTULO[f]}</div>
  <div class="cartao-prob">Probabilidade estimada de IA: <b>{pct(r.p_final)}</b></div>
  <div class="cartao-sub">{r.n_palavras} palavras · {len(r.chunks)} trecho(s) analisado(s) ·
     faixas: humano ≤ {T_HUMAN:.0%} · IA ≥ {T_AI:.0%}</div>
  {f'<ul class="cartao-avisos">{avisos}</ul>' if avisos else ''}
  <div class="cartao-aviso">{AVISO_CURTO}</div>
</div>"""


def cartao_modelo(tipo, nome, chave_metricas, p):
    """Conclusão de UM modelo, binária (corte em 50%)."""
    f = faixa_modelo(p)
    cor = COR_FAIXA[f]
    icone = "▲" if f == "ia" else "▼"
    m = META.get("test_metrics", {}).get(chave_metricas, {})
    met = (f"No teste: acurácia {m['acc']:.0%} · humanos marcados como IA {m['fpr']:.0%}" if m else "")
    return f"""
<div class="cartao cartao-modelo" style="border-top:5px solid {cor}">
  <div class="cm-tipo">{tipo}</div>
  <div class="cm-nome">{html.escape(nome)}</div>
  <div class="cm-rotulo" style="color:{cor}">{icone} {ROTULO_MODELO[f]}</div>
  <div class="cm-prob">P(IA) = <b>{pct(p)}</b> <span class="cartao-sub">· corte em 50%</span></div>
  <div class="cartao-sub">{met}</div>
</div>"""


def cartoes_modelos(r):
    return (cartao_modelo("Modelo clássico", META.get("classico", "TF-IDF caracteres + LR"),
                          META.get("classico", ""), r.p_classico),
            cartao_modelo("Transformer", "BERTimbau (fine-tuning)", "BERTimbau (fine-tuning)", r.p_bert))


def destaque(r):
    """Texto com cada palavra colorida pela contribuição no modelo clássico."""
    if not r.palavras_trechos:
        return "<p>Explicação indisponível para este tipo de modelo clássico.</p>"
    mx = max((abs(s) for t in r.palavras_trechos for _, s in t), default=1) or 1
    partes = []
    for i, trecho in enumerate(r.palavras_trechos):
        ws = []
        for w, s in trecho:
            a = min(abs(s) / mx, 1) * 0.55
            cor = f"rgba(227,73,72,{a:.2f})" if s > 0 else f"rgba(42,120,214,{a:.2f})"
            tip = f"{'IA' if s > 0 else 'humano'} {s:+.3f}"
            ws.append(f'<span class="w" style="background:{cor}" title="{tip}">{html.escape(w)}</span>')
        cab = (f'<div class="trecho-cab">Trecho {i + 1} · clássico {pct(r.p_classico_trechos[i])} · '
               f'BERTimbau {pct(r.p_bert_trechos[i])}</div>') if len(r.palavras_trechos) > 1 else ""
        partes.append(f'<div class="trecho">{cab}<p>{" ".join(ws)}</p></div>')
    if r.resto:
        partes.append(f'<div class="trecho resto"><div class="trecho-cab">Não analisado (menos de 50 palavras)</div>'
                      f'<p>{html.escape(r.resto)}</p></div>')
    legenda = ('<div class="legenda"><span class="w" style="background:rgba(227,73,72,0.45)">vermelho</span> puxou '
               'para IA · <span class="w" style="background:rgba(42,120,214,0.45)">azul</span> puxou para humano · '
               'intensidade = força. Passe o mouse sobre uma palavra para ver o valor.</div>')
    return f'<div class="destaque">{legenda}{"".join(partes)}</div>'
