"""
app.py - Plataforma de análise de autoria textual (humano vs. IA) · interface Gradio

Projeto: Humano ou IA? Identificação de autoria textual em português (FATEC Jundiaí)
Autores: Allan Vytor Francisco Oliveira, Gabriel Passos, Guilherme Tenório Cavalcante

Execução (na pasta Plataforma):
    .venv\\Scripts\\python app.py      ->  http://127.0.0.1:7860
ou dê dois cliques em iniciar.bat.
"""
import json
import os

# Privacidade: nada sai deste computador. Desliga a telemetria do Gradio e impede qualquer acesso
# ao Hugging Face Hub (os modelos são lidos só da pasta models/). Precisa vir ANTES de importar gradio/transformers.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Console limpo para quem usa o executável: sem avisos de depreciação do Gradio, sem barras de progresso do
# transformers e sem o ruído inofensivo "ConnectionResetError [WinError 10054]" do asyncio no Windows.
import logging  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

import gradio as gr  # noqa: E402
import pandas as pd

import charts
import core
from stylometry import TRUNC_WORDS

EXEMPLOS_PATH = os.path.join(core.DATA, "exemplos.json")
EXEMPLOS = json.load(open(EXEMPLOS_PATH, encoding="utf-8")) if os.path.exists(EXEMPLOS_PATH) else []

AVISO_TOPO = f"""
<div class="aviso-topo">
  <div class="aviso-icone">!</div>
  <div><b>Estimativa experimental, não prova de autoria.</b> Os modelos foram treinados com trechos de
  ~{TRUNC_WORDS} palavras em português (Wikipédia e notícias) e com textos de dois modelos de linguagem abertos
  (Qwen 2.5 1.5B e Phi-3.5 mini). O resultado é uma probabilidade estatística que pode errar, inclusive marcando
  como IA um texto escrito por uma pessoa. <b>Não use este resultado para acusar ou punir ninguém.</b></div>
</div>"""

CSS = """
.gradio-container {max-width: 1200px !important; margin: 0 auto;}
.cabecalho h1 {margin-bottom: 2px;}
.cabecalho p {color: #52514e; margin-top: 0;}
.aviso-topo {display:flex; gap:12px; align-items:flex-start; background:#fff7e6; border:1px solid #f1d29a;
  border-radius:10px; padding:12px 14px; color:#3d2e00; font-size:14px; line-height:1.45;}
.aviso-icone {flex:0 0 26px; height:26px; border-radius:50%; background:#fab219; color:#0b0b0b;
  font-weight:700; display:flex; align-items:center; justify-content:center;}
.cartao {background:#fcfcfb; border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:16px 18px;}
.cartao-rotulo {font-size:24px; font-weight:700; margin-bottom:6px;}
.cartao-prob {font-size:16px; color:#0b0b0b; margin-bottom:4px;}
.cartao-sub {color:#52514e; font-size:13px;}
.cartao-avisos {margin:10px 0 0 0; padding-left:18px; color:#3d2e00; font-size:13px;}
.cm-tipo {font-size:11.5px; color:#898781; text-transform:uppercase; letter-spacing:.04em; margin-bottom:2px;}
.cm-nome {font-size:13px; color:#52514e; margin-bottom:6px;}
.cm-rotulo {font-size:22px; font-weight:700; margin-bottom:4px;}
.cm-prob {font-size:14px; color:#0b0b0b; margin-bottom:4px;}
.cartao-modelo {height:100%;}
.titulo-modelos {font-size:13px; color:#52514e; margin:4px 0 -4px 2px;}
.cartao-aviso {margin-top:12px; padding-top:10px; border-top:1px solid #e1e0d9; font-size:12.5px; color:#52514e;}
.destaque {font-size:15px; line-height:2.0;}
.destaque .w {border-radius:3px; padding:1px 2px;}
.destaque .legenda {font-size:13px; color:#52514e; margin-bottom:8px; line-height:1.8;}
.trecho {border:1px solid #e1e0d9; border-radius:8px; padding:6px 12px; margin-bottom:10px; background:#fcfcfb;}
.trecho p {margin:4px 0;}
.trecho-cab {font-size:12px; color:#898781; text-transform:uppercase; letter-spacing:.03em;}
.resto p {color:#898781;}
.nota {font-size:13px; color:#52514e;}
footer {display:none !important;}
.modebar-container {display:none !important;}
"""

def tema_unico():
    """Tema sem diferença entre claro e escuro: cada variável *_dark recebe o valor da versão clara.
    Assim, mesmo que o sistema/navegador esteja em modo escuro, a página fica idêntica à do modo claro."""
    tema = gr.themes.Soft(primary_hue="blue")
    for k in [k for k in vars(tema) if k.endswith("_dark")]:
        if hasattr(tema, k[:-5]):
            setattr(tema, k, getattr(tema, k[:-5]))
    return tema


HEAD_SEM_MODO_ESCURO = '<meta name="color-scheme" content="light">'

VAZIO = (None,) * 11


def tabela_modelos(r):
    return pd.DataFrame([
        {"Modelo": core.NOME_CLASSICO, "P(IA)": core.pct(r.p_classico, 1),
         "Classificação": core.ROTULO_MODELO[core.faixa_modelo(r.p_classico)] + " (corte 50%)"},
        {"Modelo": core.NOME_BERT, "P(IA)": core.pct(r.p_bert, 1),
         "Classificação": core.ROTULO_MODELO[core.faixa_modelo(r.p_bert)] + " (corte 50%)"},
        {"Modelo": "Combinada (média dos dois)", "P(IA)": core.pct(r.p_final, 1),
         "Classificação": core.ROTULO[r.faixa] + (" (modelos discordam)" if r.discordam else "")},
    ])


def tabela_indicadores(tab):
    t = tab.reindex(tab["effect"].abs().sort_values(ascending=False).index)
    t = t[["Indicador", "Este texto", "Média humanos", "Média IA", "Mais próximo de"]].copy()
    for c in ["Este texto", "Média humanos", "Média IA"]:
        t[c] = t[c].map(lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")
    return t


def _falha(msg):
    return (gr.update(value=f"⚠️ {msg}", visible=True), gr.update(visible=False), *VAZIO,
            gr.update(visible=False), gr.update(visible=False))


def analisar_ui(texto):
    try:
        return _analisar_ui(texto)
    except Exception as e:  # mostra o erro na tela em vez de falhar em silêncio
        import traceback; traceback.print_exc()
        return _falha(f"Erro inesperado na análise: {type(e).__name__}: {e}")


def _analisar_ui(texto):
    r = core.analisar(texto)
    if not r.ok:
        return _falha(r.mensagem)
    multi = len(r.chunks) > 1
    return (gr.update(value="", visible=False), gr.update(visible=True),
            charts.cartao(r), *charts.cartoes_modelos(r),
            charts.medidor(r.p_final, r.faixa), charts.modelos(r), tabela_modelos(r),
            charts.trechos(r) if multi else None,
            charts.indicadores(r.indicadores), tabela_indicadores(r.indicadores),
            charts.destaque(r), charts.ngramas(r.ngramas),
            gr.update(visible=multi), gr.update(visible=True))


def contar(texto):
    n = len((texto or "").split())
    dica = "ideal" if n >= TRUNC_WORDS else ("baixa confiabilidade" if n < core.MIN_WORDS_WARN else "aceitável")
    return f"<span class='nota'>{n} palavras · {dica} (o treino usou trechos de {TRUNC_WORDS} palavras)</span>"


def rodape():
    linhas = [f"{n}: AUROC {m['auroc']:.3f} · F1 {m['f1']:.3f} · humanos marcados como IA (FPR) {m['fpr']:.1%}"
              for n, m in core.META.get("test_metrics", {}).items()]
    return ("<div class='nota'><b>Desempenho no conjunto de teste (mesma distribuição do treino):</b> "
            + " | ".join(linhas) + f"<br>Modelos treinados em {core.META.get('criado_em', '?')[:10]}. "
            "Projeto Integrador · FATEC Jundiaí.</div>")


SOBRE_COMO = f"""
### Como a análise funciona
1. **Limpeza.** O texto passa pelo mesmo pré-processamento do treino (`stylometry.preprocess`): normalização Unicode,
   remoção de URLs, e-mails, emojis e marcações, e padronização de aspas e travessões.
2. **Divisão em trechos.** Os modelos foram treinados com trechos de {TRUNC_WORDS} palavras. Textos maiores são
   divididos em trechos desse tamanho (o último só entra se tiver 50 palavras ou mais), e as probabilidades são a
   média dos trechos.
3. **Dois modelos independentes.**
   - **Clássico ({core.META.get('classico')})**: n-gramas de 3 a 5 caracteres, ponderados por TF-IDF, com regressão
     logística. Foi o melhor modelo clássico na **validação**.
   - **Transformer (BERTimbau com fine-tuning)**: `{core.META.get('bert_model')}`, ajustado para a tarefa
     (até {core.MAX_LEN} tokens por trecho).
4. **Conclusão de cada modelo, separadamente**: IA se a P(IA) do modelo for ≥ 50%, senão humano. É o mesmo corte
   usado nas métricas por modelo do notebook (acurácia, FPR).
   **Probabilidade combinada** = média simples das duas probabilidades.
5. **Classificação em três faixas**:
   - P(IA) ≤ {core.T_HUMAN:.2f} → *provavelmente humano*;
   - P(IA) ≥ {core.T_AI:.3f} → *provavelmente IA*. Esse limiar é o maior valor entre 0,65 e o ponto que deixa no
     máximo 5% dos humanos da validação acima dele;
   - entre os dois → *inconclusivo*.
6. **Regra de discordância.** Se as probabilidades dos dois modelos diferirem em mais de
   {core.DISAGREE * 100:.0f} pontos percentuais, o resultado é **sempre inconclusivo**, mesmo que a média caia
   em outra faixa. A média de duas opiniões opostas não é evidência de nada.

### Como ler as visualizações
- **Indicadores estilométricos** são **descritivos**: mostram se o texto se parece mais com a média humana ou com a
  média de IA do treino. O modelo clássico exportado **não** usa esses indicadores, então eles não explicam
  diretamente a classificação.
- **"O que pesou no modelo clássico"** mostra a contribuição de cada n-grama de caracteres no cálculo da regressão
  logística. Essa explicação vale **só para o modelo clássico**: o BERTimbau não é explicado aqui.
"""

SOBRE_LIMITES = """
### Limitações
- **Domínio.** O treino usou apenas textos da Wikipédia e de portais de notícias em português. Redações, posts,
  e-mails, textos técnicos ou literários estão fora desse domínio.
- **Geradores.** Os textos de IA do treino vieram de dois modelos pequenos e abertos (Qwen 2.5 1.5B e Phi-3.5 mini).
  O teste com o Gemini (gerador externo) foi bom, mas outros modelos ou outras instruções de geração podem enganar
  os detectores.
- **Tamanho.** Abaixo de ~50 palavras o desempenho cai (veja "curto (30/50 palavras)" no gráfico).
- **Edição humana.** Textos de IA editados por pessoas, ou textos humanos revisados por IA, não foram testados.
- **Amostra.** O conjunto tem cerca de 1.200 textos. As métricas são estimativas com incerteza, não valores exatos.

### Ética
Detectores de IA erram, e o erro mais grave é **acusar uma pessoa de usar IA quando ela não usou** (falso positivo,
FPR). Por isso esta plataforma:
- usa uma faixa "inconclusivo" em vez de forçar uma decisão;
- mostra os dois modelos separadamente e avisa quando eles discordam;
- deixa claro que o resultado é uma **estimativa experimental, e não uma prova de autoria**.

O texto analisado é processado **apenas localmente**, neste computador: nada é enviado para a internet nem salvo.
"""

with gr.Blocks(title="Humano ou IA? · Análise de autoria", theme=tema_unico(), css=CSS,
               head=HEAD_SEM_MODO_ESCURO) as demo:
    gr.HTML("<div class='cabecalho'><h1>Humano ou IA? Análise de autoria textual</h1>"
            "<p>Projeto Integrador · FATEC Jundiaí · Allan Vytor Francisco Oliveira, Gabriel Passos e "
            "Guilherme Tenório Cavalcante</p></div>")
    gr.HTML(AVISO_TOPO)

    with gr.Tabs():
        with gr.Tab("Analisar texto"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    entrada = gr.Textbox(lines=13, max_lines=30, label="Texto em português",
                                         placeholder=f"Cole aqui o texto (ideal: {TRUNC_WORDS} palavras ou mais)")
                    contador = gr.HTML(contar(""))
                    with gr.Row():
                        botao = gr.Button("Analisar", variant="primary", scale=3)
                        limpar = gr.Button("Limpar", scale=1)
                    if EXEMPLOS:
                        gr.Examples(examples=[[e["texto"]] for e in EXEMPLOS], inputs=[entrada],
                                    example_labels=[e["titulo"] for e in EXEMPLOS],
                                    label="Exemplos (textos do conjunto de teste, nunca vistos no treino, e um do Gemini)")
                with gr.Column(scale=2):
                    erro = gr.Markdown(visible=False)
                    with gr.Column(visible=False) as col_resumo:
                        out_cartao = gr.HTML()
                        gr.HTML("<div class='titulo-modelos'>Conclusão de cada modelo, separadamente:</div>")
                        with gr.Row(equal_height=True):
                            out_cartao_cls = gr.HTML()
                            out_cartao_bert = gr.HTML()
                        out_medidor = gr.Plot(show_label=False)

            with gr.Column(visible=False) as col_detalhes:
                gr.Markdown("### Resultado por modelo")
                with gr.Row():
                    with gr.Column(scale=3):
                        out_modelos = gr.Plot(show_label=False)
                    with gr.Column(scale=2):
                        out_tab_modelos = gr.Dataframe(interactive=False, wrap=True, show_label=False)
                with gr.Column(visible=False) as col_trechos:
                    gr.Markdown("### Probabilidade por trecho\n<span class='nota'>Cada ponto é um trecho de "
                                f"~{TRUNC_WORDS} palavras. Trechos muito diferentes entre si sugerem um texto "
                                "misto ou um resultado instável.</span>")
                    out_trechos = gr.Plot(show_label=False)

                with gr.Tabs():
                    with gr.Tab("Indicadores estilométricos"):
                        gr.Markdown("<span class='nota'>Posição do texto entre a média dos humanos (0) e a média "
                                    "da IA (1) do treino, nos 10 indicadores que mais separam as duas classes. "
                                    "Os indicadores são <b>descritivos</b>: o modelo clássico exportado não os usa "
                                    "diretamente.</span>")
                        out_ind = gr.Plot(show_label=False)
                        out_tab_ind = gr.Dataframe(interactive=False, wrap=True,
                                                   label="Todos os 24 indicadores (ordenados pelo poder de separação no treino)")
                    with gr.Tab("O que pesou no modelo clássico"):
                        gr.Markdown("<span class='nota'>Contribuição de cada trecho do texto para a decisão do "
                                    "modelo clássico (TF-IDF de caracteres + regressão logística). Vale só para o "
                                    "modelo clássico; o BERTimbau não é explicado aqui.</span>")
                        out_destaque = gr.HTML()
                        gr.Markdown("#### N-gramas de caracteres com maior peso neste texto\n"
                                    "<span class='nota'>Sequências de 3 a 5 caracteres; ␣ indica espaço (início "
                                    "ou fim de palavra).</span>")
                        out_ngr = gr.Plot(show_label=False)

        with gr.Tab("Sobre os modelos e limitações"):
            gr.Markdown(SOBRE_COMO)
            gr.Markdown("### Desempenho no conjunto de teste (mesma distribuição do treino)")
            gr.Dataframe(core.metricas_teste(), interactive=False, wrap=True, show_label=False)
            rob = core.robustez()
            if rob is not None:
                gr.Markdown("### Robustez: e quando as condições mudam?\n<span class='nota'>Taxa de textos "
                            "<b>humanos</b> classificados como IA (FPR) em cada experimento da Etapa 11 do "
                            "notebook. As barras mais longas indicam as situações de maior risco de acusação "
                            "injusta.</span>")
                gr.Plot(charts.robustez(rob), show_label=False)
                gr.Dataframe(rob[["experimento", "detector", "auroc", "f1", "fpr"]]
                             .rename(columns={"auroc": "AUROC", "f1": "F1", "fpr": "FPR"}).round(3),
                             interactive=False, wrap=True, label="Tabela de robustez")
            gr.Markdown(SOBRE_LIMITES)

    gr.HTML(rodape())

    saidas = [erro, col_resumo, out_cartao, out_cartao_cls, out_cartao_bert, out_medidor, out_modelos, out_tab_modelos, out_trechos,
              out_ind, out_tab_ind, out_destaque, out_ngr, col_trechos, col_detalhes]
    botao.click(analisar_ui, inputs=entrada, outputs=saidas)
    entrada.change(contar, inputs=entrada, outputs=contador)
    limpar.click(lambda: ("", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
                 outputs=[entrada, col_resumo, col_detalhes, erro])

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print(" Humano ou IA? - plataforma de análise de autoria (uso local)")
    print(" O navegador vai abrir sozinho. Se não abrir, use o endereço abaixo.")
    print(" Para ENCERRAR a plataforma, feche esta janela.")
    print("=" * 64 + "\n")
    # Porta: 7860 por padrão; se estiver ocupada, o Gradio procura a próxima livre (7861, 7862...).
    porta = int(os.environ["PORTA"]) if os.environ.get("PORTA") else None
    demo.launch(server_name="127.0.0.1", server_port=porta, inbrowser=os.environ.get("ABRIR_NAVEGADOR", "1") == "1",
                show_api=False)
