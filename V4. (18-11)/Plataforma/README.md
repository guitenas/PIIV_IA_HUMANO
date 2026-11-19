# Humano ou IA? · Plataforma de análise de autoria textual

Projeto Integrador IV · FATEC Jundiaí
Allan Vytor Francisco Oliveira, Gabriel Passos e Guilherme Tenório Cavalcante

> **O resultado é uma estimativa experimental, não uma prova de autoria.**

## Instalar sem Python (Windows 10/11, 64 bits)

1. Execute `Instalador\HumanoOuIA_Setup_1.0.0.exe`, na pasta `V4. (18-11)`.
2. Não precisa de administrador. O programa é instalado para o usuário atual e cria um atalho no menu Iniciar (e,
   se você marcar, na Área de Trabalho).
3. Abra **Humano ou IA**. Aparece uma janela preta (o servidor local) e, depois de ~15 s, o navegador abre a
   plataforma. **Para encerrar, feche a janela preta.**
4. Para desinstalar: Configurações → Aplicativos → "Humano ou IA?".

O Windows SmartScreen pode avisar que o programa "não é reconhecido", porque o instalador não tem assinatura
digital. Nesse caso, clique em **Mais informações → Executar assim mesmo**.

### Como o instalador é gerado

```powershell
.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\python empacotamento\gerar_icone.py
.venv\Scripts\python -m PyInstaller empacotamento\HumanoOuIA.spec --noconfirm
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" /DOrigem="$PWD\dist\HumanoOuIA" empacotamento\instalador.iss
```

O PyInstaller junta Python, bibliotecas e modelos numa pasta com `HumanoOuIA.exe`, e o Inno Setup compacta essa
pasta no instalador.

## Rodar a partir do código (com Python)

**Jeito mais fácil:** dê dois cliques em `iniciar.bat`. Na primeira vez, ele cria o ambiente e instala as
dependências (cerca de 1 GB). Depois disso, só abre o app no navegador em http://127.0.0.1:7860.

**Pelo terminal (PowerShell), dentro desta pasta:**

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

Tudo roda **localmente, na CPU**: nenhum texto é enviado para a internet.

## O que a interface mostra (requisitos do item 8.5)

| Requisito | Onde aparece |
|---|---|
| Classificação humano/IA | Cartão de resultado, em três faixas: provavelmente humano (P(IA) ≤ 0,35) · inconclusivo · provavelmente IA (P(IA) ≥ 0,661). Se os modelos discordarem em mais de 40 pontos, o resultado é sempre inconclusivo |
| Probabilidade estimada | Medidor com a P(IA) combinada (média simples dos dois modelos) |
| Indicadores estilométricos | Aba "Indicadores estilométricos": gráfico dos 10 mais discriminativos e tabela com os 24 |
| Resultado do modelo clássico | Cartão próprio com a conclusão **IA ou Humano** (corte em 50%), além do gráfico e da tabela "Resultado por modelo" (TF-IDF de caracteres + regressão logística) |
| Resultado do Transformer | Cartão próprio com a conclusão **IA ou Humano** (corte em 50%), além do mesmo gráfico e tabela (BERTimbau com fine-tuning) |
| Visualizações | Medidor; barras por modelo; probabilidade por trecho (textos longos); posição nos indicadores; texto destacado e n-gramas que mais pesaram no modelo clássico; robustez (aba "Sobre") |
| Aviso de estimativa experimental | Faixa fixa no topo, repetida no cartão de resultado e na aba "Sobre" |

## Arquivos

| Arquivo | Função |
|---|---|
| `app.py` | Interface (Gradio) |
| `core.py` | Carrega os modelos e faz a análise (pode ser testado sozinho: `python core.py`) |
| `charts.py` | Gráficos (Plotly) e trechos de HTML |
| `stylometry.py` | **Mesmo** módulo do notebook: pré-processamento e os 24 indicadores |
| `models/` | Artefatos exportados pelo notebook (Etapa 13.2): `classico.joblib`, `bertimbau/`, `meta.json`, `reference_stats.json` |
| `data/exemplos.json` | Textos de exemplo do conjunto de **teste** e um do Gemini (gerado por `ferramentas/validar_modelos.py`) |
| `data/resultados_robustez.csv` | Resultados da Etapa 11 (aba "Sobre") |
| `ferramentas/validar_modelos.py` | Recria a divisão de teste do notebook e confere se os modelos reproduzem as métricas |
| `app_original_notebook.py` | Versão do app gerada pelo notebook, mantida só como referência |

## Validação feita nesta máquina

`ferramentas/validar_modelos.py` (precisa da pasta `../Resultados/data`) reproduziu o notebook:

| Modelo | AUROC aqui | AUROC no notebook |
|---|---|---|
| TF-IDF caracteres + LR | 0,9569 | 0,9569 |
| BERTimbau (fine-tuning) | 0,9768 | 0,9767 |

Com o limiar de IA (0,661), a regra combinada marcou como IA 8,9% dos humanos e 94,4% dos textos de IA do teste,
iguais ao notebook.

**Efeito da regra de discordância no conjunto de teste (180 textos).** 25% dos textos caem na regra:

| Origem real | → humano | → inconclusivo | → IA |
|---|---|---|---|
| Humanos (90) | 65,6% | 31,1% | **3,3%** (era 8,9% sem a regra) |
| IA (90) | 1,1% | 22,2% | 76,7% (era 94,4% sem a regra) |

A regra reduz as acusações injustas (humanos marcados como IA) de 8,9% para 3,3%. O custo é deixar mais textos
como inconclusivos.

## Decisões de interpretação

- **Conclusão de cada modelo** (decisão do grupo, 19/11/2026): binária, **IA** se P(IA) ≥ 50%, senão **Humano**.
  É o mesmo corte das métricas por modelo do notebook. O veredito combinado continua sendo o resultado principal.
- **Aparência única:** a página é sempre clara, independentemente do modo claro/escuro do sistema ou do navegador.
- **Probabilidade combinada** = média simples das P(IA) dos dois modelos, como no notebook.
- **Regra de discordância (decisão do grupo, 19/11/2026):** se |P(IA) clássico − P(IA) BERTimbau| > 0,40, a
  classificação final é **inconclusivo**, independentemente da média. O notebook só exibia um aviso nesse caso;
  a plataforma passa a forçar a faixa inconclusiva. Isso torna a regra mais conservadora que a avaliada na Etapa 13.2
  (veja os números abaixo).
- **Textos longos** são divididos em trechos de 100 palavras (o tamanho usado no treino). A probabilidade é a média
  dos trechos, e o último trecho só entra se tiver 50 palavras ou mais.
- **Os indicadores estilométricos são descritivos.** O modelo clássico exportado usa n-gramas de caracteres, não
  esses indicadores.
- **A explicação por n-gramas vale só para o modelo clássico.** Ela é exata: a soma das contribuições mais o
  intercepto reproduz a probabilidade do modelo.
- **Probabilidades nunca aparecem como 0% ou 100%** (aparecem como "<1%" ou ">99%"), para não sugerir certeza.
- **Avisos automáticos** aparecem para: texto curto (< 50 palavras), modelos discordando em mais de 40 pontos,
  trecho final descartado e texto que parece não estar em português.
