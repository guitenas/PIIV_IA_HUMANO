# PIIV_IA_HUMANO
Repositório para o desenvolvimento do projeto integrador em suas respectivas versões

## V4 (18-11): Humano ou IA? Plataforma de análise de autoria textual

Projeto Integrador IV · FATEC Jundiaí
Allan Vytor Francisco Oliveira, Gabriel Passos e Guilherme Tenório Cavalcante

> O resultado da plataforma é uma estimativa experimental, não uma prova de autoria.

| Na pasta [`V4. (18-11)`](V4.%20(18-11)) | Conteúdo |
|---|---|
| `PIIV Humano X I.A.ipynb` | Notebook com todas as etapas do projeto |
| `Artigo/` | Artigo (versão V4) |
| `Apresentação Humano ou IA.pdf` | Apresentação |
| `Plataforma/` | Código da plataforma (veja o [README](V4.%20(18-11)/Plataforma/README.md)) |
| `Resultados/` | Tabelas de resultados (`data/`) e figuras (`figs/`) |
| `Instalador/LEIA-ME.txt` | Como instalar e usar o programa no Windows |

### Instalador e modelos (na Release)

O instalador e os modelos treinados passam do limite de 100 MB por arquivo do GitHub. Por isso estão na
Release [**V4 (18-11)**](https://github.com/guitenas/PIIV_IA_HUMANO/releases/tag/v4.0):

- **`HumanoOuIA_Setup_1.0.0.exe`**: instalador para Windows 10/11 (64 bits). Não precisa de Python. Veja
  `Instalador/LEIA-ME.txt`.
- **`modelos_plataforma_v4.zip`**: necessário só para rodar a plataforma a partir do código. Descompacte dentro de
  `V4. (18-11)/Plataforma/`, de modo que fiquem `Plataforma/models/classico.joblib` e
  `Plataforma/models/bertimbau/`. Depois siga a seção "Rodar a partir do código" do README da plataforma.

### O que não está no repositório

Os CSVs com o texto integral do corpus (dataset bruto, dataset limpo e gerações dos modelos) não foram publicados,
como informa o artigo. Por isso o script `Plataforma/ferramentas/validar_modelos.py` não roda a partir deste
repositório.
