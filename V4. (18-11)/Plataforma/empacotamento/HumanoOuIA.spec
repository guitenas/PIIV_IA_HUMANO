# -*- mode: python ; coding: utf-8 -*-
# PyInstaller: empacota a plataforma em uma pasta com HumanoOuIA.exe (não precisa de Python instalado).
# Uso (na pasta Plataforma):  .venv\Scripts\python -m PyInstaller empacotamento\HumanoOuIA.spec
import os

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

PLAT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [
    (os.path.join(PLAT, "models"), "models"),
    (os.path.join(PLAT, "data", "exemplos.json"), "data"),
    (os.path.join(PLAT, "data", "resultados_robustez.csv"), "data"),
]
binaries, hiddenimports = [], []

# Gradio e dependências leem arquivos de dados (frontend, versões, templates) em tempo de execução
for pkg in ["gradio", "gradio_client", "safehttpx", "groovy"]:
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
datas += collect_data_files("transformers", include_py_files=False)
for pkg in ["gradio", "gradio_client", "transformers", "tokenizers", "safetensors", "huggingface-hub", "torch",
            "numpy", "scikit-learn", "tqdm", "regex", "requests", "packaging", "filelock", "pyyaml", "httpx",
            "fastapi", "starlette", "uvicorn", "pydantic", "plotly", "pandas"]:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass
hiddenimports += ["sklearn.feature_extraction.text", "sklearn.linear_model._logistic", "sklearn.pipeline",
                  "sklearn.utils._typedefs", "sklearn.utils._heap", "sklearn.utils._sorting",
                  "sklearn.utils._vector_sentinel", "sklearn.neighbors._partition_nodes",
                  "plotly.graph_objects", "plotly.io"]

a = Analysis(
    [os.path.join(PLAT, "app.py")],
    pathex=[PLAT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["IPython", "jupyter", "notebook", "tkinter", "torchvision", "torchaudio",
              "tensorflow", "keras", "jax", "flax", "pytest"],
    # O Gradio lê o código-fonte dos próprios componentes: precisa dos .py, não só do bytecode
    module_collection_mode={"gradio": "py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="HumanoOuIA",
    console=True,            # a janela preta mostra o endereço e, ao ser fechada, encerra a plataforma
    icon=os.path.join(SPECPATH, "icone.ico"),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="HumanoOuIA")
