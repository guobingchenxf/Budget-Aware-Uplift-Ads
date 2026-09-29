"""uvicorn 入口：从环境变量 BAUA_MODELS 指定的目录加载模型并暴露 ASGI app。

用法：
    set BAUA_MODELS=artifacts/models/served
    .venv\Scripts\python.exe -m uvicorn serve_app:app --port 8000
（或直接用 `python -m baua.cli serve --models ...`，它会自动处理这一切）
"""
import os

from baua.serve import create_app
from baua.persistence import load_bundle

MODELS = os.environ.get("BAUA_MODELS", os.path.join("artifacts", "models", "served"))
app = create_app(load_bundle(MODELS))
