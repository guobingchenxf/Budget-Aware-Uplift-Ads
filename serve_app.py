"""


"""
import os

from baua.serve import create_app
from baua.persistence import load_bundle

MODELS = os.environ.get("BAUA_MODELS", os.path.join("artifacts", "models", "served"))
app = create_app(load_bundle(MODELS))
