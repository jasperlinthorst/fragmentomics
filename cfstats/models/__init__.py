import os

_MODELS_DIR = os.path.dirname(__file__)

def get_model_path(filename):
    """Return the absolute path to a bundled model file."""
    return os.path.join(_MODELS_DIR, filename)
