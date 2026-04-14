import os
import logging

_MODELS_DIR = os.path.dirname(__file__)

HF_REPO_ID = "jasperlinthorst/cfstats-models"

log = logging.getLogger(__name__)

def get_model_path(filename):
    """Return the absolute path to a bundled model file."""
    return os.path.join(_MODELS_DIR, filename)


def get_hf_model_path(filename, repo_id=None):
    """Download a model file from Hugging Face Hub on first use and return its local path.

    The file is cached locally by huggingface_hub so subsequent calls are instant.
    """
    from huggingface_hub import hf_hub_download

    repo = repo_id or HF_REPO_ID
    log.info("Resolving model '%s' from HF repo '%s' (downloads on first use)", filename, repo)
    local_path = hf_hub_download(repo_id=repo, filename=filename)
    log.info("Model resolved to: %s", local_path)
    return local_path
