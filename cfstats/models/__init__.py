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
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    repo = repo_id or HF_REPO_ID

    cached = try_to_load_from_cache(repo_id=repo, filename=filename)
    if cached is not None:
        log.info("Model '%s' found in local HF cache: %s", filename, cached)
        return cached

    log.info("Model '%s' not in cache; downloading from HF repo '%s' ...", filename, repo)
    local_path = hf_hub_download(repo_id=repo, filename=filename)
    log.info("Model downloaded and cached at: %s", local_path)
    return local_path


HF_SPACE_ID = "jasperlinthorst/cfstats-umap-api"


def remote_umap_transform(features, hf_token, space_id=None):
    """Call the remote UMAP FastAPI Space to transform feature vectors.

    Args:
        features: numpy array of shape (n_samples, n_features).
        hf_token: Hugging Face API token for authentication.
        space_id: Override the default Space ID.

    Returns:
        (embedding, k) where embedding is a numpy array of shape (n_samples, 2)
        and k is the kmer size used by the model.
    """
    import numpy as np
    import httpx

    space = space_id or HF_SPACE_ID
    base_url = f"https://{space.replace('/', '-')}.hf.space"
    url = f"{base_url}/transform"

    log.info("Calling remote UMAP API: %s (%d samples)", url, features.shape[0])
    resp = httpx.post(
        url,
        json={"features": features.tolist()},
        headers={"Authorization": f"Bearer {hf_token}"},
        timeout=300.0,
    )
    resp.raise_for_status()
    result = resp.json()

    embedding = np.array(result["embedding"], dtype=np.float64)
    k = int(result.get("k", 4))
    log.info("Remote UMAP API returned embedding shape %s (k=%d)", embedding.shape, k)
    return embedding, k


def _remote_post(endpoint, payload, hf_token, space_id=None):
    """Send a POST request to the remote cfstats API Space."""
    import httpx

    space = space_id or HF_SPACE_ID
    base_url = f"https://{space.replace('/', '-')}.hf.space"
    url = f"{base_url}/{endpoint}"

    log.info("Calling remote API: %s", url)
    resp = httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {hf_token}"},
        timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()


def remote_dnase1l3_predict(features, hf_token, feature_names=None, space_id=None):
    """Call the remote DNASE1L3 classifier endpoint.

    Args:
        features: numpy array of shape (n_samples, n_features).
        hf_token: Hugging Face API token for authentication.
        feature_names: optional list of feature column names.
        space_id: Override the default Space ID.

    Returns:
        (predictions, probabilities) where predictions is a list of class labels
        and probabilities is a list of [p_class0, p_class1] per sample.
    """
    import numpy as np

    log.info("Sending %d samples to remote DNASE1L3 API", features.shape[0])
    payload = {"features": features.tolist()}
    if feature_names:
        payload["feature_names"] = feature_names
    result = _remote_post("dnase1l3", payload, hf_token, space_id)

    preds = result["predictions"]
    probs = np.array(result["probabilities"], dtype=np.float64)
    log.info("Remote DNASE1L3 API returned %d predictions", len(preds))
    return preds, probs


def remote_ff_predict(columns, counts, hf_token, space_id=None):
    """Call the remote FF estimation endpoint.

    Args:
        columns: list of column names for the bincount data.
        counts: numpy array of shape (n_samples, n_bins).
        hf_token: Hugging Face API token for authentication.
        space_id: Override the default Space ID.

    Returns:
        list of predicted fetal fractions.
    """
    log.info("Sending %d samples to remote FF API", counts.shape[0])
    result = _remote_post(
        "ff",
        {"columns": columns, "counts": counts.tolist()},
        hf_token,
        space_id,
    )
    ffs = result["ff"]
    log.info("Remote FF API returned %d predictions", len(ffs))
    return ffs
