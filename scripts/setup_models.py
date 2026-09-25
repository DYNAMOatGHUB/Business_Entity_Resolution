"""Download model weights once into models/ directory for offline execution.
Models:
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache 2.0)
- intfloat/multilingual-e5-small (MIT)
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer


def download_models(target_dir: str = "models"):
    models_path = Path(target_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    b5_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Downloading {b5_model_name}...")
    model_b5 = SentenceTransformer(b5_model_name)
    model_b5.save(str(models_path / "paraphrase-multilingual-MiniLM-L12-v2"))

    ce_model_name = "intfloat/multilingual-e5-small"
    print(f"Downloading {ce_model_name}...")
    model_ce = SentenceTransformer(ce_model_name)
    model_ce.save(str(models_path / "multilingual-e5-small"))

    print("Model downloads complete.")


if __name__ == "__main__":
    download_models()
