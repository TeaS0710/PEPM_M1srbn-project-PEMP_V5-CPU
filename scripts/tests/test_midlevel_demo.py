"""Smoke-test d'intégration sur le mini corpus TEI web1.

1. Garantit la présence du corpus de démo.
2. Résout le profil ideo_quick avec overrides pour web1.
3. Exécute prepare -> train -> evaluate (sklearn) via `make run`.
4. Vérifie la présence des artefacts clés.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.core.core_utils import resolve_profile_base
DEST_CORPUS = ROOT / "data/raw/web1/corpus.xml"
DEFAULT_SRC = Path("/mnt/data/corpus.xml")


def ensure_corpus() -> None:
    if DEST_CORPUS.exists():
        return
    if DEFAULT_SRC.exists():
        DEST_CORPUS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DEFAULT_SRC, DEST_CORPUS)
    else:
        raise FileNotFoundError("Corpus TEI de démo introuvable")


def run_cmd(args):
    subprocess.run(args, check=True, cwd=ROOT)


def main() -> None:
    ensure_corpus()

    params = resolve_profile_base("ideo_quick", overrides=["corpus_id=web1"])
    assert Path(params["corpus"]["corpus_path"]).resolve() == DEST_CORPUS

    run_cmd(["make", "run", "STAGE=prepare", "PROFILE=ideo_quick", "CORPUS_ID=web1", "VIEW=ideology_global"])
    run_cmd(
        [
            "make",
            "run",
            "STAGE=train",
            "PROFILE=ideo_quick",
            "CORPUS_ID=web1",
            "VIEW=ideology_global",
            "FAMILY=sklearn",
        ]
    )
    run_cmd(
        [
            "make",
            "run",
            "STAGE=evaluate",
            "PROFILE=ideo_quick",
            "CORPUS_ID=web1",
            "VIEW=ideology_global",
            "FAMILY=sklearn",
        ]
    )

    train_path = ROOT / "data/interim/web1/ideology_global/train.tsv"
    metrics_path = ROOT / "reports/web1/ideology_global/sklearn/tfidf_svm_quick/metrics.json"
    assert train_path.exists() and train_path.stat().st_size > 0
    assert metrics_path.exists() and metrics_path.stat().st_size > 0


if __name__ == "__main__":
    main()
