# scripts/core/core_evaluate.py

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

from scripts.core.core_utils import (
    resolve_profile_base,
    debug_print_params,
    PIPELINE_VERSION,
    apply_global_seed,
    log,

)


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="V4 core_evaluate : évaluation multi-familles (spaCy, sklearn, HF, check)"
    )
    ap.add_argument(
        "--profile",
        required=True,
        help="Nom du profil (sans .yml) dans configs/profiles/",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config (clé=valeur, ex: view=ideology_global)",
    )
    ap.add_argument(
        "--only-family",
        choices=["spacy", "sklearn", "hf", "check"],
        help="Limiter l'évaluation à une seule famille (optionnel)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les paramètres résolus",
    )
    return ap.parse_args()


# ----------------- Utils généraux -----------------


def set_blas_threads(n_threads: int) -> None:
    """
    Limiter les threads BLAS (MKL/OPENBLAS/OMP) pour éviter la sur-souscription.
    Même en éval, ça peut éviter des surprises.
    """
    if n_threads is None or n_threads <= 0:
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    print(f"[core_evaluate] BLAS threads fixés à {n_threads}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_model_output_dir(params: Dict[str, Any], family: str, model_id: str) -> Path:
    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")
    return Path("models") / corpus_id / view / family / model_id


def get_reports_dir(params: Dict[str, Any], family: str, model_id: str) -> Path:
    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")
    return Path("reports") / corpus_id / view / family / model_id


def load_job_tsv(params: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Charger job.tsv (ou fallback train.tsv) depuis data/interim/{corpus_id}/{view}/
    Retourne (texts, labels).
    """
    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")
    interim_dir = Path("data") / "interim" / corpus_id / view
    job_path = interim_dir / "job.tsv"
    train_path = interim_dir / "train.tsv"

    target = job_path if job_path.exists() else train_path
    if not target.exists():
        raise SystemExit(f"[core_evaluate] Ni job.tsv ni train.tsv trouvés dans {interim_dir}")

    if target == train_path:
        print("[core_evaluate] WARNING: job.tsv absent, évaluation réalisée sur train.tsv (sur-apprentissage possible).")

    texts: List[str] = []
    labels: List[str] = []
    with target.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row.get("text") or ""
            label = row.get("label")
            if not text or not label:
                continue
            texts.append(text)
            labels.append(label)

    if not texts:
        raise SystemExit(f"[core_evaluate] Aucune donnée valide dans {target}")

    return texts, labels


def maybe_debug_subsample_eval(
    texts: List[str],
    labels: List[str],
    params: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Si debug_mode=True, limiter la taille du dataset d'évaluation.
    """
    if not params.get("debug_mode"):
        return texts, labels

    max_docs = 1000
    if len(texts) <= max_docs:
        return texts, labels

    seed = int(params.get("seed", 42))
    print(f"[core_evaluate] debug_mode actif : sous-échantillon de {max_docs} docs sur {len(texts)} (seed={seed})")
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    idx_sel = sorted(indices[:max_docs])
    texts_sub = [texts[i] for i in idx_sel]
    labels_sub = [labels[i] for i in idx_sel]
    return texts_sub, labels_sub


def compute_basic_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report_dict,
    }


def save_eval_outputs(
    params: Dict[str, Any],
    family: str,
    model_id: str,
    metrics: Dict[str, Any],
) -> None:
    reports_dir = get_reports_dir(params, family, model_id)
    ensure_dir(reports_dir)

    # metrics.json : numérique + report dict
    metrics_path = reports_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[core_evaluate] metrics.json écrit : {metrics_path}")

    # classification_report.txt : version textuelle
    # On regénère un report texte à partir du dict pour avoir quelque chose de lisible
    report_txt_path = reports_dir / "classification_report.txt"
    with report_txt_path.open("w", encoding="utf-8") as f:
        # Reconstruction simple du report global
        if "classification_report" in metrics:
            # On pourrait reformatter, mais pour V4-v1 on fait simple:
            f.write(json.dumps(metrics["classification_report"], ensure_ascii=False, indent=2))
        else:
            f.write("No classification_report field in metrics.\n")
    print(f"[core_evaluate] classification_report.txt écrit : {report_txt_path}")

    # meta_eval.json : contexte de l'évaluation
    meta_eval = {
        "profile": params.get("profile"),
        "corpus_id": params.get("corpus_id", params["corpus"].get("corpus_id")),
        "view": params.get("view"),
        "family": family,
        "model_id": model_id,
        "n_eval_docs": int(metrics.get("n_eval_docs", 0)),
        "pipeline_version": PIPELINE_VERSION,
    }
    meta_eval_path = reports_dir / "meta_eval.json"
    with meta_eval_path.open("w", encoding="utf-8") as f:
        json.dump(meta_eval, f, ensure_ascii=False, indent=2)
    print(f"[core_evaluate] meta_eval.json écrit : {meta_eval_path}")


# ----------------- Éval spaCy -----------------


def eval_spacy_model(params: Dict[str, Any], model_id: str) -> None:
    try:
        import spacy
        from spacy.tokens import DocBin
    except ImportError:
        raise SystemExit("[core_evaluate] spaCy n'est pas installé, impossible d'évaluer la famille 'spacy'.")

    model_dir = get_model_output_dir(params, "spacy", model_id)
    if not model_dir.exists():
        print(f"[core_evaluate:spacy] Modèle spaCy introuvable: {model_dir}, skip.")
        return

    print(f"[core_evaluate:spacy] Chargement modèle depuis {model_dir}")
    nlp = spacy.load(model_dir)

    # On reconstruit le chemin vers les DocBin éventuels produits par core_prepare
    corpus_id = params.get("corpus_id", params.get("corpus", {}).get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")
    spacy_proc_dir = Path("data") / "processed" / corpus_id / view / "spacy"

    # Chercher tous les DocBin "job*.spacy" (shardés ou non)
    job_docbins = sorted(spacy_proc_dir.glob("job*.spacy"))

    if job_docbins:
        print(f"[core_evaluate:spacy] Utilisation de {len(job_docbins)} DocBin job*.spacy dans {spacy_proc_dir}")
        job_docs = []
        for path in job_docbins:
            db = DocBin().from_disk(path)
            job_docs.extend(list(db.get_docs(nlp.vocab)))

        # On extrait textes + labels véritables depuis doc.cats
        texts = [doc.text for doc in job_docs]
        labels_true: List[str] = []
        for doc in job_docs:
            if not doc.cats:
                labels_true.append("__NO_LABEL__")
            else:
                # label avec score max (gold encodé à 1.0)
                best_label = max(doc.cats.items(), key=lambda kv: kv[1])[0]
                labels_true.append(best_label)

        # Sous-échantillon éventuel en debug_mode
        texts, labels_true = maybe_debug_subsample_eval(texts, labels_true, params)

    else:
        # ---- Fallback TSV (flux V2) ----
        print("[core_evaluate:spacy] Aucun DocBin job*.spacy trouvé, fallback sur job.tsv")
        texts, labels_true = load_job_tsv(params)
        texts, labels_true = maybe_debug_subsample_eval(texts, labels_true, params)

    print(f"[core_evaluate:spacy] Évaluation sur {len(texts)} docs.")
    labels_pred: List[str] = []

    # On suppose un TextCat multi-classes exclusif
    for doc in nlp.pipe(texts, disable=[]):
        if not doc.cats:
            labels_pred.append("__NO_PRED__")
            continue
        # label avec score max (prédiction du modèle)
        best_label = max(doc.cats.items(), key=lambda kv: kv[1])[0]
        labels_pred.append(best_label)

    metrics = compute_basic_metrics(labels_true, labels_pred)
    metrics["family"] = "spacy"
    metrics["model_id"] = model_id
    metrics["n_eval_docs"] = len(texts)

    save_eval_outputs(params, "spacy", model_id, metrics)



# ----------------- Éval sklearn -----------------


def eval_sklearn_model(params: Dict[str, Any], model_id: str) -> None:
    model_dir = get_model_output_dir(params, "sklearn", model_id)
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        print(f"[core_evaluate:sklearn] Modèle sklearn introuvable: {model_path}, skip.")
        return

    print(f"[core_evaluate:sklearn] Chargement modèle depuis {model_path}")
    bundle = joblib.load(model_path)
    vectorizer = bundle["vectorizer"]
    estimator = bundle["estimator"]

    texts, labels_true = load_job_tsv(params)
    texts, labels_true = maybe_debug_subsample_eval(texts, labels_true, params)

    print(f"[core_evaluate:sklearn] Évaluation sur {len(texts)} docs.")
    X = vectorizer.transform(texts)
    labels_pred = estimator.predict(X)

    metrics = compute_basic_metrics(labels_true, list(labels_pred))
    metrics["family"] = "sklearn"
    metrics["model_id"] = model_id
    metrics["n_eval_docs"] = len(texts)
    metrics["n_features"] = int(getattr(X, "shape", (0, 0))[1])

    save_eval_outputs(params, "sklearn", model_id, metrics)


# ----------------- Éval HF (squelette) -----------------


def eval_hf_model(params: Dict[str, Any], model_id: str) -> None:
    """
    Évaluation générique HuggingFace (famille 'hf').

    - Charge le modèle et le tokenizer depuis models/{corpus}/{view}/hf/{model_id}
    - Évalue sur job.tsv (ou fallback train.tsv)
    - Utilise compute_basic_metrics pour produire metrics.json + meta_eval.json.
    """
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("[core_evaluate:hf] Transformers ou torch non installés. Skip HF.")
        return

    model_dir = get_model_output_dir(params, "hf", model_id)
    if not model_dir.exists():
        print(f"[core_evaluate:hf] Modèle HF introuvable: {model_dir}, skip.")
        return

    # Charger meta_model pour récupérer le mapping labels
    meta_path = model_dir / "meta_model.json"
    label2id = None
    id2label = None
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        extra = meta.get("extra", {})
        label2id = extra.get("label2id")
        id2label = extra.get("id2label")

    print(f"[core_evaluate:hf] Chargement modèle HF depuis {model_dir}")
    if id2label is not None and label2id is not None and isinstance(id2label, dict):
        # Normaliser les clés (str/int)
        id2label_norm = {int(k): v for k, v in id2label.items()}
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            id2label=id2label_norm,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    texts, labels_true = load_job_tsv(params)
    texts, labels_true = maybe_debug_subsample_eval(texts, labels_true, params)

    if not texts:
        print("[core_evaluate:hf] Aucun document à évaluer.")
        return

    class HFEvalDataset(Dataset):
        def __init__(self, texts: List[str], tokenizer, max_length: int):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            return {k: torch.tensor(v) for k, v in enc.items()}

    # max_length : si présent dans meta_model.extra.trainer_params.max_length
    max_length = 256
    if meta:
        extra = meta.get("extra", {})
        max_length = int(extra.get("trainer_params", {}).get("max_length", max_length))

    dataset = HFEvalDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    preds_ids: List[int] = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            logits = outputs.logits
            batch_pred = torch.argmax(logits, dim=-1).tolist()
            preds_ids.extend(batch_pred)

    # Conversion ids -> labels
    labels_pred: List[str]
    if id2label and isinstance(id2label, dict):
        id2label_int = {int(k): v for k, v in id2label.items()}
        labels_pred = [id2label_int.get(pid, str(pid)) for pid in preds_ids]
    else:
        # Fallback : labels = str(id)
        labels_pred = [str(pid) for pid in preds_ids]

    metrics = compute_basic_metrics(labels_true, labels_pred)
    metrics["family"] = "hf"
    metrics["model_id"] = model_id
    metrics["n_eval_docs"] = len(texts)

    save_eval_outputs(params, "hf", model_id, metrics)



# ----------------- Éval "check" -----------------


def eval_check_model(params: Dict[str, Any], model_id: str = "check_default") -> None:
    """
    Famille 'check' vue comme un pseudo-modèle :
    évaluation = refaire des stats simples sur job.tsv et consigner.
    """
    texts, labels_true = load_job_tsv(params)
    labels_set = sorted(set(labels_true))
    label_counts = {l: labels_true.count(l) for l in labels_set}

    metrics: Dict[str, Any] = {
        "family": "check",
        "model_id": model_id,
        "n_eval_docs": len(texts),
        "labels": labels_set,
        "label_counts": label_counts,
        "note": "Famille 'check' = pseudo-modèle, évaluation = stats brutes sur job.tsv",
    }

    save_eval_outputs(params, "check", model_id, metrics)


# ----------------- main -----------------


# ----------------- main -----------------


def main() -> None:
    args = parse_args()
    params = resolve_profile_base(args.profile, args.override)

    if args.verbose:
        debug_print_params(params)

    # Seed globale optionnelle (comme dans core_train)
    seed_applied = apply_global_seed(params.get("seed"))
    log("evaluate", "seed", f"Global seed: {'appliquée' if seed_applied else 'non appliquée'} ({params.get('seed')})")


    hw = params.get("hardware", {})
    blas_threads = hw.get("blas_threads", 1)
    set_blas_threads(blas_threads)

    families = params.get("families", []) or []
    if args.only_family and args.only_family in families:
        families = [args.only_family]

    models_to_eval: List[Dict[str, Any]] = []

    if "check" in families:
        models_to_eval.append({"family": "check", "model_id": "check_default"})

    if "spacy" in families:
        for mid in params.get("models_spacy", []) or []:
            models_to_eval.append({"family": "spacy", "model_id": mid})

    if "sklearn" in families:
        for mid in params.get("models_sklearn", []) or []:
            models_to_eval.append({"family": "sklearn", "model_id": mid})

    if "hf" in families:
        for mid in params.get("models_hf", []) or []:
            models_to_eval.append({"family": "hf", "model_id": mid})

    if not models_to_eval:
        print(f"[core_evaluate] Aucun modèle à évaluer pour le profil '{params.get('profile')}'. Rien à faire.")
        return

    print("[core_evaluate] Modèles à évaluer :")
    for m in models_to_eval:
        print(f"  - {m['family']}::{m['model_id']}")

    for m in models_to_eval:
        family = m["family"]
        mid = m["model_id"]
        if family == "spacy":
            eval_spacy_model(params, mid)
        elif family == "sklearn":
            eval_sklearn_model(params, mid)
        elif family == "hf":
            eval_hf_model(params, mid)
        elif family == "check":
            eval_check_model(params, mid)
        else:
            print(f"[core_evaluate] WARNING: famille inconnue '{family}', ignorée.")


if __name__ == "__main__":
    main()

