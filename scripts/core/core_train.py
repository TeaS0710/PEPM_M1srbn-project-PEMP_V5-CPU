# scripts/core/core_train.py

import argparse
import csv
import json
import os
import random
import sys
import importlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib

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
        description="V4 core_train : entraînement multi-familles (spaCy, sklearn, HF, check)"
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
        help="Override config (clé=valeur, ex: hardware_preset=lab)",
    )
    ap.add_argument(
        "--only-family",
        choices=["spacy", "sklearn", "hf", "check"],
        help="Limiter l'entraînement à une seule famille (optionnel)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les paramètres résolus",
    )
    return ap.parse_args()


# ----------------- Utils généraux -----------------

def compute_class_weights_from_counts(label_counts: Counter) -> Dict[str, float]:
    """
    Même formule que dans core_prepare : w(label) = n_samples / (n_labels * n_label).

    Utilisé ici pour remplir class_weight des modèles sklearn quand la
    stratégie d'équilibrage est 'class_weights' et que le config demande
    explicitement class_weight: "from_balance".
    """
    total = sum(label_counts.values())
    n_labels = len(label_counts) or 1
    if total <= 0:
        return {lab: 1.0 for lab in label_counts}

    weights: Dict[str, float] = {}
    for lab, c in label_counts.items():
        if c <= 0:
            weights[lab] = 0.0
        else:
            weights[lab] = total / (n_labels * c)
    return weights



def set_blas_threads(n_threads: int) -> None:
    """
    Limiter les threads BLAS (MKL/OPENBLAS/OMP) pour éviter la sur-souscription.
    """
    if n_threads is None or n_threads <= 0:
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    print(f"[core_train] BLAS threads fixés à {n_threads}")


def import_string(path: str):
    """
    Import dynamique d'une classe ou fonction à partir d'une chaîne:
    ex: 'sklearn.svm.LinearSVC' -> class.
    """
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_model_output_dir(params: Dict[str, Any], family: str, model_id: str) -> Path:
    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")
    return Path("models") / corpus_id / view / family / model_id


def load_tsv_dataset(params: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    Charger train.tsv et job.tsv depuis data/interim/{corpus_id}/{view}/
    Retourne (train_texts, train_labels, job_texts).
    job_labels ne sont pas strictement nécessaires pour l'entraînement (évent. early stopping),
    on peut les charger plus tard lors de l'évaluation.
    """
    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")
    interim_dir = Path("data") / "interim" / corpus_id / view
    train_path = interim_dir / "train.tsv"
    job_path = interim_dir / "job.tsv"

    if not train_path.exists():
        raise SystemExit(
            "[core_train] train.tsv introuvable: {p}\n"
            "  -> Aucune donnée d'entraînement trouvée pour "
            "corpus_id={cid}, view={view}.\n"
            "  -> Vérifie que core_prepare a bien tourné et qu'il n'a pas "
            "filtré tous les documents (min_chars, label_map, etc.).".format(
                p=train_path, cid=corpus_id, view=view
            )
        )


    def read_tsv(path: Path) -> Tuple[List[str], List[str]]:
        texts: List[str] = []
        labels: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("text") or ""
                label = row.get("label")
                if not text or not label:
                    continue
                texts.append(text)
                labels.append(label)
        return texts, labels

    train_texts, train_labels = read_tsv(train_path)

    if job_path.exists():
        job_texts, _job_labels = read_tsv(job_path)
    else:
        print(f"[core_train] WARNING: job.tsv introuvable, on utilisera train comme job pour certains usages.")
        job_texts = train_texts

    return train_texts, train_labels, job_texts


def maybe_debug_subsample(
    texts: List[str],
    labels: List[str],
    params: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Si debug_mode=True, limiter la taille du dataset (ex: 1000 docs max),
    en utilisant le seed du profil pour une reproductibilité globale.
    """
    if not params.get("debug_mode"):
        return texts, labels

    max_docs = 1000
    if len(texts) <= max_docs:
        return texts, labels

    seed = int(params.get("seed", 42))
    print(f"[core_train] debug_mode actif : sous-échantillon de {max_docs} docs sur {len(texts)} (seed={seed})")
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    idx_sel = sorted(indices[:max_docs])
    texts_sub = [texts[i] for i in idx_sel]
    labels_sub = [labels[i] for i in idx_sel]
    return texts_sub, labels_sub



def save_meta_model(
    params: Dict[str, Any],
    family: str,
    model_id: str,
    model_dir: Path,
    extra: Dict[str, Any],
) -> None:
    meta = {
        "profile": params.get("profile"),
        "description": params.get("description", ""),
        "corpus_id": params.get("corpus_id", params["corpus"].get("corpus_id")),
        "view": params.get("view"),
        "family": family,
        "model_id": model_id,
        "hardware": params.get("hardware", {}),
        "debug_mode": params.get("debug_mode", False),
        "pipeline_version": PIPELINE_VERSION,
    }
    meta.update(extra)
    meta_path = model_dir / "meta_model.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[core_train] meta_model.json écrit : {meta_path}")


# ----------------- Entraînement spaCy -----------------

def train_spacy_model(params: Dict[str, Any], model_id: str) -> None:
    """
    Entraînement spaCy générique (config-first) pour la famille 'spacy'.

    - Charge un template .cfg depuis models.yml (config_template)
    - Récupère les DocBin construits par core_prepare (possiblement shardés)
    - Merge les shards en un seul DocBin temporaire si besoin,
      en respectant éventuellement hardware.max_train_docs_spacy
    - Override hyperparams (epochs, dropout) depuis models.yml
    """
    import json, tempfile
    from pathlib import Path
    from scripts.core.core_utils import get_model_output_dir, ensure_dir, save_meta_model, log

    try:
        import spacy
        from spacy.tokens import DocBin
        from spacy.util import load_config
        from spacy.cli.train import train as spacy_train
    except ImportError:
        log("train", "spacy", "spaCy non installé. Skip spaCy.")
        return

    spacy_models = params["models_cfg"]["families"]["spacy"]
    model_cfg = spacy_models.get(model_id)
    if not model_cfg:
        log("train", "spacy", f"Modèle spaCy '{model_id}' introuvable dans models.yml. Skip.")
        return

    config_template = model_cfg.get("config_template")
    if not config_template:
        log("train", "spacy", f"Pas de 'config_template' pour {model_id}. Skip.")
        return

    # Langue pour le merge DocBin
    lang = model_cfg.get("lang", "xx")

    # Localiser processed_dir / spacy_dir (utile pour fallback)
    corpus_id = params.get("corpus_id", params.get("corpus", {}).get("corpus_id"))
    view = params.get("view", params.get("profile_raw", {}).get("view"))
    processed_dir = Path("data/processed") / str(corpus_id) / str(view)
    spacy_dir = processed_dir / "spacy"

    meta_formats_path = processed_dir / "meta_formats.json"
    train_paths: List[str] = []
    dev_paths: List[str] = []

    def _normalize_paths(x) -> List[str]:
        if not x:
            return []
        if isinstance(x, str):
            return [x]
        if isinstance(x, list):
            return [str(p) for p in x]
        return []

    def resolve_paths_from_meta() -> bool:
        if not meta_formats_path.exists():
            return False
        try:
            fm = json.loads(meta_formats_path.read_text(encoding="utf-8"))
            spacy_meta = fm.get("families", {}).get("spacy", {})
            tr = _normalize_paths(spacy_meta.get("train_spacy"))
            dv = _normalize_paths(spacy_meta.get("job_spacy"))
            train_paths.extend(tr)
            dev_paths.extend(dv)
            return bool(train_paths and dev_paths)
        except Exception as e:
            log("train", "spacy", f"Erreur lecture meta_formats.json: {e}")
            return False

    ok = resolve_paths_from_meta()
    if not ok:
        # Fallback simple: fichiers train.spacy / job.spacy dans le dossier spacy/
        ts = spacy_dir / "train.spacy"
        js = spacy_dir / "job.spacy"
        if ts.exists():
            train_paths.append(str(ts))
        if js.exists():
            dev_paths.append(str(js))

    if not (train_paths and dev_paths):
        raise SystemExit("[core_train:spacy] DocBin train/dev introuvables.")

    # Limite matérielle éventuelle sur le nombre de docs spaCy
    hw = params.get("hardware", {}) or {}
    max_docs_spacy = int(hw.get("max_train_docs_spacy", 0) or 0)

    def merge_docbins(paths: List[str], out_path: Path, max_docs: int = 0) -> int:
        """
        Merge une liste de DocBin en un seul.
        max_docs > 0 -> on arrête après max_docs docs (pour limiter RAM / temps).
        Retourne le nombre de docs écrits.
        """
        nlp = spacy.blank(lang)
        db_out = DocBin()
        total = 0
        for p in paths:
            db_in = DocBin().from_disk(p)
            for doc in db_in.get_docs(nlp.vocab):
                db_out.add(doc)
                total += 1
                if max_docs and total >= max_docs:
                    break
            if max_docs and total >= max_docs:
                break
        db_out.to_disk(out_path)
        return total

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        train_merged = tmp_dir / "train_merged.spacy"
        dev_merged = tmp_dir / "dev_merged.spacy"

        if len(train_paths) > 1 or max_docs_spacy:
            n_train_eff = merge_docbins(train_paths, train_merged, max_docs=max_docs_spacy)
            train_bin = str(train_merged)
        else:
            train_bin = train_paths[0]
            n_train_eff = None  # inconnu sans parsing

        if len(dev_paths) > 1:
            # Pour dev, on ne limite pas forcément le nb de docs (tu peux mettre max_docs_spacy si tu veux)
            _ = merge_docbins(dev_paths, dev_merged, max_docs=0)
            dev_bin = str(dev_merged)
        else:
            dev_bin = dev_paths[0]

        # Charger le template et override les chemins + hyperparams
        cfg = load_config(config_template)
        if "paths" not in cfg:
            cfg["paths"] = {}
        cfg["paths"]["train"] = train_bin
        cfg["paths"]["dev"] = dev_bin

        if "training" not in cfg:
            cfg["training"] = {}
        epochs = model_cfg.get("epochs")
        if epochs is not None:
            cfg["training"]["max_epochs"] = int(epochs)
        dropout = model_cfg.get("dropout")
        if dropout is not None:
            cfg["training"]["dropout"] = float(dropout)

        # Répertoire de sortie
        model_dir = get_model_output_dir(params, "spacy", model_id)
        ensure_dir(model_dir)

        # Seed spécifique spaCy
        seed_val = params.get("seed")
        try:
            import spacy.util as spacy_util
            if seed_val is not None and str(seed_val).lower() not in {"none", "null", ""} and int(seed_val) >= 0:
                spacy_util.fix_random_seed(int(seed_val))
                log("train", "spacy", f"Seed spaCy={int(seed_val)}")
        except Exception:
            pass

        log(
            "train",
            "spacy",
            f"Train {model_id} | epochs={cfg['training'].get('max_epochs')} "
            f"| template={config_template} | max_docs_spacy={max_docs_spacy or '∞'}",
        )

        # Entraînement via API spaCy (équivalent CLI)
        spacy_train(cfg, output_path=model_dir, overrides={})

        extra = {
            "arch": model_cfg.get("arch"),
            "config_template": config_template,
            "n_train_docs_effective": n_train_eff,
            "lang": lang,
            "max_docs_spacy": max_docs_spacy or None,
        }
        save_meta_model(params, "spacy", model_id, model_dir, extra=extra)




# ----------------- Entraînement sklearn -----------------

def train_sklearn_model(params: Dict[str, Any], model_id: str) -> None:
    models_cfg = params["models_cfg"]["families"]["sklearn"][model_id]
    vect_cfg = models_cfg["vectorizer"]
    est_cfg = models_cfg["estimator"]

    vect_class = import_string(vect_cfg["class"])
    est_class = import_string(est_cfg["class"])

    vect_params = dict(vect_cfg.get("params", {}))
    est_params = dict(est_cfg.get("params", {}))

    # Permettre random_state=from_seed dans models.yml
    rs = est_params.get("random_state")
    if isinstance(rs, str) and rs == "from_seed":
        try:
            est_params["random_state"] = int(params.get("seed"))
        except Exception:
            est_params.pop("random_state", None)


    # Charger les données depuis train.tsv / job.tsv
    train_texts, train_labels, job_texts = load_tsv_dataset(params)

    # Sous-échantillonnage éventuel en mode debug
    train_texts, train_labels = maybe_debug_subsample(train_texts, train_labels, params)

    # Limite hardware éventuelle sur le nombre de docs
    hw = params.get("hardware", {}) or {}
    max_docs = int(hw.get("max_train_docs_sklearn") or 0)
    n_raw = len(train_texts)
    if max_docs > 0 and n_raw > max_docs:
        print(f"[core_train:sklearn] max_train_docs_sklearn={max_docs}, tronque {n_raw} -> {max_docs} docs")
        train_texts = train_texts[:max_docs]
        train_labels = train_labels[:max_docs]

    # Poids de classe éventuels (stratégie 'class_weights')
    if params.get("balance_strategy") == "class_weights":
        label_counts = Counter(train_labels)
        class_weights = compute_class_weights_from_counts(label_counts)
        if est_params.get("class_weight") == "from_balance":
            est_params = dict(est_params)
            est_params["class_weight"] = class_weights

    # Ajuster n_jobs si possible en fonction du preset hardware
    max_procs = hw.get("max_procs")
    if max_procs and "n_jobs" in est_params and est_params["n_jobs"] in (None, -1):
        est_params["n_jobs"] = max_procs

    vectorizer = vect_class(**vect_params)
    estimator = est_class(**est_params)

    print(
        f"[core_train:sklearn] Modèle={model_id}, "
        f"{len(train_texts)} docs d'entraînement (raw={n_raw})."
    )

    X_train = vectorizer.fit_transform(train_texts)
    estimator.fit(X_train, train_labels)

    model_dir = get_model_output_dir(params, "sklearn", model_id)
    ensure_dir(model_dir)
    model_path = model_dir / "model.joblib"
    joblib.dump({"vectorizer": vectorizer, "estimator": estimator}, model_path)
    print(f"[core_train:sklearn] Modèle sklearn sauvegardé dans {model_path}")

    save_meta_model(
        params,
        "sklearn",
        model_id,
        model_dir,
        extra={
            "vectorizer_class": vect_cfg["class"],
            "estimator_class": est_cfg["class"],
            "vectorizer_params": vect_params,
            "estimator_params": est_params,
            "n_train_docs_raw": int(n_raw),
            "n_train_docs_effective": int(len(train_texts)),
            "max_train_docs_sklearn": int(max_docs),
            "n_features": int(getattr(X_train, "shape", (0, 0))[1]),
            "balance_strategy": params.get("balance_strategy", "none"),
        },
    )




# ----------------- Entraînement HF (squelette) -----------------


def train_hf_model(params: Dict[str, Any], model_id: str) -> None:
    """
    Entraînement générique HuggingFace (famille 'hf') en mode config-first.

    - Lit les hyperparamètres dans params["models_cfg"]["families"]["hf"][model_id]
    - Lit les données via load_tsv_dataset(params)
    - Ne dépend d'aucune logique spécifique au modèle :
      ajout de modèles via models.yml uniquement.
    """
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import TrainingArguments, Trainer
    except ImportError:
        print("[core_train:hf] Transformers ou torch non installés. Skip HF.")
        return

    models_cfg = params["models_cfg"]["families"]["hf"][model_id]

    model_name = models_cfg.get("model_name")
    if not model_name:
        raise SystemExit(f"[core_train:hf] 'model_name' manquant pour le modèle HF '{model_id}' dans models.yml")

    tokenizer_class_path = models_cfg.get("tokenizer_class", "transformers.AutoTokenizer")
    model_class_path = models_cfg.get("model_class", "transformers.AutoModelForSequenceClassification")
    trainer_params = models_cfg.get("trainer_params", {}) or {}
    use_class_weights = bool(models_cfg.get("use_class_weights", models_cfg.get("trainer_params", {}).get("use_class_weights", False)))

    # ---------- Données ----------
    train_texts, train_labels_str, _job_texts = load_tsv_dataset(params)
    train_texts, train_labels_str = maybe_debug_subsample(train_texts, train_labels_str, params)

    if not train_texts:
        raise SystemExit("[core_train:hf] Dataset d'entraînement vide.")

    # Limite hardware éventuelle
    hw = params.get("hardware", {}) or {}
    max_docs = int(hw.get("max_train_docs_hf") or 0)
    n_raw = len(train_texts)
    if max_docs > 0 and n_raw > max_docs:
        print(f"[core_train:hf] max_train_docs_hf={max_docs}, tronque {n_raw} -> {max_docs} docs")
        train_texts = train_texts[:max_docs]
        train_labels_str = train_labels_str[:max_docs]

    # Mapping label -> id (stable, loggable)
    unique_labels = sorted(set(train_labels_str))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    train_labels = [label2id[lab] for lab in train_labels_str]

    # ---------- Import dynamique des classes HF ----------
    import importlib

    def import_class(path: str):
        mod_name, cls_name = path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)

    TokCls = import_class(tokenizer_class_path)
    ModelCls = import_class(model_class_path)

    tokenizer = TokCls.from_pretrained(model_name)
    num_labels = len(unique_labels)
    model = ModelCls.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ---------- Poids de classe éventuels ----------
    class_weights_tensor = None
    label_weights = None
    if use_class_weights and params.get("balance_strategy") == "class_weights":
        # Idéalement fournis par core_prepare via params["class_weights"]
        label_weights = params.get("class_weights")
        if not label_weights:
            from collections import Counter
            label_counts = Counter(train_labels_str)
            label_weights = compute_class_weights_from_counts(label_counts)
        # Ordonner les poids selon unique_labels -> vecteur pour CrossEntropyLoss
        weights_list = [float(label_weights.get(lab, 1.0)) for lab in unique_labels]
        class_weights_tensor = torch.tensor(weights_list, dtype=torch.float)

    max_length = models_cfg.get("max_length") or trainer_params.get("max_length", 256)

    class HFDataset(Dataset):
        def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
            self.texts = texts
            self.labels = labels
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
            enc = {k: torch.tensor(v) for k, v in enc.items()}
            enc["labels"] = torch.tensor(self.labels[idx])
            return enc

    train_dataset = HFDataset(train_texts, train_labels, tokenizer, max_length)

    # ---------- Répertoires de sortie ----------
    model_dir = get_model_output_dir(params, "hf", model_id)
    ensure_dir(model_dir)
    output_dir = model_dir / "hf_outputs"
    ensure_dir(output_dir)

    # ---------- Hyperparams / hardware (config-first) ----------
    train_batch_size = int(trainer_params.get("per_device_train_batch_size", 8))
    eval_batch_size = int(trainer_params.get("per_device_eval_batch_size", train_batch_size))
    num_train_epochs = float(trainer_params.get("num_train_epochs", 3.0))
    learning_rate = float(trainer_params.get("learning_rate", 2e-5))
    weight_decay = float(trainer_params.get("weight_decay", 0.0))
    warmup_ratio = float(trainer_params.get("warmup_ratio", 0.0))
    grad_accum = int(trainer_params.get("gradient_accumulation_steps", 1))

    # -- seed optionnelle (None/"none"/"" ou <0 => pas de seed) --
    seed_val = params.get("seed")
    seed_int = None
    if seed_val is not None:
        try:
            if isinstance(seed_val, str) and seed_val.strip().lower() in {"none", "null", ""}:
                seed_int = None
            else:
                seed_int = int(seed_val)
                if seed_int < 0:
                    seed_int = None
        except Exception:
            seed_int = None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=grad_accum,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=False,
        **({"seed": seed_int, "data_seed": seed_int} if seed_int is not None else {}),
    )






    # Trainer pondéré optionnel
    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    if class_weights_tensor is not None:
        TrainerCls = WeightedTrainer
        trainer_kwargs["class_weights"] = class_weights_tensor
    else:
        TrainerCls = Trainer

    print(f"[core_train:hf] Entraînement HF pour '{model_id}' avec {len(train_texts)} docs (raw={n_raw}).")
    trainer = TrainerCls(**trainer_kwargs)
    trainer.train()

    # Sauvegarde du modèle final + tokenizer dans model_dir
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    extra = {
        "hf_model_name": model_name,
        "label2id": label2id,
        "id2label": id2label,
        "trainer_params": trainer_params,
        "use_class_weights": use_class_weights,
        "class_weights": label_weights,
        "n_train_docs_raw": int(n_raw),
        "n_train_docs_effective": int(len(train_texts)),
        "max_train_docs_hf": int(max_docs),
    }
    save_meta_model(params, "hf", model_id, model_dir, extra=extra)




# ----------------- Entraînement "check" -----------------


def train_check_model(params: Dict[str, Any], model_id: str = "check_default") -> None:
    """
    Famille 'check' vue comme un "pseudo-modèle" :
    il peut générer des stats, des sanity checks, etc., et écrire un meta_model.json.
    Pour l'instant on se contente de consigner les stats de base.
    """
    train_texts, train_labels, job_texts = load_tsv_dataset(params)
    labels_set = sorted(set(train_labels))
    label_counts = {l: train_labels.count(l) for l in labels_set}

    model_dir = get_model_output_dir(params, "check", model_id)
    ensure_dir(model_dir)

    save_meta_model(
        params,
        "check",
        model_id,
        model_dir,
        extra={
            "n_train_docs": len(train_texts),
            "n_labels": len(labels_set),
            "label_counts": label_counts,
            "note": "Famille 'check' = modèle virtuel pour sanity checks / stats",
        },
    )
    print(f"[core_train:check] Checks de base consignés dans {model_dir}")


# ----------------- main -----------------


def main() -> None:
    args = parse_args()
    params = resolve_profile_base(args.profile, args.override)

    if args.verbose:
        debug_print_params(params)

    # Seed de base pour une reproductibilité minimaliste
    seed_applied = apply_global_seed(params.get("seed"))
    log("train", "seed", f"Global seed: {'appliquée' if seed_applied else 'non appliquée'} ({params.get('seed')})")


    hw = params.get("hardware", {})
    blas_threads = hw.get("blas_threads", 1)
    set_blas_threads(blas_threads)

    families = params.get("families", []) or []
    if args.only_family and args.only_family in families:
        families = [args.only_family]

    # Construire la liste des modèles à entraîner
    models_to_train: List[Dict[str, Any]] = []

    if "check" in families:
        # Pour l'instant un seul pseudo-modèle check_default
        models_to_train.append({"family": "check", "model_id": "check_default"})

    if "spacy" in families:
        for mid in params.get("models_spacy", []) or []:
            models_to_train.append({"family": "spacy", "model_id": mid})

    if "sklearn" in families:
        for mid in params.get("models_sklearn", []) or []:
            models_to_train.append({"family": "sklearn", "model_id": mid})

    if "hf" in families:
        for mid in params.get("models_hf", []) or []:
            models_to_train.append({"family": "hf", "model_id": mid})

    if not models_to_train:
        print(f"[core_train] Aucun modèle à entraîner pour le profil '{params.get('profile')}'. Rien à faire.")
        return

    print("[core_train] Modèles à entraîner :")
    for m in models_to_train:
        print(f"  - {m['family']}::{m['model_id']}")

    # Entraînement
    for m in models_to_train:
        family = m["family"]
        mid = m["model_id"]
        if family == "spacy":
            train_spacy_model(params, mid)
        elif family == "sklearn":
            train_sklearn_model(params, mid)
        elif family == "hf":
            train_hf_model(params, mid)
        elif family == "check":
            train_check_model(params, mid)
        else:
            print(f"[core_train] WARNING: famille inconnue '{family}', ignorée.")


if __name__ == "__main__":
    main()
