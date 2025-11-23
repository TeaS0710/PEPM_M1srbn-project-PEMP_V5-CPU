# scripts/pre/pre_check_config.py

import argparse
import os
from typing import Any, Dict

from scripts.core.core_utils import (
    resolve_profile_base,
    load_label_map,
    debug_print_params,
)


def validate_models(params: Dict[str, Any]) -> None:
    """
    Vérifier que les modèles référencés dans le profil existent dans models.yml
    pour chaque famille (spacy/sklearn/hf).
    """
    models_cfg_families = params.get("models_cfg", {}).get("families", {}) or {}
    families = params.get("families", []) or []

    def check_model_list(family_key: str, list_key: str) -> None:
        model_ids = params.get(list_key, []) or []
        family_cfg = models_cfg_families.get(family_key, {}) or {}
        for mid in model_ids:
            if mid not in family_cfg:
                raise SystemExit(
                    f"[config] Modèle '{mid}' non trouvé dans models.yml "
                    f"(famille '{family_key}', clé '{list_key}')"
                )

    if "spacy" in families:
        check_model_list("spacy", "models_spacy")
    if "sklearn" in families:
        check_model_list("sklearn", "models_sklearn")
    if "hf" in families:
        check_model_list("hf", "models_hf")
    if "check" in families:
        # pseudo-modèle interne, rien à vérifier ici
        pass


def validate_label_map(params: Dict[str, Any]) -> None:
    path = params.get("label_map")
    if not path:
        print("[config] WARNING: label_map non défini dans le profil")
        return
    if not os.path.exists(path):
        raise SystemExit(f"[config] label_map introuvable: {path}")
    mapping = load_label_map(path)
    if not mapping:
        print(f"[config] WARNING: label_map '{path}' chargé mais mapping vide (valeurs vides ?)")


def validate_corpus_and_labels(params: Dict[str, Any]) -> None:
    """Valider que le corpus et les label_maps référencés existent et sont chargeables."""
    corpus = params.get("corpus", {}) or {}
    corpus_path = corpus.get("corpus_path")
    if not corpus_path:
        raise SystemExit("[config] corpus.corpus_path manquant dans les paramètres résolus.")

    if not os.path.exists(corpus_path):
        print(f"[config] WARNING: corpus_path introuvable sur disque : {corpus_path}")

    label_map_path = params.get("label_map")
    if label_map_path:
        if not os.path.exists(label_map_path):
            raise SystemExit(f"[config] label_map inexistant : {label_map_path}")
        try:
            load_label_map(label_map_path)
        except Exception as e:
            raise SystemExit(f"[config] Impossible de charger label_map={label_map_path} : {e}")

    families = params.get("families") or []
    known_families = {"spacy", "sklearn", "hf", "check"}
    for fam in families:
        if fam not in known_families:
            raise SystemExit(
                f"[config] Famille inconnue dans profil : {fam!r} "
                f"(attendu dans {sorted(known_families)})"
            )


def validate_spacy_templates(params: Dict[str, Any]) -> None:
    """
    Vérifier que les templates spaCy référencés dans models.yml existent bien sur disque.
    """
    families_cfg = params.get("models_cfg", {}).get("families", {}) or {}
    spacy_models = families_cfg.get("spacy", {}) or {}

    missing = []
    for mid, mc in spacy_models.items():
        tpl = mc.get("config_template")
        if not tpl:
            continue
        from pathlib import Path

        p = Path(tpl)
        if not p.exists():
            missing.append((mid, tpl))

    if missing:
        for mid, tpl in missing:
            print(f"[pre_check] MISSING spaCy config_template for model '{mid}': {tpl}")
        raise SystemExit("[pre_check] Missing spaCy config templates.")


def validate_families_and_hardware(params: Dict[str, Any]) -> None:
    """
    Vérifier que:
      - les familles demandées existent dans models.yml (sauf 'check' qui est pseudo-famille),
      - le hardware_preset est connu,
      - le hardware résolu est raisonnable.
    """
    models_cfg_all = params.get("models_cfg", {}) or {}
    families_cfg = models_cfg_all.get("families", {}) or {}

    families_req = set(params.get("families", []) or [])

    # 'check' est une pseudo-famille interne : on ne l'exige pas dans models.yml
    families_req_for_models = {f for f in families_req if f != "check"}

    known_fams = set(families_cfg.keys())
    unknown = families_req_for_models - known_fams
    if unknown:
        raise SystemExit(f"[pre_check] Unknown families requested in profile: {sorted(unknown)}")

    hardware_cfg = params.get("hardware_cfg", {}) or {}
    hp = params.get("hardware_preset", "small")
    presets = hardware_cfg.get("presets", {}) or {}
    if hp not in presets:
        raise SystemExit(f"[pre_check] Unknown hardware_preset: {hp}")

    hw = params.get("hardware", {}) or {}
    if not hw:
        print("[config] WARNING: pas de hardware_preset appliqué (hardware vide)")
    else:
        if hw.get("ram_gb", 0) <= 0:
            print("[config] WARNING: ram_gb non réaliste")
        if hw.get("max_procs", 0) <= 0:
            print("[config] WARNING: max_procs non réaliste")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pré-check d'un profil V4 (cohérence configs)"
    )
    ap.add_argument("--profile", required=True, help="Nom du profil (sans .yml)")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config (clé=valeur, ex: train_prop=0.7)",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Afficher les params résolus"
    )
    args = ap.parse_args()

    params = resolve_profile_base(args.profile, args.override)

    # Validations
    validate_label_map(params)
    validate_corpus_and_labels(params)
    validate_models(params)
    validate_spacy_templates(params)
    validate_families_and_hardware(params)

    if args.verbose:
        debug_print_params(params)

    print(f"[OK] Profil '{args.profile}' validé.")


if __name__ == "__main__":
    main()
