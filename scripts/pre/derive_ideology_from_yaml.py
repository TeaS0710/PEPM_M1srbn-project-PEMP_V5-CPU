#!/usr/bin/env python
"""
derive_ideology_from_yaml.py

À partir d'un fichier d'annotation simple (crawl_id -> label brut),
et d'un squelette d'acteurs (ideology_actors.yml),
dérive automatiquement :

- un fichier acteurs enrichi (side_binary, global_five, intra_left / intra_right),
- une vue globale domaine/crawl -> label 5-classes (ideology_global.yml),
- des vues intra-gauche / intra-droite (ideology_left_intra.yml / ideology_right_intra.yml).
"""

import argparse
import collections
from typing import Dict, Any, Tuple, Optional, List

import yaml


def normalize_key(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    for ch in [" ", "\t", "\n"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


# À adapter si tu as d'autres variantes
LEFT_RAW = {
    "gauche",
    "gauche_moderee",
    "gauche_moderée",
    "left",
    "centre_gauche",
    "centre-left",
}

FAR_LEFT_RAW = {
    "exgauche",
    "extreme_gauche",
    "extrême_gauche",
    "far_left",
}

RIGHT_RAW = {
    "droite",
    "droite_moderee",
    "droite_moderée",
    "right",
    "centre_droite",
    "centre-right",
}

FAR_RIGHT_RAW = {
    "exdroite",
    "extreme_droite",
    "extrême_droite",
    "far_right",
}


def classify_raw_label(raw: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    À partir d'un label brut (gauche, exgauche, droite, ...),
    retourne :
      - side_binary : "left" / "right" / None
      - global_five : "far_left" / "left" / "center" / "right" / "far_right" / None
      - intra_left  : étiquette intra si camp left, sinon None
      - intra_right : étiquette intra si camp right, sinon None
    """
    norm = normalize_key(raw)

    if norm in FAR_LEFT_RAW:
        return "left", "far_left", "exgauche", None
    if norm in LEFT_RAW:
        return "left", "left", "gauche", None

    if norm in FAR_RIGHT_RAW:
        return "right", "far_right", None, "exdroite"
    if norm in RIGHT_RAW:
        return "right", "right", None, "droite"

    if norm in {"centre", "center"}:
        return None, "center", None, None

    return None, None, None, None


def aggregate_labels(labels: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Agrège les labels bruts sur un acteur (domain) : majority vote.
    Si conflit binaire (left+right), on renvoie tout à None.
    """
    if not labels:
        return None, None, None, None

    side_counts = collections.Counter()
    five_counts = collections.Counter()
    left_intra_counts = collections.Counter()
    right_intra_counts = collections.Counter()

    for raw in labels:
        side, five, left_intra, right_intra = classify_raw_label(raw)
        if side:
            side_counts[side] += 1
        if five:
            five_counts[five] += 1
        if left_intra:
            left_intra_counts[left_intra] += 1
        if right_intra:
            right_intra_counts[right_intra] += 1

    # Conflit gauche/droite
    if len(side_counts) > 1:
        return None, None, None, None

    side_binary = side_counts.most_common(1)[0][0] if side_counts else None
    global_five = five_counts.most_common(1)[0][0] if five_counts else None
    intra_left = left_intra_counts.most_common(1)[0][0] if left_intra_counts else None
    intra_right = right_intra_counts.most_common(1)[0][0] if right_intra_counts else None

    return side_binary, global_five, intra_left, intra_right


def main() -> None:
    ap = argparse.ArgumentParser(description="Dériver les vues idéologiques à partir d'un YAML crawl->label brut.")
    ap.add_argument("--ideology-yaml", required=True, help="YAML de base (crawl_id -> label brut)")
    ap.add_argument("--actors-yaml", required=True, help="YAML acteurs (squelette) à enrichir")
    ap.add_argument("--out-actors", required=True, help="Chemin de sortie pour le YAML acteurs enrichi")
    ap.add_argument("--out-global", required=True, help="Chemin de sortie pour ideology_global.yml")
    ap.add_argument("--out-left-intra", required=True, help="Chemin de sortie pour ideology_left_intra.yml")
    ap.add_argument("--out-right-intra", required=True, help="Chemin de sortie pour ideology_right_intra.yml")
    args = ap.parse_args()

    # 1) YAML de base (crawl -> label brut)
    with open(args.ideology_yaml, "r", encoding="utf-8") as f:
        raw_ideo = yaml.safe_load(f) or {}
    if not isinstance(raw_ideo, dict):
        raise SystemExit(f"[ERR] {args.ideology_yaml} doit être un mapping crawl_id -> label.")

    crawl_label: Dict[str, str] = {}
    for k, v in raw_ideo.items():
        if v is None:
            continue
        v_str = str(v).strip()
        if not v_str:
            continue
        crawl_label[normalize_key(str(k))] = v_str

    # 2) YAML acteurs (squelette)
    with open(args.actors_yaml, "r", encoding="utf-8") as f:
        actors_yaml = yaml.safe_load(f) or {}
    actors_tbl = actors_yaml.get("actors") or {}
    if not isinstance(actors_tbl, dict):
        raise SystemExit(f"[ERR] {args.actors_yaml} doit contenir une clé 'actors'.")

    global_mapping: Dict[str, str] = {}
    left_intra_mapping: Dict[str, str] = {}
    right_intra_mapping: Dict[str, str] = {}

    for actor_id, info in actors_tbl.items():
        info = info or {}
        crawls = info.get("crawls") or []
        domains = info.get("domains") or []

        labels_for_actor: List[str] = []
        for c in crawls:
            c_norm = normalize_key(str(c))
            if c_norm in crawl_label:
                labels_for_actor.append(crawl_label[c_norm])

        side_binary, global_five, intra_left, intra_right = aggregate_labels(labels_for_actor)

        info["side_binary"] = side_binary
        info["global_five"] = global_five
        info["intra_left"] = intra_left
        info["intra_right"] = intra_right
        actors_tbl[actor_id] = info

        keys = list(crawls) + list(domains)

        if global_five:
            for k in keys:
                global_mapping[str(k)] = global_five
        if intra_left:
            for k in keys:
                left_intra_mapping[str(k)] = intra_left
        if intra_right:
            for k in keys:
                right_intra_mapping[str(k)] = intra_right

    actors_yaml["actors"] = actors_tbl

    with open(args.out_actors, "w", encoding="utf-8") as f:
        yaml.safe_dump(actors_yaml, f, sort_keys=True, allow_unicode=True)

    def dump_mapping(path: str, mapping: Dict[str, str]) -> None:
        payload = {
            "mapping": dict(sorted(mapping.items())),
            "unknown_labels": {"policy": "drop", "other_label": "other"},
        }
        with open(path, "w", encoding="utf-8") as out:
            yaml.safe_dump(payload, out, sort_keys=True, allow_unicode=True)

    dump_mapping(args.out_global, global_mapping)
    dump_mapping(args.out_left_intra, left_intra_mapping)
    dump_mapping(args.out_right_intra, right_intra_mapping)


if __name__ == "__main__":
    main()
