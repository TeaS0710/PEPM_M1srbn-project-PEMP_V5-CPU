#Projet PEPM By Yi Fan && Adrien
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_ideology_skeleton.py (V4 acteurs)

Parcourt un corpus TEI et met à jour `configs/label_maps/ideology_actors.yml`.

Fonctionnalités :
- Ajoute les nouveaux acteurs détectés dans le corpus.
- Met à jour la liste `crawls` pour les acteurs existants sans écraser les
  champs manuels (`side_binary`, `global_five`, `intra_left`, `intra_right`).
- Conserve les acteurs existants absents du corpus (flag optionnel `stale`).
- Génère un rapport TSV des comptes par acteur.
"""

from __future__ import annotations
import argparse
import re
import sys
import unicodedata
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import yaml


def norm_key(s: str) -> str:
    """normalise en snake_case ASCII (accents supprimés)."""
    s = (s or "").lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def extract_actor_tags(tei_el: ET.Element) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Récupère (domain, crawl, fallback) depuis le TEI."""

    domain = None
    crawl = None
    folder = None
    folder_path = None

    for term in tei_el.findall(".//{*}keywords/{*}term"):
        ttype = term.attrib.get("type", "").lower()
        if ttype == "domain" and (term.text or "").strip():
            domain = term.text.strip()
        elif ttype == "crawl" and (term.text or "").strip():
            crawl = term.text.strip()
        elif ttype == "folder" and (term.text or "").strip():
            folder = term.text.strip()
        elif ttype == "folder_path" and (term.text or "").strip():
            folder_path = term.text.strip()

    fallback = folder or folder_path or tei_el.attrib.get("{http://www.w3.org/XML/1998/namespace}id", "").strip()
    return domain, crawl, fallback


def text_len_chars(tei_el: ET.Element) -> int:
    """estime la longueur du texte (head + p) pour filtrer les doc trop courts."""
    total = 0
    head = tei_el.find(".//{*}text/{*}body/{*}div/{*}head")
    if head is not None and head.text:
        total += len(head.text)
    for p in tei_el.findall(".//{*}text/{*}body//{*}p"):
        if p.text:
            total += len(p.text)
    return total


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out-yaml", type=Path, default=Path("configs/label_maps/ideology_actors.yml"))
    ap.add_argument("--out-report", type=Path, default=None)
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument("--top-variants", type=int, default=5)
    return ap.parse_args()


def load_existing_actors(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    if isinstance(data, dict) and isinstance(data.get("actors"), dict):
        actors_raw = data.get("actors") or {}
    else:
        # legacy format {crawl: label}
        actors_raw = {}
        for key in data.keys():
            actors_raw[key] = {}

    actors: Dict[str, Dict[str, Any]] = {}
    for actor_id, info in actors_raw.items():
        base_info = {
            "crawls": [],
            "domains": [],
            "side_binary": None,
            "global_five": None,
            "intra_left": None,
            "intra_right": None,
        }
        if isinstance(info, dict):
            base_info.update(info)
        crawls = base_info.get("crawls") or []
        base_info["crawls"] = sorted({norm_key(c) for c in crawls if c})
        domains = base_info.get("domains") or []
        base_info["domains"] = sorted({norm_key(d) for d in domains if d})
        actors[actor_id] = base_info
    return actors


def merge_actors(existing: Dict[str, Dict[str, Any]], found: Dict[str, Dict[str, Any]]):
    merged: Dict[str, Dict[str, Any]] = {}
    new_actors = 0
    stale_actors = 0

    for actor_id, finfo in found.items():
        base = existing.get(actor_id, {})
        if actor_id not in existing:
            new_actors += 1
        merged_info = {
            "crawls": sorted(set((base.get("crawls") or []) + list(finfo.get("crawls", [])))),
            "domains": sorted(set((base.get("domains") or []) + list(finfo.get("domains", [])))),
            "side_binary": base.get("side_binary"),
            "global_five": base.get("global_five"),
            "intra_left": base.get("intra_left"),
            "intra_right": base.get("intra_right"),
        }
        if "doc_count" in finfo:
            merged_info["doc_count"] = finfo.get("doc_count")
        merged[actor_id] = merged_info

    for actor_id, info in existing.items():
        if actor_id in merged:
            continue
        merged_info = {
            "crawls": info.get("crawls", []),
            "domains": info.get("domains", []),
            "side_binary": info.get("side_binary"),
            "global_five": info.get("global_five"),
            "intra_left": info.get("intra_left"),
            "intra_right": info.get("intra_right"),
            "stale": True,
        }
        stale_actors += 1
        merged[actor_id] = merged_info

    return merged, new_actors, stale_actors


def main():
    args = parse_args()
    if not args.corpus.exists():
        print(f"[ERR] corpus introuvable: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    # Déterminer le rapport TSV par défaut en fonction du corpus
    if args.out_report is None:
        corpus_id = args.corpus.parent.name or "corpus"
        args.out_report = Path(f"data/configs/actors_counts_{corpus_id}.tsv")

    totals: Dict[str, int] = Counter()
    variants: Dict[str, Counter] = defaultdict(Counter)
    domains: Dict[str, set] = defaultdict(set)
    crawls: Dict[str, set] = defaultdict(set)

    ctx = ET.iterparse(args.corpus, events=("end",))
    for event, elem in ctx:
        if elem.tag.endswith("TEI"):
            if args.min_chars > 0 and text_len_chars(elem) < args.min_chars:
                elem.clear()
                continue

            domain_raw, crawl_raw, fallback_raw = extract_actor_tags(elem)
            raw = domain_raw or crawl_raw or fallback_raw or ""
            if not raw:
                elem.clear()
                continue

            key = norm_key(raw)
            if not key:
                elem.clear()
                continue

            if domain_raw:
                domains[key].add(norm_key(domain_raw))
            if crawl_raw:
                crawls[key].add(norm_key(crawl_raw))

            totals[key] += 1
            variants[key][raw] += 1
            elem.clear()

    if not totals:
        print("[WARN] Aucun acteur trouvé (vérifie le XML et --min-chars).")
        return

    found_actors: Dict[str, Dict[str, Any]] = {}
    for key, count in totals.items():
        found_actors[key] = {
            "crawls": crawls.get(key, set()),
            "domains": domains.get(key, set()),
            "doc_count": count,
        }

    existing = load_existing_actors(args.out_yaml)
    merged, new_actors, stale_actors = merge_actors(existing, found_actors)

    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"actors": merged}, f, allow_unicode=True, sort_keys=True)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(found_actors.items(), key=lambda kv: (-kv[1].get("doc_count", 0), kv[0]))
    with args.out_report.open("w", encoding="utf-8") as f:
        f.write("actor_id\tcrawl_ids\tdomain_ids\tdoc_count\tvariant_examples\n")
        for actor_id, info in ordered:
            crawls_list = sorted(info.get("crawls", []))
            domains_list = sorted(info.get("domains", []))
            sample = "; ".join(
                f"{val}×{cnt}" for val, cnt in variants.get(actor_id, {}).most_common(args.top_variants)
            )
            f.write(
                f"{actor_id}\t{';'.join(crawls_list)}\t{';'.join(domains_list)}\t{info.get('doc_count', 0)}\t{sample}\n"
            )

    total_docs = sum(totals.values())
    print(f"[OK] YAML acteurs mis à jour → {args.out_yaml}")
    print(f"[OK] Rapport TSV            → {args.out_report}")
    print(
        f"[INFO] Acteurs uniques: {len(found_actors)} | Total docs: {total_docs} | "
        f"nouveaux: {new_actors} | obsolètes marqués: {stale_actors}"
    )


if __name__ == "__main__":
    main()
