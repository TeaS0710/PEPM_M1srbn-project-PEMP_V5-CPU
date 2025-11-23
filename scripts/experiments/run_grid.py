#!/usr/bin/env python
"""
scripts/experiments/run_grid.py

Script minimal pour lancer des sweeps combinant profils / familles / overrides.

Usage typique :
    python scripts/experiments/run_grid.py \
        --profiles ideo_quick crawl_full \
        --families spacy sklearn \
        --max-runs 10 \
        --dry-run

Cela génère des commandes 'make pipeline PROFILE=... FAMILY=... OVERRIDES=...'.
"""

import argparse
import itertools
import shlex
import subprocess
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="Liste des profils à tester (ex: ideo_quick crawl_full)",
    )
    p.add_argument(
        "--families",
        nargs="+",
        default=["spacy", "sklearn", "hf", "check"],
        help="Familles de modèles à inclure (spacy/sklearn/hf/check)",
    )
    p.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Liste d'overrides globaux (clé=val) appliqués à tous les runs.",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Limite dure sur le nombre de runs (0 = pas de limite).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les commandes sans les exécuter.",
    )
    p.add_argument(
        "--make",
        default="make",
        help="Chemin/commande pour make (défaut: 'make').",
    )
    return p.parse_args()


def build_commands(
    profiles: List[str],
    families: List[str],
    overrides: List[str],
    max_runs: int,
) -> List[Tuple[str, str, str]]:
    """Retourne une liste (profile, family, cmd) à exécuter."""
    ovr_str = " ".join(f"{ovr}" for ovr in overrides) if overrides else ""
    combos = list(itertools.product(profiles, families))
    if max_runs and len(combos) > max_runs:
        combos = combos[:max_runs]

    cmds: List[Tuple[str, str, str]] = []
    for profile, family in combos:
        env_overrides = f'OVERRIDES="{ovr_str}"' if ovr_str else ""
        family_str = f"FAMILY={family}"
        cmd = f'{env_overrides} PROFILE={profile} {family_str} pipeline'
        cmds.append((profile, family, cmd))
    return cmds


def main() -> None:
    args = parse_args()
    cmds = build_commands(args.profiles, args.families, args.overrides, args.max_runs)

    if not cmds:
        print("[run_grid] Aucune combinaison de runs.")
        return

    print(f"[run_grid] {len(cmds)} runs prévus.")

    for profile, family, cmd in cmds:
        full_cmd = f"{args.make} {cmd}"
        print(f"[run_grid] >>> {full_cmd}")
        if args.dry_run:
            continue
        # On passe par shlex.split pour éviter les surprises
        subprocess.run(shlex.split(full_cmd), check=False)


if __name__ == "__main__":
    main()
