# scripts/pre/pre_check_env.py

import argparse
import importlib
import sys
from pathlib import Path
import py_compile


CRITICAL_MODULES = [
    "yaml",
    "sklearn",
    "spacy",
    "transformers",
]


def check_imports() -> bool:
    print("[env] Vérification des imports critiques...")
    ok = True
    for mod in CRITICAL_MODULES:
        try:
            importlib.import_module(mod)
            print(f"  [OK] import {mod}")
        except ImportError as e:
            ok = False
            print(f"  [FAIL] import {mod} : {e}")
    return ok


def check_py_compile() -> bool:
    print("[env] Compilation py_compile de scripts/ ...")
    ok = True
    for p in Path("scripts").rglob("*.py"):
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as e:
            ok = False
            print(f"  [FAIL] py_compile {p}: {e}")
        else:
            # optionnel : ne pas spammer
            # print(f"  [OK] {p}")
            pass
    if ok:
        print("  [OK] Tous les scripts compilent.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnostic rapide de l'environnement (imports + py_compile)."
    )
    ap.add_argument(
        "--profile",
        help="Profil à tester (optionnel, juste pour info visuelle).",
    )
    args = ap.parse_args()
    if args.profile:
        print(f"[env] Profil cible (info): {args.profile}")

    ok_imports = check_imports()
    ok_py = check_py_compile()

    if ok_imports and ok_py:
        print("[env] Diagnostic OK.")
        sys.exit(0)
    else:
        print("[env] Diagnostic avec erreurs (voir détails ci-dessus).")
        sys.exit(1)


if __name__ == "__main__":
    main()
