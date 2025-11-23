# PEPM_V4 – Pipeline de classification de textes (config-first)

Version `4.0.0-dev` – pipeline supervisé multi-modèles piloté par YAML et Make. Cette page donne les commandes essentielles pour installer l'environnement, préparer une vue, entraîner et évaluer les profils préconfigurés, avec des pointeurs vers la doc détaillée.

## 1. Pré-requis rapides
- Python 3.10+ (CPU only).
- Corpus TEI disponible sous `data/raw/{corpus_id}/corpus.xml`.
- Dépendances : `pip install -r requirements.txt` dans un virtualenv recommandé.

### 1.1 Créer et activer un virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.2 Structure minimale du repo
- Corpus TEI : `data/raw/{corpus_id}/corpus.xml`.
- Sorties intermédiaires : `data/interim/{corpus_id}/{view}/train.tsv` et `job.tsv`.
- Modèles : `models/{corpus_id}/{view}/{family}/{model_id}/`.
- Archives V1/V2/V3 déplacées dans `old_version/` pour ne pas polluer la racine.

## 2. Commandes de base (profils préconfigurés)
Les variables Make principales : `PROFILE` (défaut `ideo_quick`), `OVERRIDES` (clé=val), `FAMILY` (filtre d'entraînement/éval).

### 2.1 Préparer une vue
```bash
# Vue idéologie quick binaire (mode actors)
make prepare PROFILE=ideo_quick

# Variante 5 classes
make prepare PROFILE=ideo_quick OVERRIDES="ideology.view=five_way"

# Intra-camp gauche avec filtre d'acteurs conservés
make prepare PROFILE=ideo_quick \
  OVERRIDES='ideology.view=left_intra,ideology.unknown_actors.policy=keep,actors.include=["MELENCHON","JLM"],actors.min_docs=25'
```
Sorties : `data/interim/{corpus}/{view}/train.tsv`, `job.tsv`, `meta_view.json`, `meta_formats.json`.

# Générer/mettre à jour le mapping acteurs depuis un corpus TEI
```bash
make ideology_skeleton CORPUS_ID=web1
```

### 2.2 Entraîner
```bash
# Toutes les familles actives du profil
make train PROFILE=ideo_quick

# Forcer une famille
make train PROFILE=ideo_quick FAMILY=sklearn

# Tester un modèle HF alternatif (config-first)
make train PROFILE=ideo_quick FAMILY=hf \
  OVERRIDES="models_cfg.families.hf.active=flaubert_quick"
```
Modèles sauvegardés dans `models/{corpus}/{view}/{family}/{model_id}/` avec `meta_model.json`.

### 2.3 Évaluer
```bash
# Évaluation standard sur le job
make evaluate PROFILE=ideo_quick

# Évaluer une seule famille
make evaluate PROFILE=ideo_quick FAMILY=spacy
```
Rapports sous `reports/{corpus}/{view}/{family}/{model_id}/` (`metrics.json`, `classification_report.txt`, `meta_eval.json`).

### 2.4 Check-up rapide
- Vérifier les metas : `jq '.' data/interim/{corpus}/{view}/meta_view.json`.
- Inspecter la répartition des labels : `python scripts/tools/inspect_meta.py --path data/interim/{corpus}/{view}/meta_view.json` (si l'outil est présent).
- Nettoyer un run précédent : `rm -rf data/interim/{corpus}/{view} models/{corpus}/{view} reports/{corpus}/{view}`.

## 3. Profils, configs et overrides
- Profils : `configs/profiles/*.yml` (corpus, vue, bloc `ideology`, filtres, équilibrage, familles actives).
- Configs communes :
  - `configs/common/corpora.yml` (corpus TEI),
  - `configs/common/balance.yml` (équilibrage),
  - `configs/common/models.yml` (catalogue de modèles),
  - `configs/common/hardware.yml` (presets CPU/sharding).
- Idéologie : **source unique** `configs/label_maps/ideology_actors.yml` (mode `actors` avec vues `binary`, `five_way`, `left_intra`, `right_intra`).
  - Les anciens mappings sont conservés en archive dans `configs/label_maps_legacy/`.
- Overrides CLI via `OVERRIDES="cle=val,cle2=val2"` pour modifier ponctuellement un profil.

## 4. Comprendre le pipeline
- Le prepare applique une résolution de labels configurable (bloc `ideology`), filtre éventuellement des acteurs et produit un split **stratifié** avant d'équilibrer le train.
- Les formats spécifiques (DocBin spaCy, TSV HF/sklearn) sont construits par `build_formats` et répertoriés dans `meta_formats.json`.
- `core_train` instancie dynamiquement les modèles déclarés dans `configs/common/models.yml` et trace les hyperparamètres dans `meta_model.json`.
- `core_evaluate` rejoue l'évaluation sur le job et écrit `metrics.json`/`meta_eval.json`.
- Voir **dev_V4.md** pour les diagrammes Mermaid détaillant chaque étape, et **dev_V4.1.md** pour la roadmap d'évolution.

## 5. Diagnostics et métadonnées
- `meta_view.json` : répartition des labels, stratégie d'équilibrage, filtres appliqués, acteurs retenus.
- `meta_formats.json` : chemins des formats par famille, langue, sharding spaCy.
- `meta_model.json` : hyperparamètres et tailles de jeu utilisés à l'entraînement.
- `meta_eval.json` : jeu évalué, métriques, seed et timestamp.

## 6. Documentation détaillée
- **dev_V4.md** : audit complet du pipeline, diagrammes Mermaid du core, backlog fusionné (patch + todo).
- **ref_V4_parameters.md** : tableaux de paramètres et CLI détaillés.
- **dev_V4.1.md** : pistes de développement ultérieur (clustering, upgrades HF, MLOps).

