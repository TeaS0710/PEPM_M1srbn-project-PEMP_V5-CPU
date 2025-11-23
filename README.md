# PEPM_M1srbn-project-PEMP_V4.05-CPU

Pipeline **config-first** pour l’analyse de textes politiques (idéologie) à grande échelle, basé sur :

* un **corpus TEI XML** volumineux (≈ 3+ Go),
* un cœur **multi-corpus / multi-vues / multi-familles de modèles**,
* une logique d’**annotation idéologique dérivée** à partir d’un mapping acteurs ⇨ idéologie,
* un entraînement **multi-méthodes** (spaCy, sklearn, HF) piloté par YAML.

Le but : pouvoir lancer, rejouer et comparer des expériences complètes (préparation, entraînement, évaluation) uniquement via :

* des **profils YAML** (`configs/profiles/*.yml`),
* quelques **variables Makefile** (`PROFILE`, `CORPUS_ID`, etc.),
* et des `--override key=val` pour les cas spécifiques.

Version : **V4.05 – CPU**.

---

## 1. Vue d’ensemble

Le pipeline V4.05 est structuré autour de trois scripts « cœur » :

* `scripts/core/core_prepare.py`
  ➜ TEI XML → TSV / formats entraînables (spaCy, sklearn, HF).

* `scripts/core/core_train.py`
  ➜ Entraîne les modèles listés dans `configs/common/models.yml` pour chaque **famille** (`check`, `spacy`, `sklearn`, `hf`).

* `scripts/core/core_evaluate.py`
  ➜ Charge les modèles, applique sur le jeu de test/job, produit des rapports (`reports/...`).

Tout est piloté par un **profil** (`configs/profiles/ideo_quick.yml` par exemple) :

* **corpus** : `corpus_id` (ex. `web1`)
* **vue** : `view` (ex. `ideology_global`)
* **familles** : `families: [check, spacy, sklearn]`
* **modèles** : `models_spacy`, `models_sklearn`, `models_hf`, `models_check`
* **idéologie** : bloc `ideology: { ... }` basé sur `ideology_actors.yml`
* **hardware** : `hardware_preset`, limites `max_train_docs_*`
* **équilibrage** : `balance_strategy`, `balance_preset`
* **split** : `train_prop`, `seed`

---

## 2. Installation & prérequis

### 2.1. Prérequis

* Python **3.10+** recommandé
* `git` + `make`
* CPU « raisonnable » (le pipeline est prévu pour **CPU**, avec presets `small`, `lab`, etc.)
* Un corpus TEI XML placé dans :
  `data/raw/web1/corpus.xml` (par défaut)

### 2.2. Installation rapide

Dans le répertoire racine du projet :

```bash
# Installation / venv / dépendances / arbo de base / check
make setup
```

`make setup` enchaîne :

1. `make venv`
   → crée `.venv/` si nécessaire et installe `pip` à jour.
2. `make install`
   → installe les dépendances de `requirements.txt` dans `.venv`.
3. `make init_dirs`
   → crée l’arborescence minimale sous `data/`, `models/`, `reports/`, `logs/`.
4. `make check PROFILE=ideo_quick`
   → diagnostics d’environnement + validation du profil par défaut.

Cela crée notamment :

* `.venv/` : environnement virtuel
* `data/raw/web1/` (si non présent)
* `data/interim/web1/ideology_global/`
* `data/processed/web1/ideology_global/`
* `models/web1/ideology_global/`
* `reports/web1/ideology_global/`
* `logs/`

Ensuite, assure-toi que ton corpus TEI est bien ici :

```bash
ls -lh data/raw/web1/corpus.xml
```

*(ou adapte `CORPUS_ID` et `corpora.yml` si tu utilises un autre identifiant)*

---

## 3. Organisation du dépôt

Schéma simplifié :

```text
.
├── configs
│   ├── common
│   │   ├── balance.yml       # stratégies d'équilibrage / presets
│   │   ├── corpora.yml       # métadonnées par corpus_id (web1, web2…)
│   │   ├── hardware.yml      # presets hardware (small, lab, ...)
│   │   └── models.yml        # définition des modèles (spaCy, sklearn, HF, check)
│   ├── label_maps
│   │   ├── ideology.yml      # base conceptuelle : familles idéologiques
│   │   ├── ideology_actors.yml
│   │   │   # mapping par acteur (domaines / crawls → acteur → labels global/intra/binaire)
│   │   ├── ideology_global.yml
│   │   ├── ideology_left_intra.yml
│   │   └── ideology_right_intra.yml
│   └── profiles
│       ├── ideo_quick.yml    # profil principal : idéologie, vue globale (rapide)
│       ├── ideo_full.yml     # version plus lourde
│       ├── crawl_quick.yml   # profils dédiés au crawl (si utilisé)
│       ├── crawl_full.yml
│       ├── check_only.yml    # profil simple de validation
│       └── custom.yml        # squelette pour expériences spécifiques
├── data
│   ├── raw
│   │   └── web1/corpus.xml   # corpus TEI principal
│   ├── interim
│   │   └── web1/ideology_global   # TSV + formats intermédiaires
│   ├── processed
│   │   └── web1/ideology_global   # données prêtes pour entraînement/éval
│   └── configs
│       └── actors_counts_web1.tsv # rapport acteurs (make_ideology_skeleton)
├── models
│   └── web1/ideology_global      # artefacts (spaCy, joblib sklearn, HF, meta_model.json)
├── reports
│   └── web1/ideology_global      # métriques, rapports détaillés
├── scripts
│   ├── core
│   │   ├── core_prepare.py
│   │   ├── core_train.py
│   │   ├── core_evaluate.py
│   │   └── core_utils.py
│   ├── pre
│   │   ├── pre_check_env.py
│   │   ├── pre_check_config.py
│   │   ├── make_ideology_skeleton.py
│   │   └── derive_ideology_from_yaml.py
│   ├── post
│   │   └── post_aggregate_metrics.py
│   ├── tools
│   │   ├── corpus_stats.py
│   │   └── sysinfo.py
│   └── experiments
│       └── run_grid.py
├── makefile
├── README.md
├── dev_V4.md               # doc dev (architecture détaillée)
└── ref_V4_parameters.md    # référence détaillée des paramètres (si présent)
```

---

## 4. Logique du pipeline & modules

### 4.1. Flow global

```mermaid
flowchart LR
    A[TEI corpus.xml<br/>data/raw/webX/corpus.xml]
      --> B[core_prepare.py<br/>STAGE=prepare]
    B -->|TSV + *.spacy + meta| C[data/interim / data/processed]
    C --> D[core_train.py<br/>STAGE=train]
    D --> E[models/webX/view/<famille>/<modèle>]
    C --> F[core_evaluate.py<br/>STAGE=evaluate]
    E --> F
    F --> G[reports/webX/view/...]
```

### 4.2. `core_prepare.py`

* parse le TEI, extrait les textes, métadonnées (`crawl`, `domain`, `modality`, etc.),
* résout le label idéologique via `ideology_actors.yml` et la vue demandée,
* applique des filtres (longueur, modalités, acteurs),
* split **stratifié** train/job (`train_prop`, `seed`),
* applique un **équilibrage** selon `balance.yml`,
* produit :

  * TSV par split (`train.tsv`, `job.tsv`),
  * formats pour spaCy (`*.spacy`, DocBin shardés pour éviter les erreurs de taille),
  * matrices/TSV pour sklearn / HF,
  * méta-fichiers (`meta_view.json`, `meta_formats.json`).

### 4.3. `core_train.py`

* lit `configs/common/models.yml`,
* pour chaque **famille** demandée (`families` dans le profil) :

  * **spaCy** : charge les DocBin, entraîne selon `textcat_*` et sauvegarde le pipeline,
  * **sklearn** : vectorisation TF-IDF/BOW, modèles SVM, Perceptron, RandomForest, etc.,
  * **HF** : support expérimental / squelette selon la config,
  * **check** : tests de base + sauvegarde d’un `meta_model.json` de contrôle.

### 4.4. `core_evaluate.py`

* charge les modèles entraînés,
* évalue sur le split **job** (test),
* écrit :

  * métriques agrégées (accuracy, F1 macro/micro, F1 par classe),
  * fichiers de rapport dans `reports/...` (JSON, TSV, etc.),
  * `meta_eval.json` pour tracer les conditions d’évaluation.

`scripts/post/post_aggregate_metrics.py` permet ensuite d’agréger plusieurs runs.

---

## 5. Configuration : profils & cartes d’idéologie

### 5.1. Profils (`configs/profiles/*.yml`)

Un profil contient notamment :

* **métadonnées** : `profile`, `description`
* **corpus / vue** : `corpus_id`, `view`, `modality`
* **idéologie** : bloc `ideology: { ... }`
* **familles / modèles** : `families`, `models_spacy`, `models_sklearn`, `models_hf`, `models_check`
* **split / équilibrage** : `train_prop`, `seed`, `balance_strategy`, `balance_preset`
* **hardware** : `hardware_preset`, `max_train_docs_*`
* **debug** : `debug_mode`, éventuellement des limites de taille (`debug_max_*`)

Exemple ultra-simplifié (`ideo_quick`) :

```yaml
profile: ideo_quick
description: >
  Profil rapide pour vue ideology_global sur web1.

corpus_id: web1
view: ideology_global
modality: text

ideology:
  mode: actors
  view: global_five
  actors_yaml: configs/label_maps/ideology_actors.yml
  unknown_actors:
    policy: drop   # ou keep
    label: unknown_actor

families: [check, spacy, sklearn]

models_spacy:
  - spacy_cnn_quick
  - spacy_bow_quick

models_sklearn:
  - tfidf_svm_quick

hardware_preset: small
train_prop: 0.8
seed: 42
balance_strategy: alpha_total
balance_preset: default
```

### 5.2. Maps d’idéologie

Le design sépare :

1. **Base conceptuelle** : `configs/label_maps/ideology.yml`

   * décrit la structure des familles idéologiques (`far_right`, `right`, `center`, `left`, `far_left`),
   * regroupe des acteurs dans des ensembles.

2. **Mapping par acteur** : `configs/label_maps/ideology_actors.yml`

   * généré par `make_ideology_skeleton` à partir du TEI (crawl / domain),
   * enrichi par `derive_ideology_from_yaml.py` (labels global/intra, binary, etc.),
   * corrigé / affiné manuellement si nécessaire.

3. **Vues dérivées** :

   * `ideology_global.yml` (5 classes),
   * `ideology_left_intra.yml`,
   * `ideology_right_intra.yml`.

#### Commandes associées

**Générer un squelette d’acteurs** (à partir du TEI) :

```bash
make ideology_skeleton CORPUS_ID=web1
# -> configs/label_maps/ideology_actors.yml
# -> data/configs/actors_counts_web1.tsv
```

**Dériver l’idéologie globale & intra** depuis `ideology.yml` :

```bash
make ideology_from_yaml CORPUS_ID=web1
# -> MAJ de ideology_actors.yml
# -> configs/label_maps/ideology_global.yml
# -> configs/label_maps/ideology_left_intra.yml
# -> configs/label_maps/ideology_right_intra.yml
```

Tout en une fois :

```bash
make ideology_all CORPUS_ID=web1
```

Workflow typique :

1. `make ideology_skeleton` pour découvrir / inspecter les acteurs.
2. Affiner `ideology.yml` (groupes d’acteurs → familles idéologiques).
3. `make ideology_from_yaml` pour projeter cette logique sur `ideology_actors.yml` + générer les vues global/intra.
4. Corriger éventuellement `ideology_actors.yml` à la main pour les cas limites.

---

## 6. Paramètres du pipeline

Pour le détail exhaustif, se référer à **`ref_V4_parameters.md`** (si présent).
Ici : vue synthétique des paramètres utiles côté utilisateur.

### 6.1. Variables Makefile / CLI

Ces variables sont converties en `--override` automatiquement par le Makefile :

* `PROFILE` → `profile` (nom du profil YAML sans extension)
* `CORPUS_ID` → `corpus_id` (doit exister dans `common/corpora.yml`)
* `TRAIN_PROP` → `train_prop` (0.0–1.0)
* `BALANCE_STRATEGY` → `balance_strategy`
* `BALANCE_PRESET` → `balance_preset`
* `HARDWARE_PRESET` → `hardware_preset`
* `FAMILIES` → `families` (liste séparée par virgules : `spacy,sklearn`)
* `SEED` → `seed`
* `MAX_DOCS_SKLEARN` → `max_train_docs_sklearn`
* `MAX_DOCS_SPACY` → `max_train_docs_spacy`
* `MAX_DOCS_HF` → `max_train_docs_hf`
* `OVERRIDES` → liste libre d’overrides `key=value`
  (ex : `OVERRIDES="ideology.view=binary debug_mode=true"`)

### 6.2. Paramètres globaux de profil

* `profile` : nom logique du profil (`ideo_quick`).
* `description` : description textuelle.
* `corpus_id` : identifiant du corpus (clé dans `corpora.yml`).
* `view` : vue/logique de tâche (ex : `ideology_global`).
* `modality` : type de données (souvent `text` pour le web).

### 6.3. Paramètres d’idéologie

Dans le bloc `ideology` :

* `mode` : actuellement `actors` (pipeline basé sur `ideology_actors.yml`).
* `view` : type de label final, par exemple :

  * `global_five` : `far_right` / `right` / `center` / `left` / `far_left`
  * `binary` : `right` / `left`
  * `right_intra`, `left_intra` : analyses intra-camp
* `actors_yaml` : chemin vers `configs/label_maps/ideology_actors.yml`.
* `unknown_actors.policy` :

  * `drop` : docs non mappés ignorés,
  * `keep` : docs non mappés reçoivent un label `unknown_actors.label`.
* `unknown_actors.label` : label pour les inconnus (`unknown_actor`, etc.).

### 6.4. Paramètres de filtrage / split

* `train_prop` : proportion de documents dans le split train (stratifié).
* `seed` : seed globale (random, numpy, torch, spaCy).
* `dedup_on` : colonnes utilisées pour la déduplication (si configuré).
* `min_chars` : nb de caractères minimum par document.
* `max_tokens` : nb de tokens max (pour couper les textes trop longs).
* `actors.include` / `actors.exclude` : listes d’acteurs à garder / exclure.
* `actors.min_docs` : nb min de docs par acteur pour être conservé.

### 6.5. Paramètres d’équilibrage (balance)

Pilotés par `balance.yml` + les champs :

* `balance_strategy` :

  * `none` : pas d’équilibrage,
  * `alpha_total` : pondération type alpha sur les classes,
  * `cap_docs` : limite le nb de docs par classe,
  * `class_weights` : calcul de poids de classe (exploité côté sklearn).
* `balance_preset` : nom d’un preset dans `balance.yml` (ex. `default`, `binary`, etc.).
* `balance_mode` : alias humain possible, mappé vers `balance_strategy` dans `core_utils`.

Les poids de classes calculés peuvent être réutilisés dans sklearn (`class_weight: from_balance` dans `models.yml`).

### 6.6. Familles & modèles

* `families` : liste des familles activées :

  * `check`, `spacy`, `sklearn`, `hf`
* `models_spacy` : liste de clés définies dans `models.yml`,
* `models_sklearn` : idem,
* `models_hf` : modèles HF (optionnel, dépend du hardware),
* `models_check` : pseudo-modèles / diagnostics.

Les hyperparamètres détaillés sont dans `configs/common/models.yml`.

### 6.7. Paramètres hardware

* `hardware_preset` : clé d’un preset dans `configs/common/hardware.yml` (`small`, `lab`, etc.).
* bloc `hardware` dérivé contenant par ex. :

  * `n_jobs_sklearn`
  * `blas_threads`
  * `hf_device` (cpu, cuda, etc.)
  * `hf_batch_size`
* Surcouches top-level :

  * `max_train_docs_sklearn`
  * `max_train_docs_spacy`
  * `max_train_docs_hf`

Ces champs permettent de limiter la taille effective des jeux d’entraînement par famille.

### 6.8. Debug / limites

* `debug_mode: true` : logs plus verbeux, protections supplémentaires.
* certains profils peuvent exposer :

  * `debug_max_train`, `debug_max_job`, `debug_shuffle`, etc.

---

## 7. Commandes & scénarios d’utilisation

### 7.1. Point d’entrée universel (Makefile)

Le Makefile définit un point d’entrée unique :

```bash
make run STAGE=<stage> PROFILE=<profil> [variables...]
```

* `STAGE` ∈ `check | prepare | prepare_dry | train | evaluate | pipeline`
* `PROFILE` = nom d’un profil YAML (sans `.yml`), ex. `ideo_quick`.

Quelques cibles directes :

```bash
# Lister les profils disponibles
make list_profiles

# Pipeline complet (check + prepare + train + evaluate)
make run STAGE=pipeline PROFILE=ideo_quick

# Préparation seule
make run STAGE=prepare PROFILE=ideo_quick

# Entraînement seul
make run STAGE=train PROFILE=ideo_quick

# Évaluation seule
make run STAGE=evaluate PROFILE=ideo_quick
```

### 7.2. Premier lancement sur le corpus complet web1 (idéologie)

En supposant que `data/raw/web1/corpus.xml` contient ton corpus complet (~3.1 Go).

1. Vérifier / installer :

   ```bash
   make setup
   ```

2. Générer / mettre à jour les maps d’idéologie :

   ```bash
   # Squelette + stats acteurs
   make ideology_skeleton CORPUS_ID=web1

   # Projection des groupes d'acteurs (ideology.yml) sur les acteurs
   make ideology_from_yaml CORPUS_ID=web1
   # ou tout en une fois :
   make ideology_all CORPUS_ID=web1
   ```

3. Lancer le pipeline complet (profil rapide `ideo_quick`) :

   ```bash
   make run STAGE=pipeline PROFILE=ideo_quick CORPUS_ID=web1 \
     HARDWARE_PRESET=small \
     TRAIN_PROP=0.8 \
     SEED=42
   ```

Cela va :

* vérifier la config (`check`),
* préparer les données (`prepare`)
  → `data/interim/web1/ideology_global`
  → `data/processed/web1/ideology_global`
* entraîner les modèles (`train`)
  → `models/web1/ideology_global/...`
* évaluer et produire des rapports (`evaluate`)
  → `reports/web1/ideology_global/...`

### 7.3. Overrides avancés

Quelques exemples utiles :

```bash
# Changer de vue idéologique (ex: binary) sans éditer le profil
make run STAGE=pipeline PROFILE=ideo_quick CORPUS_ID=web1 \
  OVERRIDES="ideology.view=binary"

# Garder les acteurs inconnus comme une classe à part
make run STAGE=pipeline PROFILE=ideo_quick CORPUS_ID=web1 \
  OVERRIDES="ideology.unknown_actors.policy=keep ideology.unknown_actors.label=unknown"

# Limiter les docs entraînés pour un test rapide
make run STAGE=pipeline PROFILE=ideo_quick CORPUS_ID=web1 \
  MAX_DOCS_SKLEARN=5000 MAX_DOCS_SPACY=5000
```

Changer de preset hardware et de proportion d’entraînement :

```bash
make run STAGE=pipeline PROFILE=ideo_quick CORPUS_ID=web1 \
  HARDWARE_PRESET=lab TRAIN_PROP=0.7
```

### 7.4. Outils de diagnostic

```bash
# Informations système & versions
make sysinfo

# Check de l'env (libs, chemins…)
make diagnostics PROFILE=ideo_quick

# Compilation à blanc de tous les scripts Python
make check_scripts

# Vérification complète (env + scripts + profil + configs)
make check PROFILE=ideo_quick
```

### 7.5. Nettoyage

```bash
make clean
# Supprime :
#   data/interim/*
#   data/processed/*
#   models/*
# (les raw + configs sont conservés)
```

---

## 8. Interprétation des sorties

* `data/interim/...` :
  TSV, DocBin spaCy (splittés), métadonnées diverses.
  → utile pour debug ou réutilisation.

* `data/processed/...` :
  formats « propres » par split, prêts pour entraînement/éval.

* `models/...` : artefacts de modèles :

  * `spaCy` : dossiers de pipelines (chargeables via `spacy.load`),
  * `sklearn` : `model.joblib` + pipeline associé,
  * `hf` : checkpoints / configs (si utilisé),
  * `check` : `meta_model.json` + résumés.

* `reports/...` :
  JSON / TSV / (éventuels PNG) résumant :

  * accuracy, F1 macro/micro, F1 par classe,
  * matrices de confusion,
  * comparaisons inter-modèles.

Pour des agrégations multi-runs :

```bash
python scripts/post/post_aggregate_metrics.py ...
```

---

## 9. Documentation interne

* `dev_V4.md` : architecture détaillée du core, explications internes, choix d’implémentation, notes techniques.
* `ref_V4_parameters.md` : référence détaillée des paramètres de profils (tous les champs, valeurs par défaut, interactions).
* `configs/common/models.yml` : définition des modèles (hyperparamètres spaCy / sklearn / HF).
