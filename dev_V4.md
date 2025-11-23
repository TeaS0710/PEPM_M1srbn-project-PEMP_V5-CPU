# V4 – Documentation de développement (audit complet)

Cette version combine la documentation de conception V4, les notes de patch et le backlog historique dans un seul document de référence. Elle décrit le pipeline actuel, les points de contrôle clés et les travaux restants.

## 1. Vue d'ensemble du pipeline

### 1.1 Objectifs de V4
- **Config-first** : les expériences se décrivent via `configs/*.yml` + `OVERRIDES`, pas dans le code.
- **Multimodèle** : familles `spacy`, `sklearn`, `hf`, `check` activables au besoin.
- **Multicorpus / multimodale** : chaque TEI porte un `corpus_id` et une `modality` (définie dans les métadonnées ou `corpora.yml`).
- **Reproductibilité** : métadonnées (`meta_view.json`, `meta_formats.json`, `meta_model.json`, `meta_eval.json`) gardent les paramètres et les répartitions de labels.

### 1.2 Flux global (prepare → formats → train → eval)
```mermaid
flowchart TD
  A[Profil YAML + overrides] --> B(core_prepare.collect)
  B --> C{resolve labels}
  C -->|ideology block| D[manual/derived lookup]\n+label_map + unknown policy
  C -->|fallback legacy| E[label_fields + label_map]
  D --> F[filters: actors, modality, min_chars/max_tokens]
  E --> F
  F --> G[stratified split train/job]
  G --> H[balance train only]
  H --> I[train.tsv]
  G --> J[job.tsv]
  I & J --> K[meta_view/meta_formats]
  K --> L{families active}
  L --> M[build_formats]
  M --> N[core_train]
  N --> O[models + meta_model]
  O --> P[core_evaluate]
  P --> Q[reports + meta_eval]
```

1. **core_prepare**
   - Charge le profil (`configs/profiles/{profile}.yml`), applique les presets communs (`corpora`, `hardware`, `balance`, `models`).
   - Extrait les docs TEI (`data/raw/{corpus_id}/corpus.xml`).
   - Résout les labels (bloc `ideology` ou héritage `label_fields`) en appliquant la politique `unknown_labels`.
   - Filtre optionnellement par acteurs (`actors.include`, `actors.min_docs`), qualité (`min_chars`, `max_tokens`) et modality.
   - Réalise un **split stratifié** train/job, puis applique l'**équilibrage uniquement sur le train**.
   - Écrit `train.tsv`, `job.tsv`, `meta_view.json` sous `data/interim/{corpus_id}/{view}/`.

2. **core_prepare.build_formats**
   - Construit les formats par famille à partir des TSV :
     - spaCy : `train*.spacy`, `job*.spacy` (DocBin, shardés si configuré).
     - Autres familles : TSV conservés.
   - Renseigne `meta_formats.json` avec les chemins, tailles, langue, sharding.

3. **core_train**
   - Sélectionne les familles actives (`families.active` ou `--only-family`).
   - Charge les formats, instancie les modèles depuis `configs/common/models.yml` (import dynamique pour sklearn/HF).
   - Entraîne, sauvegarde sous `models/{corpus_id}/{view}/{family}/{model_id}/`, produit `meta_model.json`.

4. **core_evaluate**
   - Charge le modèle et les formats d'éval (job TSV ou DocBin).
   - Produit `metrics.json`, `classification_report.txt`, `meta_eval.json` dans `reports/...`.

## 2. Audit détaillé du prepare

### 2.1 Résolution de label idéologique
- Bloc `ideology` dans les profils (mode par défaut : `actors`) :
  - `actors_yaml` : unique source de vérité (`configs/label_maps/ideology_actors.yml`).
  - `view` : `binary`, `five_way`, `left_intra`, `right_intra` (projection des annotations acteurs).
  - `unknown_actors.policy` : `drop` ou `keep` (+ label optionnel).
- Compatibilité arrière : `mode: legacy` reste possible pour tester les anciens mappings archivés dans `configs/label_maps_legacy/` (déprécié).

### 2.2 Filtrage et nettoyage
- Filtres `min_chars`, `max_tokens`, `modality` appliqués avant split.
- `actors.include` : liste blanche d'acteurs ; `actors.min_docs` supprime les acteurs sous-représentés.
- Les normalisations de texte (espaces, quotes, etc.) restent dans `core_utils.clean_text`.

### 2.3 Split & équilibrage
- **Stratification** par label : chaque classe est séparée selon `train_prop`, puis train/job sont reshufflés avec la `seed` du profil.
- **Équilibrage** : `apply_balance` est appelé **uniquement sur le train** (cap, oversample, alpha_total, class_weights). Le job reste naturel.
- Les compteurs avant/après sont consignés dans `meta_view.json` (label_counts, éventuels `label_weights`).

### 2.4 Sorties de prepare
- `data/interim/{corpus}/{view}/train.tsv` et `job.tsv` : texte brut + label + métadonnées utiles (actor, corpus_id, modality…).
- `meta_view.json` : profil, graines, filtres appliqués, stratégie d'équilibrage, répartition des labels (brute, équilibrée), acteurs retenus.
- `meta_formats.json` : chemins des formats par famille, langue, sharding spaCy, tailles des jeux.

### 2.5 Diagramme détaillé du cœur `core_prepare`
```mermaid
flowchart LR
  subgraph Load
    P[Profil YAML]
    O[Overrides CLI]
    C[Configs communes]
  end
  P -->|merge| R[Profil résolu]
  O -->|apply| R
  C -->|presets| R
  R --> X[Charge TEI]
  X --> Y[Parcours des docs]
  Y --> Z{resolve_ideology_label}
  Z -->|source manual| Z1[label_fields_manual]
  Z -->|source derived| Z2[label_fields_derived]
  Z1 & Z2 --> Z3[label_map + unknown policy]
  Z3 --> F1{label is None?}
  F1 -->|oui| Skip[doc droppé]
  F1 -->|non| F2[attach label + meta]
  F2 --> F3{actors filter}
  F3 -->|in panel| F4
  F3 -->|out| Skip
  F4 --> F5[clean_text + constraints]
  F5 --> F6[stratified_split(train_prop)]
  F6 --> T[Train]
  F6 --> J[Job]
  T --> B[apply_balance(train)]
  B --> T2[train équilibré]
  T2 & J --> Out[Écriture TSV + metas]
```

Points à surveiller :
- `resolve_ideology_label` applique strictement la politique `unknown_labels` (drop/keep/other) et retourne `None` si aucune valeur utilisable n'est trouvée.
- Le split stratifié se fait par label pour garantir la représentativité du job ; aucun oversampling ne touche le job.
- Les compteurs avant/après équilibrage et les filtres d'acteurs sont tracés dans `meta_view.json`.

## 3. Formats & entraînement

### 3.1 Families et formats
- **spaCy** : DocBin `train*.spacy` / `job*.spacy`; sharding contrôlé par hardware preset ; templates `configs/spacy/*.cfg` utilisables via `config_template` dans `models.yml`.
- **sklearn** : TSV ; `build_vectorizer` et `build_estimator` instanciés par import dynamique selon `models.yml` (ngram_range auto normalisé, class_weight possible).
- **HF** : TSV ; paramètres `pretrained_model_name_or_path`, `max_length`, `trainer_params`, `use_class_weights` dans `models.yml`.
- **check** : sanity-check minimal, utilise les TSV directement.

```mermaid
flowchart TD
  Fmt[build_formats] -->|spaCy| S1[DocBin train/job]
  Fmt -->|sklearn/hf/check| S2[TSV train/job]
  subgraph Train
    Sel[filtre famille active] --> Load[charge formats]
    Load --> Inst[instanciation modèle dyn.]
    Inst --> Fit[train + logs]
    Fit --> Save[models/{corpus}/{view}/{family}/{model_id}]
    Save --> MM[meta_model.json]
  end
  S1 & S2 --> Train
  Train --> Eval[core_evaluate]
  Eval --> Rep[reports + meta_eval]
```

Notes d'implémentation :
- `build_vectorizer` / `build_estimator` (sklearn) utilisent `importlib` pour instancier les classes déclarées dans `models.yml`.
- Pour HF, `pretrained_model_name_or_path` pilote directement `AutoTokenizer` et `AutoModelForSequenceClassification`; les paramètres `trainer_params` sont passés tels quels au `Trainer`.
- La famille `check` lit simplement les TSV pour vérifier la cohérence des formats (utile pour le CI manuel rapide).

### 3.2 Métadonnées d'entraînement/éval
- `meta_model.json` : famille, modèle, hyperparams, tailles de jeu, chemins de formats utilisés, label2id/id2label pour HF.
- `meta_eval.json` : profil, modèle, famille, jeu évalué, métriques de base, timestamp, seed.

## 4. Configuration & orchestration

### 4.1 Fichiers clés
- `configs/common/corpora.yml` : chemins TEI, encodage, modality par défaut.
- `configs/common/balance.yml` : stratégies et presets d'équilibrage.
- `configs/common/models.yml` : catalogue des modèles par famille (spaCy, sklearn, HF, check).
- `configs/common/hardware.yml` : presets RAM/CPU, shards spaCy, limites de docs.
- `configs/label_maps/ideology_actors.yml` : mapping acteurs unique (vues binary/five_way/left_intra/right_intra).
- `configs/label_maps_legacy/` : anciens mappings conservés à titre d'archive (mode `legacy`).
- `configs/profiles/*.yml` : expérience complète (corpus, vue, filtres, stratégie d'équilibrage, famille active, bloc `ideology`).

### 4.2 Overrides & Makefile
- Variables Make usuelles : `PROFILE` (défaut `ideo_quick`), `OVERRIDES` (clé=val), `FAMILY` (filtre d'entraînement/éval).
- Exemples d'override idéologie :
  - `OVERRIDES="ideology.view=five_way"`
  - `OVERRIDES="ideology.view=left_intra"`
  - `OVERRIDES='ideology.unknown_actors.policy=keep,actors.include=["MELENCHON"],actors.min_docs=50'`

## 5. Backlog V4 (fusion patch + todo)

### 5.1 Prepare & data layer
- Multi-tokenizer + `max_tokens` réel (helpers spaCy, stats tokens dans `meta_view`).
- DocBin + sharding spaCy paramétrables via `hardware.yml` + `build_docbins` dédié.
- Gestion complète des stratégies d'équilibrage V2 (cap_docs, cap_tokens, alpha_total, class_weights) avec stats détaillées.

### 5.2 Entraînement
- spaCy config-first : templates `textcat_bow_base.cfg`/`textcat_cnn_base.cfg`, overrides hyperparams, meta enrichi (`arch`, `config_template`).
- Sklearn : plafond `max_train_docs_sklearn`, traçabilité `vectorizer/estimator` + class_weight.
- HF : training générique CPU (dataset HF, Trainer), support `use_class_weights`, évaluation harmonisée.
- Famille `check` : maintenir la compatibilité pour les tests rapides.

### 5.3 Évaluation & tooling
- Harmonisation des fonctions d'éval (spaCy / sklearn / HF) et format de `meta_eval`.
- Scripts de pré-check (`pre_check_config.py`) pour valider existence des configs, familles, hardware preset, label maps.
- Jeu d'échantillons et cibles Make (`sample_prepare`, `sample_train`, `sample_pipeline`).
- Logging unifié et seeding étendu (random / numpy / torch) avec traçabilité dans les metas.

## 6. Référentiels complémentaires
- Les tableaux détaillés de paramètres et de CLI restent décrits dans `ref_V4_parameters.md`.
- Le README fournit les commandes courantes et les prérequis d'exécution.

