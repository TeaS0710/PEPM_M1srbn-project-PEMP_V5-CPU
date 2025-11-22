# V4 – Documentation de développement (core + orchestration)

## 0. Contexte

V4 est une refonte du pipeline PEPM visant à :

* rester **au moins aussi riche que V2** (équilibrage, profils, flexibilité),
* éviter les erreurs de V3 (scripts jolis mais pipeline cassé / moins lisible),
* préparer une vraie architecture **multimodèle, multimodale, multicorpus**,
* avec un coeur **config-first** : on change les expériences en YAML, pas en Python.

---

## 1. Objectifs de V4

### 1.1 Objectif général

Construire un pipeline :

* **multimodèle**

  * spaCy (TextCat),
  * sklearn : SVM/SMO, Perceptron, DecisionTree, RandomForest,
  * HF : CamemBERT, FlauBERT, BERT (CPU),
  * “check” comme pseudo-modèle de sanity-checks.

* **multimodale**

  * plusieurs types de corpus : `web1`, `web2`, (futur) `asr`, `gold`, etc.,
  * modalités définies soit par le TEI (`<term type="modality">`),
  * soit par les configs `corpora.yml` (default_modality).

* **multicorpus**

  * plusieurs TEI différents, chacun identifié par `corpus_id`.

* **hyperflexible**

  * ajout de modèles/corpus/vues **sans toucher aux scripts core**,
  * tout se fait dans :

    * `configs/common/*.yml`,
    * `configs/profiles/*.yml`,
    * et via `OVERRIDES`.

* **config-first**

  * scripts Python très génériques et stables,
  * Makefile ultra-léger (juste un routeur),
  * paramétrage centralisé dans YAML.

### 1.2 Objectifs de non-régression

Par rapport à V1/V2 :

* **conserver** :

  * l’équilibrage des classes (`alpha_total`, `cap_docs`, `cap_tokens`),
  * les filtres (min_chars, max_tokens, modality),
  * la logique de split train/job (`train_prop`, `seed`),
  * le contrôle matériel (RAM, n_jobs, BLAS threads),
  * la reproductibilité (meta fichiers, logs).

Par rapport à V3 :

* **éviter** :

  * le désalignement Makefile/scripts,
  * les refactors qui suppriment de la logique scientifique,
  * la complexité inutile dans les scripts (plus de forme que de fond).

---

## 2. Architecture globale V4

### 2.1 Arborescence

## Arborescence cible du pipeline V4

L’architecture V4 repose sur une séparation nette entre :

- la **configuration** (`configs/`),
- les **données** (`data/`),
- les **modèles entraînés** (`models/`),
- les **rapports / métriques** (`reports/`),
- les **scripts** (`scripts/`).

Arborescence cible :

```text
data/
  raw/
    {corpus_id}/
      corpus.xml

  interim/
    {corpus_id}/{view}/
      train.tsv
      job.tsv
      meta_view.json

  processed/
    {corpus_id}/{view}/
      meta_formats.json
      spacy/
        train.spacy
        job.spacy
        train_000.spacy ...
      sklearn/
      hf/

models/
  {corpus_id}/{view}/{family}/{model_id}/
    meta_model.json
    ... fichiers propres à la famille (joblib, spaCy, HF)

reports/
  {corpus_id}/{view}/{family}/{model_id}/
    metrics.json
    classification_report.txt
    meta_eval.json

logs/
  run-YYYYMMDD-HHMMSS.log (optionnel)


---

## 3. Système de configuration

### 3.1 YAML communs : `configs/common/*.yml`

#### `corpora.yml`

* Déclare les **corpus sources** (TEI).

Exemple :

```yaml
web1:
  corpus_id: "web1"
  corpus_path: "data/raw/web1/corpus.xml"
  encoding: "utf-8"
  default_modality: "web"
  notes: "Corpus web initial"

web2:
  corpus_id: "web2"
  corpus_path: "data/raw/web2/corpus.xml"
  encoding: "utf-8"
  default_modality: "web"
  notes: "Deuxième corpus web"
```

#### `balance.yml`

* Déclare les stratégies d’équilibrage + presets.

Exemple :

```yaml
strategies:
  none:
    desc: "Pas d'équilibrage"

  alpha_total:
    desc: "Répartition alpha / total_docs"
    presets:
      default_alpha_total:
        alpha: 0.5
        total_docs: 50000
      small_alpha_total:
        alpha: 0.3
        total_docs: 20000

  cap_docs:
    desc: "Cap docs par label"
    presets:
      default_cap_docs:
        cap_per_label: 5000

  cap_tokens:
    desc: "Cap tokens par label"
    presets:
      default_cap_tokens:
        cap_tokens_per_label: 1000000
```

**Implémentation actuelle :**

* `none`, `cap_docs`, `alpha_total` implémentés (version V1 simplifiée).
* `cap_tokens` : placeholder (TODO).

### 3.4 Champs label : liste et normalisation

* `label_field` peut être une **chaîne** ou une **liste** (`label_fields`) ; priorité à la liste si présente, avec fallback vers le premier champ trouvant un `<term type="...">`.
* Les clés du `label_map` sont normalisées (minuscules, espaces/`-`/`/` → `_`, caractères spéciaux nettoyés) et les labels bruts sont toujours convertis dans ce format avant lookup. Exemple : `crawl-actionfrancaise-20251003_000000` → `crawl_actionfrancaise_20251003_000000` pour matcher la clé YAML.
* En l'absence de `label_map`, le label utilisé est la version **normalisée** du terme brut, garantissant cohérence de casse et de séparateurs.
* Les profils existants restent compatibles : la liste reste optionnelle et `label_field` seul fonctionne toujours.

---

## 4. Corpus TEI : structure et conventions

Cette section résume le corpus TEI ciblé par V4 (namespace `http://www.tei-c.org/ns/1.0`). Les chemins sont donnés sans préfixe de namespace pour la lisibilité.

### 4.1 Vue d'ensemble

* Racine unique `<teiCorpus>` avec un `<teiHeader>` **corpus-level** puis ~528k éléments `<TEI>` (un par document).
* Espaces de noms :
  * `tei`: `http://www.tei-c.org/ns/1.0`
  * `xml`: `http://www.w3.org/XML/1998/namespace` (pour `xml:id`, `xml:lang`).

```mermaid
flowchart TB
  teiCorpus["teiCorpus (1)"]
  teiCorpus --> corpusHeader["teiHeader (corpus)"]
  teiCorpus --> teiDoc["TEI (×528k)"]

  subgraph TEI_DOC ["Structure d'un TEI"]
    teiDoc --> docHeader["teiHeader (document)"]
    teiDoc --> docText["text @xml:lang"]
    teiDoc --> docStandOff["standOff/listAnnotation"]
    docText --> docBody["body"] --> docDiv["div @type"]
    docDiv --> docHead["head"]
    docDiv --> docP["p (texte principal)"]
  end

  subgraph METADATA ["Extraits de métadonnées"]
    docHeader --> fileDesc["fileDesc → titleStmt, publicationStmt, sourceDesc/biblStruct/idno[type]"]
    docHeader --> profileDesc["profileDesc/textClass/keywords/term[type]"]
    profileDesc --> termClass["term type='class' | 'crawl' | 'domain' | 'suffix' | 'published' | ..."]
    docHeader --> revisionDesc["revisionDesc/change (md5_text, chars)"]
  end
```

### 4.2 Détails par niveau

* **`<teiCorpus>`** : titre global, publisher, date, source.
* **`<TEI xml:id>`** : identifiant stable (utilisé en priorité pour `doc_id`).
* **`teiHeader` document** :
  * `fileDesc/titleStmt/title` et `respStmt` (origine extraction),
  * `publicationStmt` (publisher, availability/licence, date `@when`, pubPlace),
  * `sourceDesc/biblStruct/idno@type` (URL, chemin compressé, md5_html),
  * `profileDesc/langUsage/language@ident` (≈528k occurrences),
  * `textClass/keywords/term@type` : **labels potentiels** (`ideology` si présent, sinon `crawl`, `class`, etc.).
* **`text@xml:lang`** : bloc supervisé ; contient `body/div/head` + `p` (présent pour chaque doc).
* **`revisionDesc/change`** : méta calculées (md5_text, chars) utilisables pour QA.

---

## 5. Modélisation (Mermaid)

### 5.1 Flux core V4

```mermaid
flowchart LR
  subgraph CLI ["Entrée utilisateur (CLI / Makefile)"]
    cli_prepare["python core_prepare.py\n--profile --override..."]
    cli_train["python core_train.py\n--profile --override..."]
    cli_eval["python core_evaluate.py\n--profile --override...\n[--only-family] [--dry-run]"]
  end

  subgraph CONFIGS ["Configs YAML (source unique de vérité)"]
    cfg_profile["configs/profiles/*.yml\nprofile, corpus_id, view, modality,\nlabel_field(s), label_map, train_prop,\nmin_chars, max_tokens, balance_*,\nfamilies, models_*, hardware_preset,\ndebug_mode, seed"]
    cfg_corpora["configs/common/corpora.yml\ncorpus_id → corpus_path, encoding,\ndefault_modality, notes"]
    cfg_balance["configs/common/balance.yml\nstrategies: none, alpha_total,\ncap_docs, cap_tokens (TODO)\n+ presets (alpha, total_docs, cap_per_label, ...)"]
    cfg_hardware["configs/common/hardware.yml\npresets: small, medium, lab\nram_gb, max_procs, blas_threads,\ntsv_chunk_rows, spacy_shard_docs"]
    cfg_models["configs/common/models.yml\nfamilles: spacy / sklearn / hf\nhyperparams: epochs, lr,\nvectorizer, estimator, trainer_params..."]
    cfg_labelmaps["configs/label_maps/*.yml\nmapping label_raw → label\n+ unknown_labels.policy"]
  end

  subgraph CORE_UTILS ["core_utils.resolve_profile_base()"]
    core_utils["Fusion profil + overrides + commons\n→ params (dict unifié)"]
    blas_env["set_blas_threads()\n→ OMP_NUM_THREADS / MKL / OPENBLAS"]
  end

  subgraph RAW ["Données brutes"]
    tei_xml["data/raw/{corpus_id}/corpus.xml\nTEI complet (textes + <term> + modalités)"]
  end

  subgraph PREPARE ["core_prepare.py\n(build_view + build_formats)"]
    prep_stream["Streaming TEI\niterparse(<TEI>) + elem.clear()"]
    prep_extract["Extraction par doc\nid(xml:id), text (<text>), modality, label_raw\nlabel_field ou label_fields"]
    prep_labelmap["load_label_map()\nlabel_raw → label\nmatch exact + label normalisé"]
    prep_filters["Filtres\nmin_chars, max_tokens (tokenizer split/simple/spacy), modality"]
    prep_balance["apply_balance()\n• none\n• alpha_total\n• cap_docs\n• cap_tokens (placeholder)"]
    prep_split["Split train/job\ntrain_prop + seed"]
    tsv_train["data/interim/{corpus_id}/{view}/train.tsv"]
    tsv_job["data/interim/{corpus_id}/{view}/job.tsv"]
    meta_view["meta_view.json"]
    spacy_train["data/processed/{corpus_id}/{view}/spacy/train.spacy"]
    spacy_job["data/processed/{corpus_id}/{view}/spacy/job.spacy"]
    meta_formats["meta_formats.json\n(par famille)"]
  end

  subgraph TRAIN_CORE ["core_train.py"]
    train_select_models["Construction models_to_train\n(families + models_*)"]

    subgraph TRAIN_CHECK ["Famille 'check'"]
      check_in["train.tsv"]
      check_out["meta_model.json (stats)"]
    end

    subgraph TRAIN_SPACY ["Famille 'spacy'"]
      spacy_in["Préférence DocBin (train.spacy)\nFallback train.tsv"]
      spacy_train_loop["Training TextCat"]
      spacy_model_dir["models/{corpus_id}/{view}/spacy/{model_id}/..."]
    end

    subgraph TRAIN_SK ["Famille 'sklearn'"]
      sk_in["train.tsv"]
      sk_vec_clf["vectorizer + estimator"]
      sk_model_dir["models/{corpus_id}/{view}/sklearn/{model_id}/..."]
    end

    subgraph TRAIN_HF ["Famille 'hf' (stub)"]
      hf_in["train.tsv"]
      hf_stub["train_hf_model() TODO"]
      hf_model_dir["models/{corpus_id}/{view}/hf/{model_id}/..."]
    end
  end

  subgraph EVAL_CORE ["core_evaluate.py"]
    eval_load_models["Sélection models_to_eval"]
    eval_job_src["job.tsv (ou fallback train.tsv)"]
    eval_check["eval_check_model()"]
    eval_spacy["eval_spacy_model()"]
    eval_sk["eval_sklearn_model()"]
    eval_hf["eval_hf_model() stub"]
    metrics_json["metrics.json"]
    clf_report["classification_report.txt"]
    meta_eval["meta_eval.json"]
  end

  tei_xml --> prep_stream --> prep_extract --> prep_labelmap --> prep_filters --> prep_balance --> prep_split
  prep_split --> tsv_train
  prep_split --> tsv_job
  tsv_train --> spacy_train --> meta_formats
  tsv_job --> spacy_job --> meta_formats
  tsv_train --> meta_view
  tsv_job --> meta_view

  tsv_train --> check_in --> check_out
  tsv_train --> sk_in --> sk_vec_clf --> sk_model_dir
  spacy_train --> spacy_in --> spacy_train_loop --> spacy_model_dir
  tsv_train --> hf_in --> hf_stub --> hf_model_dir

  sk_model_dir --> eval_load_models
  spacy_model_dir --> eval_load_models
  hf_model_dir --> eval_load_models
  check_out --> eval_load_models
  tsv_job --> eval_job_src

  eval_job_src --> eval_check --> metrics_json
  eval_job_src --> eval_spacy --> metrics_json
  eval_job_src --> eval_sk --> metrics_json
  eval_job_src --> eval_hf --> metrics_json

  eval_check --> clf_report
  eval_spacy --> clf_report
  eval_sk --> clf_report
  eval_hf --> clf_report

  eval_check --> meta_eval
  eval_spacy --> meta_eval
  eval_sk --> meta_eval
  eval_hf --> meta_eval
```

### 5.2 Orchestration & pré-scripts

```mermaid
flowchart TB
  subgraph USER ["Dev / utilisateur"]
    user_shell["Terminal / IDE"]
  end

  subgraph ORCH ["Orchestration (optionnelle)"]
    make_v4["Makefile V4\nlist_profiles, check_profile,\nprepare, train, evaluate, pipeline"]
    cli_direct["CLI direct\npython scripts/core/*.py"]
  end

  subgraph PRE_SCRIPTS ["Pré-scripts (validation / labellisation)"]
    pre_check["scripts/pre/pre_check_config.py\nvalide profil complet"]
    ideo_skel["scripts/pre/make_ideology_skeleton.py\nTEI → squelette label_map\n+ actors_counts_*.tsv"]
  end

  subgraph CFG ["Configs & profils"]
    cfg_prof["configs/profiles/*.yml"]
    cfg_common["configs/common/*.yml"]
    cfg_labels["configs/label_maps/*.yml"]
  end

  subgraph CORE ["Core V4 (config-first)"]
    core_prepare["core_prepare.py"]
    core_train["core_train.py"]
    core_eval["core_evaluate.py"]
  end

  subgraph DATA ["Hiérarchie data/models/reports"]
    data_interim["data/interim/{corpus}/{view}/..."]
    data_proc["data/processed/{corpus}/{view}/..."]
    models_store["models/{corpus}/{view}/{family}/{model_id}/..."]
    reports_store["reports/{corpus}/{view}/{family}/{model_id}/..."]
  end

  user_shell --> ORCH
  ORCH --> PRE_SCRIPTS
  ORCH --> CORE
  PRE_SCRIPTS --> CFG
  CFG --> CORE
  CORE --> DATA
```

#### `hardware.yml`

* Déclare des presets machine.

Exemple :

```yaml
presets:
  small:
    desc: "Laptop / petite machine"
    ram_gb: 8
    max_procs: 2
    blas_threads: 1
    tsv_chunk_rows: 20000
    spacy_shard_docs: 10000

  medium:
    desc: "Machine intermédiaire"
    ram_gb: 16
    max_procs: 4
    blas_threads: 2
    tsv_chunk_rows: 50000
    spacy_shard_docs: 25000

  lab:
    desc: "Machine labo / serveur"
    ram_gb: 64
    max_procs: 8
    blas_threads: 2
    tsv_chunk_rows: 100000
    spacy_shard_docs: 50000
```

**Utilisation :**

* `hardware_preset` dans un profil → `params["hardware"]` (ram_gb, max_procs, etc.),
* BLAS threads fixés via `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`.

#### `models.yml`

* Registry complet des modèles, classés par famille :

  * `spacy`,
  * `sklearn`,
  * `hf`.

**Principe :**

> Ajouter un nouveau modèle = ajouter un bloc dans `models.yml`.
> **On ne modifie pas** `core_train.py` / `core_evaluate.py` pour ça.

Exemples (résumé) :

* spaCy : `spacy_cnn_quick`, `spacy_cnn_full`, `spacy_cnn_debug`.
* sklearn : `tfidf_svm_quick`, `tfidf_smo_linear`, `tfidf_smo_rbf`, `tfidf_perceptron`, `tfidf_randomtree`, `tfidf_randomforest`.
* HF : `camembert_base`, `flaubert_base_cased`, `bert_mbert_base`.

---

### 3.2 Label maps : `configs/label_maps/*.yml`

* Mapping des labels bruts → labels consolidés pour l’apprentissage.

Exemple typique `ideology_global.yml` :

```yaml
mapping:
  far_left: "gauche"
  left: "gauche"
  center_left: "centre"
  center: "centre"
  center_right: "centre"
  right: "droite"
  far_right: "droite"

unknown_labels:
  policy: "drop"     # ou "keep" ou "other"
  other_label: "autre"
```

#### Script de support : `scripts/pre/make_ideology_skeleton.py`

* But : extraire des entités/acteurs depuis le corpus TEI et produire un **squelette de mapping** à annoter à la main (droite/gauche/centre/…).

Utilisation via Makefile :

```bash
make ideology_skeleton \
  CORPUS_XML=data/raw/web1/corpus.xml \
  IDEO_MAP_OUT=configs/label_maps/ideology_actors.yml \
  IDEO_REPORT_OUT=data/configs/actors_counts_web1.tsv \
  MIN_CHARS_IDEO=200 \
  TOP_VARIANTS_IDEO=5
```

Flux :

1. Le script lit le XML, compte les acteurs, produit un YAML du type :

   ```yaml
   acteur_X: ""
   acteur_Y: ""
   ```

2. On complète à la main chaque valeur (`"gauche"`, `"droite"`, etc.).

3. Ce YAML peut ensuite être utilisé comme `label_map` dans un profil.

`core_utils.load_label_map()` gère deux formats :

* `{"mapping": {...}}`
* ou un dict plat `{clé: valeur}`.

---

### 3.3 Profils : `configs/profiles/*.yml`

Un **profil** décrit une expérience logique :

* quel corpus,
* quelle vue,
* quel champ de label,
* quelle stratégie d’équilibrage,
* quelles familles de modèles & IDs de modèles,
* quel preset hardware.

Ex. `ideo_quick.yml` :

```yaml
profile: "ideo_quick"
description: "Classification idéologie globale, pipeline rapide"

corpus_id: "web1"
view: "ideology_global"
modality: "web"

label_field: "ideology"
label_map: "configs/label_maps/ideology_global.yml"

train_prop: 0.8
min_chars: 280
max_tokens: 512

balance_strategy: "alpha_total"
balance_preset: "default_alpha_total"

families:
  - "check"
  - "spacy"
  - "sklearn"

models_spacy:
  - "spacy_cnn_quick"

models_sklearn:
  - "tfidf_svm_quick"

models_hf: []

hardware_preset: "small"

debug_mode: false
```

Autres profils déjà présents :

* `ideo_full.yml` : idéologie full (spaCy + sklearn + HF, hardware `lab`).
* `crawl_quick.yml` / `crawl_full.yml` : vue “crawl”.
* `check_only.yml` : ne fait que les checks/statistiques.
* `custom.yml` : profil neutre pour expérimenter.

### 3.4 Profil `custom` + overrides

* Le profil `custom.yml` sert de base.
* On peut le modifier **à la volée** via `OVERRIDES` :

```bash
make pipeline PROFILE=custom \
  OVERRIDES="corpus_id=web2 view=ideology_global hardware_preset=lab train_prop=0.7"
```

Techniquement :

* `core_utils.apply_overrides` prend des `key=value` et patch le dict `params`.
* Support de chemins imbriqués `a.b.c=value` si besoin.

---

## 4. Scripts core

### 4.1 `scripts/core/core_utils.py`

Rôle :

* charger les YAML communs (`common/`),
* charger le profil,
* produire un dict `params` unifié,
* gérer les overrides,
* charger les label_maps.

Fonction clé :

```python
params = resolve_profile_base(profile_name, overrides)
```

Sortie `params` contient :

* `profile`, `description`,
* `corpus` (dict tiré de `corpora.yml`),
* `corpus_id`, `view`, `modality`,
* `label_field`, `label_map`,
* `train_prop`, `min_chars`, `max_tokens`,
* `balance_strategy`, `balance_preset`,
* `families`,
* `models_spacy`, `models_sklearn`, `models_hf`, `models_check`,
* `hardware_preset`, `hardware`,
* `balance_cfg`, `hardware_cfg`, `models_cfg`.

### 4.2 `scripts/core/core_prepare.py`

#### 4.2.1 Rôle

1. **TEI → TSV** supervisé équilibré (`train.tsv`, `job.tsv`, `meta_view.json`).
2. **TSV → formats** pour chaque famille :

   * spaCy : `train.spacy`, `job.spacy` (DocBin),
   * autres familles : restent sur TSV pour V4-v1.

#### 4.2.2 CLI

```bash
python scripts/core/core_prepare.py \
  --profile ideo_quick \
  [--override key=value ...] \
  [--dry-run] \
  [--verbose]
```

#### 4.2.3 Étape 1 – Vue (TEI → TSV)

Points clefs :

* **Streaming TEI** avec `xml.etree.ElementTree.iterparse` :

  * on parcourt les `<text>` (ou autre unité, à adapter),
  * on n’a jamais tout le TEI en RAM.

* Extraction :

  * `doc_id` : via `xml:id` / `n` / fallback `doc_{i}`,
  * `text` : concaténation de `elem.itertext()`,
  * **modality** :

    * `extract_term(type="modality")`,
    * sinon `default_modality` du corpus,
    * sinon `"unknown"`,
  * **label brut** :

    * `extract_term(type=label_field)` (ex. `ideology`, `crawl`),
  * **label map** :

    * si `label_map` défini → application de `load_label_map`,
    * labels inconnus / vides → doc ignoré.

* Filtres :

  * `min_chars` : longueur minimale du texte,
  * `max_tokens` (naïf pour l’instant : `len(text.split()) > max_tokens` → drop),
  * `modality_filter` (= `params["modality"]`) :

    * si définie : on ne garde que les docs de cette modalité.

* Équilibrage :

  * collecter `docs` (liste de dicts) + `label_counts`,
  * appliquer `apply_balance` :

    * `none` : aucun changement,
    * `cap_docs` : nombre max de docs par label,
    * `alpha_total` :

      * V4-v1 : interpolation entre distribution uniforme et distribution actuelle (approx) → `target_per_label` docs,
      * **TODO** : si besoin de reproduire EXACTEMENT V2, remplacer par la logique historique,
    * `cap_tokens` : placeholder (“TODO : implémentation basée sur tokens”).

* Split train/job :

  * `train_prop` (ex. `0.8`),
  * seed : `params["seed"]` ou `42`,
  * random shuffle → split.

* Fichiers produits :

  * `data/interim/{corpus_id}/{view}/train.tsv`
  * `data/interim/{corpus_id}/{view}/job.tsv`
  * `data/interim/{corpus_id}/{view}/meta_view.json`

Colonnes TSV :

* `id`, `label`, `label_raw`, `modality`, `text`.

**Erreurs & contraintes :**

* Si aucun doc valide après filtrage → `SystemExit` avec message explicite.
* Si label_map invalide ou fichier manquant → `SystemExit` (via `load_label_map`).

#### 4.2.4 Étape 2 – Formats (TSV → formats)

Fonction :

```python
build_formats(params, meta_view)
```

Implémentation V4-v1 :

* spaCy :

  * lit `train.tsv` / `job.tsv`,
  * crée des `DocBin` :

    * `data/processed/{corpus_id}/{view}/spacy/train.spacy`,
    * `data/processed/{corpus_id}/{view}/spacy/job.spacy` (si job existe),
  * écrit `meta_spacy_formats.json` (lang, labels, counts, chemins).

* Textes/labels pour DocBin :

  * `doc = nlp.make_doc(text)`,
  * `doc.cats = {label: bool}` (multiclass exclusif).

* Autres familles :

  * référencent simplement les TSV dans `meta_formats.json` :

    ```json
    {
      "families": {
        "spacy": { ... },
        "sklearn": {
          "source": "tsv",
          "train_tsv": "...",
          "job_tsv": "..."
        },
        "hf": { ... },
        "check": { ... }
      }
    }
    ```

**TODO potentiel** :

* pour sklearn : précompiler `X_train.npz`, etc. (si besoin),
* pour HF : précompiler en `datasets.Dataset` (Arrow).

---

### 4.3 `scripts/core/core_train.py`

#### 4.3.1 Rôle

* Entraîner tous les modèles déclarés dans le profil (`families` + `models_*`),
* produire :

  * fichiers modèles persistants,
  * `meta_model.json`.

#### 4.3.2 CLI

```bash
python scripts/core/core_train.py \
  --profile ideo_quick \
  [--override key=value ...] \
  [--only-family spacy|sklearn|hf|check] \
  [--verbose]
```

#### 4.3.3 Logique générale

* `params = resolve_profile_base(...)`
* limiter les threads BLAS (`set_blas_threads`).
* construire `models_to_train` à partir de :

  * `families`,
  * `models_spacy`, `models_sklearn`, `models_hf`,
  * pseudo-modèle `check`.

##### Cas “check”

* `train_check_model` :

  * lit `train.tsv`,
  * calcule des stats basiques (répartition labels, etc.),
  * écrit `meta_model.json` comme pseudo-modèle.

##### Cas spaCy

* `train_spacy_model` :

  * cherche d’abord des DocBin :

    * `data/processed/{corpus_id}/{view}/spacy/train.spacy`,
  * sinon fallback TSV (`load_tsv_dataset`),
  * `debug_mode` → sous-échantillon à 1000 docs max,
  * construit un `nlp` blank(lang) + `textcat`,
  * ajoute les labels (`labels_set`),
  * boucle de training (epochs, dropout, minibatch),
  * sauvegarde dans `models/{corpus_id}/{view}/spacy/{model_id}/`,
  * écrit `meta_model.json`.

**Contraste avec V2 :**

* V2 utilisait spacy CLI/config, V4-v1 utilise une boucle manuelle simple.
* **TODO** : si tu veux du training spaCy “propre” via config `.cfg`, tu pourras intégrer `spacy train` ou l’API correspondante, pilotée par `models.yml`.

##### Cas sklearn

* `train_sklearn_model` :

  * lit `train.tsv` / `job.tsv` → `train_texts`, `train_labels`,
  * `debug_mode` → sous-échantillon,
  * vectoriser : `vectorizer = vect_class(**vect_params)`,
  * classifier : `estimator = est_class(**est_params)`,
  * ajuste `n_jobs` à `hardware.max_procs` pour les modèles qui le supportent (Perceptron, RandomForest),
  * sauvegarde `model.joblib` (`{vectorizer, estimator}`) dans `models/...`,
  * `meta_model.json` contient :

    * classes utilisées,
    * params,
    * n_features, n_train_docs.

##### Cas HF (TODO)

* `train_hf_model` : squelette avec message “non implémenté”.
* **À faire** plus tard :

  * lire `train.tsv` / `job.tsv`,
  * construire un Dataset HF (optionnellement sauvegardé),
  * initialiser `tokenizer` + `AutoModelForSequenceClassification`,
  * utiliser `Trainer` avec `trainer_params` issus de `models.yml`,
  * sauvegarder le modèle HF dans `models/...`.

---

### 4.4 `scripts/core/core_evaluate.py`

#### 4.4.1 Rôle

* Évaluer les modèles entraînés sur `job.tsv` (ou fallback `train.tsv`).
* Produire :

  * `metrics.json`,
  * `classification_report.txt`,
  * `meta_eval.json`.

#### 4.4.2 CLI

```bash
python scripts/core/core_evaluate.py \
  --profile ideo_quick \
  [--override key=value ...] \
  [--only-family spacy|sklearn|hf|check] \
  [--verbose]
```

#### 4.4.3 Logique générale

* `params = resolve_profile_base(...)`,
* fix BLAS threads,
* construire `models_to_eval`,
* pour chaque modèle :

##### spaCy

* `eval_spacy_model` :

  * charge `nlp` depuis `models/.../spacy/{model_id}/`,
  * lit `job.tsv` (ou train.tsv) → `texts`, `labels_true`,
  * `debug_mode` → sous-échantillon,
  * applique `nlp.pipe(texts)`,
  * pour chaque doc : choisit `best_label = argmax(doc.cats)`,
  * calcule :

    * `accuracy`,
    * `macro_f1`,
    * `classification_report` (dict sklearn),
  * sauvegarde `metrics.json`, `classification_report.txt`, `meta_eval.json`.

##### sklearn

* `eval_sklearn_model` :

  * charge `model.joblib`,
  * lit `job.tsv`,
  * transform `X = vectorizer.transform(texts)`,
  * `labels_pred = estimator.predict(X)`,
  * mêmes métriques que spaCy.

##### HF (TODO)

* `eval_hf_model` : squelette (écrit un meta minimal avec note “non implémenté”).

##### check

* `eval_check_model` :

  * lit `job.tsv`,
  * recalcule des stats brutes,
  * écrit `metrics.json` + `meta_eval.json`.

---

## 5. Orchestration : Makefile V4

Makefile (à la racine) :

* variables principales :

  * `PROFILE` (défaut : `ideo_quick`),
  * `OVERRIDES` : ex. `OVERRIDES="corpus_id=web2 train_prop=0.7"`,
  * `FAMILY` : pour limiter train/eval à une famille.

* cibles :

  * `make list_profiles`
    → liste les profils disponibles.
  * `make check_profile PROFILE=...`
    → appelle `pre_check_config.py` (validation profil + modèles + label_map).
  * `make prepare PROFILE=...`
    → `core_prepare` (TEI→TSV + formats).
  * `make prepare_dry`
    → `core_prepare` en `--dry-run`.
  * `make train PROFILE=... [FAMILY=...]`
  * `make evaluate PROFILE=... [FAMILY=...]`
  * `make pipeline PROFILE=...`
    → `prepare` + `train` + `evaluate`.
  * `make ideology_skeleton`
    → appelle `make_ideology_skeleton.py`.

**Compat Windows :**

* sous Windows sans `make`, il suffit de reprendre les commandes Python vues dans le Makefile :

  ```bash
  python scripts/core/core_prepare.py  --profile ideo_quick --verbose
  python scripts/core/core_train.py    --profile ideo_quick --verbose
  python scripts/core/core_evaluate.py --profile ideo_quick --verbose
  ```

---

## 6. Contraintes matérielles & limites techniques

### 6.1 RAM & fragmentation

* TEI :

  * **jamais** chargé intégralement en RAM,
  * `iterparse` par `<text>` (ou autre unité), `elem.clear()` pour relâcher la mémoire.

* Vue (TSV) V4-v1 :

  * on garde pour l’instant tous les docs retenus en RAM (liste Python) pour :

    * équilibrage,
    * split train/job.
  * **risque** : si le corpus final retenu (après filtre + mapping) est gigantesque, cela peut saturer la RAM.

**Évolution possible :**

* implémenter un équilibrage approximatif / progressif en streaming,
* ou un pipeline en 2 passes :

  1. statistiques sur les labels,
  2. écriture en streaming selon les quotas.

Pour l’instant, à utiliser **avec prudence** sur des corpus très volumineux (prévoir pré-filtrage, profils “quick”, etc.).

### 6.2 Threads & CPU

* `hardware.yml` fixe :

  * `max_procs` (n_jobs),
  * `blas_threads` (OMP, MKL, OPENBLAS),
  * chunk sizes pour TSV & DocBin.

* `core_train` / `core_evaluate` :

  * utilisent `set_blas_threads(n)` pour limiter les threads BLAS,
  * pour sklearn :

    * si un estimater a `n_jobs = -1` ou `None`, on le remplace par `hardware.max_procs`.

**À ne pas faire :**

* ne pas imposer `n_jobs=-1` en dur dans `models.yml` sans réfléchir aux presets hardware,
* ne pas multiplier les ensembles HF complets sur un laptop 8 Go.

### 6.3 Temps de calcul & HF (CPU only)

* HF (CamemBERT, FlauBERT, BERT) sur CPU :

  * **coût très élevé** pour de gros corpus,
  * à réserver aux profils `full` sur hardware `medium` / `lab`,
  * ou à des expérimentations ciblées (profil qui ne lance qu’un seul modèle HF).

---

## 7. Validation, erreurs & écueils

### 7.1 Script de pré-check : `pre_check_config.py`

* vérifie :

  * que le profil existe,
  * que `corpus_id` existe dans `corpora.yml`,
  * que tous les `models_*` existent dans `models.yml`,
  * que `label_map` est lisible et non vide,
  * que `hardware_preset` existe dans `hardware.yml`.

* usage :

  ```bash
  make check_profile PROFILE=ideo_full
  ```

### 7.2 Erreurs typiques

* **"corpus_id 'X' non défini..."**
  → `corpora.yml` ne déclare pas ce corpus.

* **"label_map introuvable..."**
  → chemin `label_map` incorrect dans le profil.

* **"Aucun document valide après filtrage" (core_prepare)**
  → combinaison de filtres trop agressive :

  * modalité,
  * min_chars,
  * max_tokens,
  * mapping de labels (labels inconnus → drop),
  * vérifie la couverture de ton `label_map`.

* **"train.tsv introuvable" (core_train)**
  → `core_prepare` n’a pas été exécuté (ou profil autre).

* **"Modèle sklearn/spaCy introuvable" (core_evaluate)**
  → `core_train` pas exécuté pour ce profil/famille/model_id.

### 7.3 Écueils à éviter

1. **Modifier les scripts core pour ajouter un modèle / corpus**
   → Tout changement doit passer par les YAML :

   * nouveaux modèles → `models.yml`,
   * nouveaux corpus → `corpora.yml`,
   * nouvelles expériences → `profiles/*.yml`.

2. **Multiplier les drapeaux CLI ad hoc**
   → les scripts core doivent rester stables :

   * `--profile`, `--override`, `--only-family`, `--dry-run`, `--verbose` c’est tout.

3. **Tuer la non-régression V1/V2 en simplifiant trop**
   → s’assurer que les profils V4 peuvent reproduire :

   * idéologie quick V2,
   * stratégies d’équilibrage critiques.

4. **Laisser les scripts core devenir des “god files” illisibles**
   → déjà mitigé par :

   * séparation families (spacy/sklearn/hf/check) en blocs,
   * fonctions de taille raisonnable (~30–80 lignes).

---

## 8. Ce qui est déjà fait vs ce qui reste à faire

### 8.1 Implémenté V4-v1

* **Config system** :

  * `configs/common/*`,
  * `profiles` ideo/crawl/custom/check,
  * label_maps + support `make_ideology_skeleton.py`.

* **Core** :

  * `core_utils` (resolve_profile_base, overrides, load_label_map),
  * `core_prepare` :

    * TEI → TSV équilibré,
    * TSV → DocBin spaCy + méta formats,
  * `core_train` :

    * famille `check`,
    * spaCy simple (DocBin ou fallback TSV),
    * sklearn,
    * stub HF,
  * `core_evaluate` :

    * famille `check`,
    * spaCy,
    * sklearn,
    * stub HF.

* **Orchestration** :

  * Makefile V4 (prepare / train / evaluate / pipeline, etc.),
  * compat Windows via duplication des commandes Python.

### 8.2 À faire / Roadmap technique

1. **Équilibrage V2 exact** :

   * remplacer `alpha_total` approximatif par la version historique si nécessaire,
   * implémenter réellement `cap_tokens`.

2. **HF training/eval** :

   * implémenter `train_hf_model` et `eval_hf_model` :

     * conversions TSV → `datasets.Dataset`,
     * initialisation AutoTokenizer/AutoModel,
     * `Trainer` + `trainer_params` de `models.yml`.

3. **Formats HF/sklearn (optionnel)** :

   * précompiler des matrices `X_train.npz` pour sklearn sur gros corpus,
   * précompiler des datasets Arrow pour HF.

4. **Scripts experiments & post** :

   * `scripts/experiments/run_grid.py` :

     * générer/soumettre des combos (profil × modèle × hyperparams),
     * limiter via `max_runs`,
   * `scripts/post/post_aggregate_metrics.py` :

     * consolider les `metrics.json` en un `summary.csv` global.

5. **Non-régression V1/V2** :

   * définir une check-list “features V1/V2 à conserver”,
   * écrire un script de comparaison (V2 vs V4) pour 1–2 expériences.

6. **Documentation utilisateur** :

   * doc séparée “user-level” (comment lancer des expériences),
   * doc “howto_experiments.md” pour utiliser `run_grid.py`.

---

## 9. Conclusion

V4 pose un **core solide** :

* 3 scripts génériques (`core_prepare`, `core_train`, `core_evaluate`),
* config-first (profils, modèles, corpus, hardware, équilibrage),
* compatibilité safe avec les limites matérielles (RAM, threads),
* séparation claire entre :

  * **labels** (idéologie/crawl),
  * **modalités** (web1/web2/asr/gold).

La priorité maintenant :

1. Stabiliser le comportement par rapport à V2 (expériences de référence).
2. Finaliser le support HF si tu en as besoin pour le mémoire/rapport.
3. Ajouter l’outillage d’expérimentation (grilles d’hyperparams, agrégation de métriques).
