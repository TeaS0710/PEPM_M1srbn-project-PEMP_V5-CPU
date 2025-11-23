Parfait, on fait un **deuxième doc** qui complète `dev_V4.md` avec toutes les “données indispensables” : référentiel de paramètres, héritage V1/V2, profils, etc.

Tu peux l’appeler par exemple : `docs/ref_V4_parameters.md`.

---

````markdown
# V4 – Référentiel des paramètres & héritage V1/V2

> Ce document complète `docs/dev_V4.md`.
> `dev_V4.md` explique **le pourquoi et le comment** (architecture, core, contraintes).
> `ref_V4_parameters.md` liste **tout ce qui est manipulé** (paramètres, profils, héritage V1/V2).

---

## 1. Vue d’ensemble

La V4 repose sur 4 couches :

1. **Core scripts** :
   `scripts/core/core_prepare.py`, `core_train.py`, `core_evaluate.py`
   → ne connaissent que des paramètres “purs Python” (`params`).

2. **Configs YAML** :
   `configs/common/*.yml`, `configs/profiles/*.yml`, `configs/label_maps/*.yml`
   → concentrent la combinatoire (corpus, modèles, hardware, équilibrage, vues).

3. **Makefile** :
   routeur fin entre CLI utilisateur et scripts core.

4. **Pré/Post/Experiments** :
   scripts autour du core (pré-check, génération de squelette de labels, agrégation de métriques, sweeps).

Ce document sert à :

- **lister tous les paramètres**, d’où ils viennent, et où ils sont utilisés ;
- **documenter les features héritées de V1/V2** (non-régression) ;
- **donner une vue claire des profils officiels**.

---

## 2. Paramètres CLI et variables d’environnement

### 2.1 Core scripts – paramètres CLI

#### `core_prepare.py`

```bash
python scripts/core/core_prepare.py \
  --profile NAME \
  [--override key=value ...] \
  [--dry-run] \
  [--verbose]
````

| Paramètre CLI | Type   | Défaut          | Rôle                                                              |
| ------------- | ------ | --------------- | ----------------------------------------------------------------- |
| `--profile`   | str    | — (obligatoire) | Nom du profil `configs/profiles/{profile}.yml`.                   |
| `--override`  | repeat | `[]`            | Overrides de clé de config : `clé=valeur`. Peut être répété.      |
| `--dry-run`   | bool   | False           | N’écrit pas les fichiers de sortie, ne fait que les stats / meta. |
| `--verbose`   | bool   | False           | Affiche les paramètres résolus et quelques logs supplémentaires.  |

#### `core_train.py`

```bash
python scripts/core/core_train.py \
  --profile NAME \
  [--override key=value ...] \
  [--only-family spacy|sklearn|hf|check] \
  [--verbose]
```

| Paramètre CLI   | Type   | Défaut | Rôle                                                                            |
| --------------- | ------ | ------ | ------------------------------------------------------------------------------- |
| `--profile`     | str    | —      | Profil utilisé.                                                                 |
| `--override`    | repeat | `[]`   | Overrides de config.                                                            |
| `--only-family` | str    | None   | Limiter l’entraînement à une seule famille (`spacy`, `sklearn`, `hf`, `check`). |
| `--verbose`     | bool   | False  | Logs détaillés.                                                                 |

#### `core_evaluate.py`

```bash
python scripts/core/core_evaluate.py \
  --profile NAME \
  [--override key=value ...] \
  [--only-family spacy|sklearn|hf|check] \
  [--verbose]
```

Même tableau que pour `core_train.py` (mêmes options).

---

### 2.2 Makefile – variables utilisateur

Dans le Makefile V4 :

| Variable Make       | Défaut                                   | Rôle                                                                      |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------------------- |
| `PYTHON`            | `python`                                 | Commande Python utilisée.                                                 |
| `PROFILE`           | `ideo_quick`                             | Profil à utiliser (`configs/profiles/PROFILE.yml`).                       |
| `OVERRIDES`         | `""`                                     | Liste de `clé=valeur` séparés par des espaces, convertis en `--override`. |
| `FAMILY`            | `""`                                     | Si non vide, passe `--only-family FAMILY` à train/evaluate.               |
| `CORPUS_XML`        | `data/raw/web1/corpus.xml`               | Source TEI pour `make ideology_skeleton`.                                 |
| `IDEO_MAP_OUT`      | `configs/label_maps/ideology_actors.yml` | YAML squelette généré.                                                    |
| `IDEO_REPORT_OUT`   | `data/configs/actors_counts_web1.tsv`    | Rapport d’acteurs.                                                        |
| `MIN_CHARS_IDEO`    | `200`                                    | Filtre min_chars pour le squelette d’idéologie.                           |
| `TOP_VARIANTS_IDEO` | `5`                                      | Nb de variantes les plus fréquentes par acteur.                           |

> **NB** : les valeurs du Makefile **ne changent pas** le comportement du core en lui-même, elles ne font que passer des arguments aux scripts.

---

### 2.3 Variables d’environnement (threads BLAS)

Les core scripts fixent :

* `OMP_NUM_THREADS`
* `MKL_NUM_THREADS`
* `OPENBLAS_NUM_THREADS`

à partir de `params["hardware"]["blas_threads"]`.

| Variable env           | Source       | Rôle                          |
| ---------------------- | ------------ | ----------------------------- |
| `OMP_NUM_THREADS`      | hardware.yml | Limiter les threads OpenMP.   |
| `MKL_NUM_THREADS`      | hardware.yml | Limiter les threads MKL.      |
| `OPENBLAS_NUM_THREADS` | hardware.yml | Limiter les threads OpenBLAS. |

---

## 3. Profils – paramètres de haut niveau

Un **profil** (fichier `configs/profiles/*.yml`) décrit une “expérience” :

* quel corpus / vue / modalité,
* quel champ de label,
* quelle stratégie d’équilibrage,
* quelles familles de modèles et quels IDs,
* quel preset hardware,
* options de debug.

### 3.1 Clés standard de profil

| Clé                | Type      | Obligatoire                  | Exemple / défaut                                      | Description                                                                                                             |
| ------------------ | --------- | ---------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `profile`          | str       | oui                          | `"ideo_quick"`                                        | Nom logique du profil.                                                                                                  |
| `description`      | str       | conseillé                    | `"Classification idéologie globale, pipeline rapide"` | Description humaine.                                                                                                    |
| `corpus_id`        | str       | oui                          | `"web1"`                                              | Doit exister dans `corpora.yml`.                                                                                        |
| `view`             | str       | oui                          | `"ideology_global"`                                   | Vue logique (idéologie, crawl, etc.). Sert à structurer `data/interim` et `data/processed`.                             |
| `modality`         | str       | non (mais recommandé)        | `"web"`                                               | Modalité filtrée sur le TEI (web, asr, gold…). Si absente, pas de filtre sur modality (ou fallback `default_modality`). |
| `label_field`      | str       | oui (hors mode actors)       | `"ideology"`                                          | Champ TEI utilisé pour extraire le label brut (type de `<term>` ou équivalent). Ignoré en `ideology.mode=actors`.       |
| `label_map`        | str       | conseillé (legacy/crawl)     | `"configs/label_maps/ideology_actors.yml"`           | Chemin de mapping. Par défaut le pipeline lit `ideology_actors.yml` ; les anciens YAML sont archivés dans `label_maps_legacy/`. |
| `train_prop`       | float     | oui                          | `0.8`                                                 | Proportion de docs dans le split d’entraînement.                                                                        |
| `min_chars`        | int       | conseillé                    | `280`                                                 | Longueur minimale en caractères pour garder un doc.                                                                     |
| `max_tokens`       | int       | conseillé                    | `512`                                                 | Longueur maximale approximative en tokens pour garder un doc.                                                           |
| `balance_strategy` | str       | oui                          | `"alpha_total"`                                       | Nom de la stratégie dans `balance.yml` (`none`, `alpha_total`, `cap_docs`, `cap_tokens`).                               |
| `balance_preset`   | str       | non (selon stratégie)        | `"default_alpha_total"`                               | Nom du preset dans la stratégie choisie (`presets`).                                                                    |
| `families`         | list[str] | oui                          | `["check", "spacy", "sklearn"]`                       | Familles de modèles à utiliser (`check`, `spacy`, `sklearn`, `hf`).                                                     |
| `models_spacy`     | list[str] | oui si `spacy` dans families | `["spacy_cnn_quick"]`                                 | IDs de modèles spaCy (doivent exister dans `models.yml`).                                                               |
| `models_sklearn`   | list[str] | oui si `sklearn`             | `["tfidf_svm_quick"]`                                 | IDs de modèles sklearn.                                                                                                 |
| `models_hf`        | list[str] | oui si `hf`                  | `["camembert_base"]`                                  | IDs de modèles HF. (Pour V4-v1, stub).                                                                                  |
| `hardware_preset`  | str       | oui                          | `"small"`                                             | Nom d’un preset dans `hardware.yml`.                                                                                    |
| `debug_mode`       | bool      | non                          | `false`                                               | Si vrai : limite le nombre de docs (prepare/train/evaluate).                                                            |
| `seed`             | int       | non                          | `42` (par défaut interne)                             | Seed pour le shuffle train/job. (À documenter dans le profil si important).                                             |

---

### 3.2 Profils officiels de base

#### 3.2.1 `ideo_quick.yml`

* But :
  *Pipeline rapide* pour l’idéologie globale sur `web1` ; sert surtout à tester et itérer vite.

* Caractéristiques :

  * `corpus_id: web1`
  * `view: ideology_global`
  * `balance_strategy: alpha_total`
  * `families: ["check", "spacy", "sklearn"]`
  * `models_spacy: ["spacy_cnn_quick"]`
  * `models_sklearn: ["tfidf_svm_quick"]`
  * `hardware_preset: small`
  * `debug_mode: false` (mais dataset limité par min_chars et max_tokens).

#### 3.2.2 `ideo_full.yml`

* But :
  Expérience “sérieuse” idéologie globale, avec plus de modèles et ressources.

* Caractéristiques typiques :

  * mêmes `corpus_id`/`view` que `ideo_quick`,
  * `families: ["check", "spacy", "sklearn", "hf"]`,
  * `models_spacy: ["spacy_cnn_full"]`,
  * `models_sklearn: ["tfidf_smo_linear", "tfidf_randomforest"]`,
  * `models_hf: ["camembert_base", "flaubert_base_cased"]`,
  * `hardware_preset: lab`.

#### 3.2.3 `crawl_quick.yml` / `crawl_full.yml`

* But :
  Expériences sur une vue “crawl” (par ex. classification URL/type, etc.), avec les mêmes principes quick/full.

* Différence principale :
  `view: crawl` ou `crawl_full`, `label_field` différent (`crawl_label` ou équivalent), `label_map` adapté.

#### 3.2.4 `check_only.yml`

* But :
  lancer uniquement la famille `check` pour vérifier la cohérence et la répartition labels.

* Caractéristiques :

  * `families: ["check"]`
  * pas de `models_spacy`/`models_sklearn`/`models_hf`.

#### 3.2.5 `custom.yml`

* But :
  profil neutre, que l’on modifie à la volée via `OVERRIDES` pour faire des expériences ad hoc.

* Utilisation typique :

```bash
make pipeline PROFILE=custom \
  OVERRIDES="corpus_id=web2 view=ideology_global hardware_preset=lab train_prop=0.7"
```

---

## 4. Configs communes – structure détaillée

### 4.1 `corpora.yml`

| Clé                | Type | Description                                              |
| ------------------ | ---- | -------------------------------------------------------- |
| `corpus_id`        | str  | Identifiant unique du corpus (ex. `web1`).               |
| `corpus_path`      | str  | Chemin TEI : ex. `data/raw/web1/corpus.xml`.             |
| `encoding`         | str  | Encodage du fichier (souvent `"utf-8"`).                 |
| `default_modality` | str  | Modalité à utiliser si aucune n’est trouvée dans le TEI. |
| `notes`            | str  | Commentaire libre.                                       |

Chaque entrée de `corpora.yml` doit être référencée par au moins un profil.

---

### 4.2 `balance.yml`

Structure :

```yaml
strategies:
  <strategy_name>:
    desc: "Description"
    presets:
      <preset_name>:
        ... paramètres spécifiques ...
```

Stratégies supportées en V4-v1 :

1. `none`
   → aucune modification de la distribution de labels.

2. `alpha_total`
   → essaie de répartir les docs par label autour d’un objectif `total_docs` et d’un mélange entre uniformité et distribution naturelle (`alpha`).

   Paramètres typiques :

   | Paramètre    | Description                                                                       |
   | ------------ | --------------------------------------------------------------------------------- |
   | `alpha`      | dans `[0,1]` : 0 = proche de l’uniforme, 1 = proche de la distribution d’origine. |
   | `total_docs` | nb cible de docs total après équilibrage.                                         |

3. `cap_docs`
   → limite le nombre de docs par label.

   | Paramètre       | Description            |
   | --------------- | ---------------------- |
   | `cap_per_label` | max de docs par label. |

4. `cap_tokens` (TODO / placeholder)
   → limite le nombre de tokens par label.

   | Paramètre              | Description              |
   | ---------------------- | ------------------------ |
   | `cap_tokens_per_label` | max de tokens par label. |

> **À faire** : si on veut une non-régression parfaite V2, il faudra recoder la logique exacte d’`alpha_total` V2 ici (ou dans `core_prepare`) et l’expliquer.

---

### 4.3 `hardware.yml`

Structure :

```yaml
presets:
  <preset_name>:
    desc: "Description"
    ram_gb: int
    max_procs: int
    blas_threads: int
    tsv_chunk_rows: int
    spacy_shard_docs: int
```

| Paramètre          | Rôle                                                                     |
| ------------------ | ------------------------------------------------------------------------ |
| `ram_gb`           | Information indicative pour calibrer certains caps (éventuellement).     |
| `max_procs`        | Limite pour `n_jobs` sklearn / `n_process` spacy / HF.                   |
| `blas_threads`     | Threads BLAS ; évite la sur-souscription.                                |
| `tsv_chunk_rows`   | Taille de chunk pour traiter les TSV par morceaux (future optimisation). |
| `spacy_shard_docs` | Nombre max de docs par shard DocBin (future optimisation).               |

---

### 4.4 `models.yml`

#### 4.4.1 Famille `spacy`

```yaml
families:
  spacy:
    <model_id>:
      desc: "..."
      arch: "cnn" | ...
      lang: "fr" | "en" | ...
      epochs: int
      dropout: float
      eval_frequency: int
      batch_start: int
      batch_stop: int
      config_template: "chemin/vers/config.cfg"
```

> **V4-v1** : `core_train` n’utilise pas encore `config_template` ni `arch`, mais ces champs sont déjà posés pour une future intégration avec la config spaCy.

#### 4.4.2 Famille `sklearn`

```yaml
families:
  sklearn:
    <model_id>:
      desc: "..."
      vectorizer:
        class: "sklearn.feature_extraction.text.TfidfVectorizer"
        params:
          ...
      estimator:
        class: "sklearn.svm.LinearSVC"
        params:
          ...
```

Remarques :

* `params.n_jobs` éventuellement mis à `-1` ou `None` → ajusté à `hardware.max_procs` par `core_train`.
* Types classiques :

  * `LinearSVC` (SVM linéaire),
  * `SVC` (SMO avec kernel RBF),
  * `Perceptron`,
  * `DecisionTreeClassifier`,
  * `RandomForestClassifier`, etc.

#### 4.4.3 Famille `hf` (Transformers)

```yaml
families:
  hf:
    <model_id>:
      desc: "..."
      model_name: "camembert-base" | "flaubert/flaubert_base_cased" | ...
      tokenizer_class: "transformers.AutoTokenizer"
      model_class: "transformers.AutoModelForSequenceClassification"
      trainer_params:
        learning_rate: float
        per_device_train_batch_size: int
        per_device_eval_batch_size: int
        num_train_epochs: int
        weight_decay: float
        warmup_ratio: float
        gradient_accumulation_steps: int
```

> **V4-v1** : `train_hf_model` / `eval_hf_model` sont encore des stubs ; mais l’interface config est prête.

---

### 4.5 `label_maps/*.yml`

Deux formats possibles :

1. **Dict plat** :

   ```yaml
   gauche: "gauche"
   centre_gauche: "gauche"
   droite: "droite"
   ```

2. **Dict avec `mapping` + comportements sur labels inconnus** :

   ```yaml
   mapping:
     far_left: "gauche"
     left: "gauche"
     center: "centre"
     right: "droite"
     far_right: "droite"

   unknown_labels:
     policy: "drop"  # ou "keep" ou "other"
     other_label: "autre"
   ```

`core_utils.load_label_map()` gère les deux formes.

---

## 5. Héritage V1/V2 – Check-list de non-régression

Cette section sert de **check-list** pour vérifier que la V4 conserve (ou dépasse) les features de V1/V2.

### 5.1 Tableau de mapping des features

| Feature V1/V2                             | V1            | V2      | Emplacement V4                     | Status V4                                          |
| ----------------------------------------- | ------------- | ------- | ---------------------------------- | -------------------------------------------------- |
| Extraction TEI → texte                    | ✔︎            | ✔︎      | `core_prepare` (TEI streaming)     | ✔︎ (simplifié, extensible)                         |
| Filtre `min_chars`                        | ✔︎            | ✔︎      | profil (`min_chars`)               | ✔︎                                                 |
| Filtre `max_tokens`                       | ✔︎            | ✔︎      | profil (`max_tokens`)              | ✔︎ (approx)                                        |
| Modalité via TEI (`term type="modality"`) | (±)           | ✔︎      | `core_prepare` + `modality` profil | ✔︎                                                 |
| Mapping de labels via YAML                | (doc interne) | ✔︎      | `label_map` + `label_maps/*.yml`   | ✔︎                                                 |
| Stratégie `alpha_total`                   | ✔︎            | ✔︎      | `balance.yml` + `core_prepare`     | ⚠︎ approximatif V4-v1 (TODO : logique exacte V2)   |
| `cap_docs` par label                      | ✔︎            | ✔︎      | `balance.yml` + `core_prepare`     | ✔︎                                                 |
| `cap_tokens` par label                    | (±)           | ✔︎      | `balance.yml` + `core_prepare`     | TODO (placeholder)                                 |
| Split train/job avec proportion & seed    | ✔︎            | ✔︎      | `train_prop` + seed (`params`)     | ✔︎ (seed explicite à stabiliser)                   |
| Profils séparés idéologie/crawl           | ✔︎            | ✔︎      | `configs/profiles/*.yml`           | ✔︎                                                 |
| Contrôle threads BLAS (`OMP_*`, etc.)     | ✔︎            | ✔︎      | `hardware.yml` + core              | ✔︎                                                 |
| `debug_mode` / runs rapides               | (±)           | ✔︎      | profil (`debug_mode`)              | ✔︎                                                 |
| spaCy TextCat training                    | ✔︎            | ✔︎      | `core_train.spacy`                 | ✔︎ (boucle manuelle, possibilité d’upgrade config) |
| sklearn baselines (TFIDF+SVM, etc.)       | ✘             | ✔︎      | `core_train.sklearn`               | ✔︎ (plus riche)                                    |
| HF baselines (CamemBERT, etc.)            | ✘             | ✔︎      | `models.yml.hf` + stubs HF         | partiel (interface ok, impl. à faire)              |
| Évaluation multi-familles (metrics)       | (spaCy only)  | ✔︎      | `core_evaluate`                    | ✔︎                                                 |
| Makefile routeur paramétrique             | ✘             | ✔︎      | Makefile V4                        | ✔︎ (plus simple, plus propre)                      |
| Doc interne très détaillée                | ✔︎            | (moins) | `dev_V4.md` + ce doc               | ✔︎ (structure différente, mais densité retrouvée)  |

### 5.2 Points à surveiller

* **alpha_total** :
  V4-v1 utilise une version simplifiée / approximative.
  → Pour une non-régression parfaite, il faudra soit :

  * reprendre le code V2 exact,
  * soit documenter clairement la différence et la justifier.

* **cap_tokens** :
  marquée TODO : à implémenter si tu as des expériences où cette contrainte compte.

* **HF training/eval** :
  stub pour l’instant, mais la structure est là ; à implémenter quand tu auras besoin des résultats HF.

---

## 6. Résumé conceptuel : V3 vs V4 idéale

### 6.1 Erreurs conceptuelles de V3 (telles que V4 les corrige)

* V3 a introduit une **modularisation de dossiers** (ingest/prepare/train/evaluate), mais :

  * le cœur prepare/train/evaluate est devenu plus difficile à lire,
  * certaines logiques de V1/V2 (équilibrage, stats) ont été perdues ou diluées,
  * le Makefile et les scripts se sont **désalignés** (API cachées, paramètres en dur).

Résultat : architecture visuellement “plus belle”, mais **moins réparable et moins transparente** scientifiquement.

### 6.2 Principes de la V4 idéale

La V4 telle que décrite dans `dev_V4.md` + ce référentiel vise :

* **Core minimal, générique, stable** :

  * 3 scripts : `core_prepare`, `core_train`, `core_evaluate`,
  * quelques paramètres CLI stables,
  * pas de logique scotchée au Makefile.

* **Tout le métier dans les configs** :

  * modèles, corpus, vues, équilibrage, hardware → YAML,
  * ajout d’un modèle/corpus/vue = ajout/modif YAML (0 modification Python).

* **Orchestration simple** :

  * Makefile = routeur + ergonomie (pas un moteur parallèle),
  * facile à répliquer sous Windows sans make.

* **Lisibilité & rigueur** :

  * doc de dev (`dev_V4.md`) = big picture + plan de construction,
  * ce référentiel (`ref_V4_parameters.md`) = inventaire des paramètres & héritage V1/V2,
  * meta fichiers (`meta_view`, `meta_model`, `meta_eval`) pour la reproductibilité.

---

## 7. Check final : comment utiliser ce doc

Pour toi (ou un·e dev dans 6 mois), ce document sert à :

1. **Vérifier un profil** :

   * Est-ce que toutes les clés de profil sont cohérentes ?
   * Est-ce que les `families` + `models_*` existent dans `models.yml` ?
   * Est-ce que le `corpus_id` existe dans `corpora.yml` et que `label_map` pointe vers un YAML valide ?

2. **Ajouter un modèle ou une famille** :

   * Ajouter uniquement un bloc dans `models.yml` (et implémenter la famille dans `core_train`/`core_evaluate` si nouvelle).
   * Documenter les paramètres dans la structure `models.yml`.

3. **Contrôler la non-régression V1/V2** :

   * Utiliser la table 5.1 comme check-list.
   * Marquer les TODO (alpha_total exact, cap_tokens, HF complet).

4. **Auditer un run** :

   * Savoir quels paramètres étaient actifs pour un run (profil + overrides),
   * retrouver dans `meta_view.json`, `meta_model.json`, `meta_eval.json` :

     * `profile`, `corpus_id`, `view`, `family`, `model_id`,
     * `pipeline_version`,
     * hardware, balance, splits, etc.

En combinant `dev_V4.md` (concept & plan) + `ref_V4_parameters.md` (inventaire & héritage), tu as un duo de docs qui :

* décrit **ce que V4 est**,
* décrit **comment V4 doit être utilisée**,
* décrit **ce qu’elle ne doit pas perdre de V1/V2**,
* et te donne un cadre clair pour continuer à l’étendre sans la casser.

## 5. Héritage V1/V2 – Check-list de non-régression

Cette section sert de **check-list** pour vérifier que la V4 conserve (ou dépasse) les features de V1/V2.

### 5.1 Tableau de mapping des features

| Feature V1/V2                             | V1 | V2 | Emplacement V4               | Status V4 |
|------------------------------------------|----|----|------------------------------|-----------|
| Extraction TEI → texte                   | ✔︎  | ✔︎  | `core_prepare` (TEI streaming) | ✔︎ (simplifié, extensible) |
| Filtre `min_chars`                       | ✔︎  | ✔︎  | profil (`min_chars`)         | ✔︎ |
| Filtre `max_tokens`                      | ✔︎  | ✔︎  | profil (`max_tokens`)        | ✔︎ (approx) |
| Modalité via TEI (`term type="modality"`) | (±) | ✔︎ | `core_prepare` + `modality` profil | ✔︎ |
| Mapping de labels via YAML               | (doc interne) | ✔︎ | `label_map` + `label_maps/*.yml` | ✔︎ |
| Stratégie `alpha_total`                  | ✔︎  | ✔︎  | `balance.yml` + `core_prepare` | ⚠︎ approximatif V4-v1 (TODO : logique exacte V2) |
| `cap_docs` par label                     | ✔︎  | ✔︎  | `balance.yml` + `core_prepare` | ✔︎ |
| `cap_tokens` par label                   | (±) | ✔︎  | `balance.yml` + `core_prepare` | TODO (placeholder) |
| Split train/job avec proportion & seed   | ✔︎  | ✔︎  | `train_prop` + seed (`params`) | ✔︎ (seed explicite à stabiliser) |
| Profils séparés idéologie/crawl          | ✔︎  | ✔︎  | `configs/profiles/*.yml`     | ✔︎ |
| Contrôle threads BLAS (`OMP_*`, etc.)    | ✔︎  | ✔︎  | `hardware.yml` + core        | ✔︎ |
| `debug_mode` / runs rapides              | (±) | ✔︎  | profil (`debug_mode`)        | ✔︎ |
| spaCy TextCat training                   | ✔︎  | ✔︎  | `core_train.spacy`           | ✔︎ (boucle manuelle, possibilité d’upgrade config) |
| sklearn baselines (TFIDF+SVM, etc.)      | ✘  | ✔︎  | `core_train.sklearn`         | ✔︎ (plus riche) |
| HF baselines (CamemBERT, etc.)           | ✘  | ✔︎  | `models.yml.hf` + stubs HF   | partiel (interface ok, impl. à faire) |
| Évaluation multi-familles (metrics)      | (spaCy only) | ✔︎ | `core_evaluate`             | ✔︎ |
| Makefile routeur paramétrique            | ✘  | ✔︎  | Makefile V4                  | ✔︎ (plus simple, plus propre) |
| Doc interne très détaillée               | ✔︎  | (moins) | `dev_V4.md` + ce doc      | ✔︎ (structure différente, mais densité retrouvée) |

### 5.2 Points à surveiller

- **alpha_total** :
  V4-v1 utilise une version simplifiée / approximative.
  → Pour une non-régression parfaite, il faudra soit :
  - reprendre le code V2 exact,
  - soit documenter clairement la différence et la justifier.

- **cap_tokens** :
  Marquée TODO : à implémenter si tu as des expériences où cette contrainte compte.

- **HF training/eval** :
  Stub pour l’instant, mais la structure est là ; à implémenter quand tu auras besoin des résultats HF.

---

## 6. Résumé conceptuel : V3 vs V4 idéale

### 6.1 Erreurs conceptuelles de V3 (telles que V4 les corrige)

V3 a introduit une **modularisation par dossiers** (`ingest/prepare/train/evaluate`), mais :

- le cœur prepare/train/evaluate est devenu plus difficile à lire,
- certaines logiques de V1/V2 (équilibrage, stats) ont été perdues ou diluées,
- le Makefile et les scripts se sont **désalignés** (API cachées, paramètres en dur).

Résultat : architecture visuellement “plus belle”, mais **moins réparable et moins transparente** scientifiquement.

### 6.2 Principes de la V4 idéale

La V4 telle que décrite dans `dev_V4.md` + ce référentiel vise :

- **Core minimal, générique, stable** :
  - 3 scripts : `core_prepare`, `core_train`, `core_evaluate`,
  - quelques paramètres CLI stables,
  - pas de logique scotchée au Makefile.

- **Tout le métier dans les configs** :
  - modèles, corpus, vues, équilibrage, hardware → YAML,
  - ajout d’un modèle/corpus/vue = ajout/modif YAML (0 modification Python).

- **Orchestration simple** :
  - Makefile = routeur + ergonomie (pas un moteur parallèle),
  - facile à répliquer sous Windows sans `make`.

- **Lisibilité & rigueur** :
  - doc de dev (`dev_V4.md`) = big picture + plan de construction,
  - ce référentiel (`ref_V4_parameters.md`) = inventaire des paramètres & héritage V1/V2,
  - meta fichiers (`meta_view`, `meta_model`, `meta_eval`) pour la reproductibilité.

---

## 7. Check final : comment utiliser ce doc

Pour toi (ou un·e dev dans 6 mois), ce document sert à :

1. **Vérifier un profil**
   - Est-ce que toutes les clés de profil sont cohérentes ?
   - Est-ce que les `families` + `models_*` existent dans `models.yml` ?
   - Est-ce que le `corpus_id` existe dans `corpora.yml` et que `label_map` pointe vers un YAML valide ?

2. **Ajouter un modèle ou une famille**
   - Ajouter uniquement un bloc dans `models.yml` (et implémenter la famille dans `core_train`/`core_evaluate` si nouvelle).
   - Documenter les paramètres dans la structure `models.yml`.

3. **Contrôler la non-régression V1/V2**
   - Utiliser la table 5.1 comme check-list.
   - Marquer les TODO (alpha_total exact, cap_tokens, HF complet).

4. **Auditer un run**
   - Savoir quels paramètres étaient actifs pour un run (profil + overrides).
   - Retrouver dans `meta_view.json`, `meta_model.json`, `meta_eval.json` :
     - `profile`, `corpus_id`, `view`, `family`, `model_id`,
     - `pipeline_version`,
     - hardware, balance, splits, etc.

---

## 8. Conclusion

En combinant :

- `dev_V4.md` (conception, architecture, plan de dev), et
- `ref_V4_parameters.md` (inventaire des paramètres, profils, héritage V1/V2),

tu as un **duo de docs** qui te permet :

- de comprendre **ce qu’est** la V4 et pourquoi elle est structurée ainsi,
- de voir **tout ce qu’elle manipule** (paramètres, configs, profils),
- de vérifier que tu ne **régresse pas** par rapport à V1/V2,
- et de continuer à l’étendre (nouveaux modèles, nouveaux corpus, nouvelles vues)
  **sans toucher au core** et sans te recréer une V2 bis ou une V3 bancale.

