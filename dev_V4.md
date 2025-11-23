# dev_V4 – Documentation de développement du core (V4)

> **But du doc**
> Expliquer *pour un dev* comment fonctionne le cœur V4 (`core_prepare`, `core_train`, `core_evaluate`), comment il dialogue avec les configs YAML et le Makefile, et comment l’étendre sans tout casser.

Ce document complète **`README.md`** (usage utilisateur) et **`ref_V4_parameters.md`** (référentiel de paramètres / héritage V1–V3).

---

## 1. Objectifs & philosophie de la V4

La V4 vise à corriger les dérives des versions précédentes :

* cœur Python devenu illisible,
* logique éparpillée entre scripts et Makefile,
* régression par rapport à V1/V2 sur l’équilibrage / stats.

Principes :

* **Core minimal & générique**

  * 3 scripts : `core_prepare.py`, `core_train.py`, `core_evaluate.py`;
  * interface CLI stable (profil + overrides).
* **Tout le métier dans les configs**

  * corpora, vues, modèles, hardware, équilibrage, idéologie → YAML :
    `configs/common/*.yml`, `configs/profiles/*.yml`, `configs/label_maps/*.yml`.
* **Multi-* par construction** :

  * multi-corpus (`corpora.yml`),
  * multi-vues (ideology_global, left/right intra, etc.),
  * multi-familles de modèles (`check`, `spacy`, `sklearn`, `hf`),
  * multi-méthodes (plusieurs modèles par famille dans `models.yml`).
* **Orchestration simple**

  * Makefile = routeur ergonomique, pas de logique métier cachée.

---

## 2. Vue d’ensemble de l’architecture

### 2.1 Pipeline global

```mermaid
flowchart LR
    subgraph RAW["data/raw/<corpus_id>/"]
        A[corpus.xml (TEI)]
    end

    subgraph PREPARE["core_prepare.py"]
        B[Extraction TEI → docs bruts]
        C[Filtrage / label mapping]
        D[Équilibrage (balance.yml)]
        E[Split train / job]
        F[TSV + formats (spacy/sklearn/hf)]
    end

    subgraph INTERIM["data/interim/<corpus>/<view>/"]
        T1[train.tsv]
        T2[job.tsv]
    end

    subgraph PROCESSED["data/processed/<corpus>/<view>/"]
        S1[spacy/train/*.spacy]
        S2[spacy/job/*.spacy]
        K1[sklearn/train.tsv (lien)]
        H1[hf/train.jsonl (optionnel)]
        MFORM[meta_formats.json]
    end

    subgraph MODELS["models/<corpus>/<view>/"]
        MCHECK[check/.../meta_model.json]
        MSPA[spacy/<model_id>/model-best]
        MSKL[sklearn/<model_id>/*.joblib]
        MHF[hf/<model_id>/*.bin]
    end

    subgraph REPORTS["reports/<corpus>/<view>/"]
        R1[metrics.json]
        R2[classification_report.txt]
        R3[meta_eval.json]
    end

    A -->|TEI parsing + label_maps| B --> C --> D --> E
    E -->|écrit| T1 & T2
    E --> F --> S1 & S2 & MFORM

    T1 & S1 -->|core_train| MODELS
    T2 & MODELS -->|core_evaluate| REPORTS
```

---

## 3. Organisation du repo (vue dev)

Les chemins importants (niveau dev) :

* `configs/common/`

  * `corpora.yml` : définition des corpus (`id`, chemin TEI, langue, etc.).
  * `hardware.yml` : presets hardware (`small`, `medium`, `lab`, …).
  * `balance.yml` : stratégies d’équilibrage.
  * `models.yml` : inventaire des modèles par famille (`check`, `spacy`, `sklearn`, `hf`).
* `configs/label_maps/`

  * `ideology.yml` : référentiel **conceptuel** de l’idéologie (global + intra).
  * `ideology_actors.yml` : mapping détaillé acteurs → catégories, dérivé.
  * `ideology_global.yml`, `ideology_left_intra.yml`, `ideology_right_intra.yml` : vues dérivées.
* `configs/profiles/`

  * `ideo_quick.yml`, `ideo_full.yml`, `crawl_*`, `check_only.yml`, `custom.yml` : profils haut niveau.
* `scripts/core/`

  * `core_prepare.py` : TEI → TSV + formats.
  * `core_train.py` : entraînement des familles.
  * `core_evaluate.py` : évaluation + rapports.
  * `core_utils.py` : résolution de profil, logging, seeds, etc.
* `scripts/pre/`

  * `pre_check_env.py`, `pre_check_config.py` : diagnostics & validation de profil.
  * `make_ideology_skeleton.py` : scan TEI → squelette d’acteurs.
  * `derive_ideology_from_yaml.py` : fusion `ideology.yml` + squelette → `ideology_actors.yml` + vues dérivées.

---

## 4. Résolution des paramètres & profil

### 4.1 Profil + overrides

L’entrée standard est :

```bash
python scripts/core/core_prepare.py --profile ideo_quick --override key=val ...
python scripts/core/core_train.py   --profile ideo_quick ...
python scripts/core/core_evaluate.py --profile ideo_quick ...
```

Le Makefile te fournit un alias :

```bash
make run STAGE=pipeline PROFILE=ideo_quick
```

Tout passe par :

```python
from scripts.core.core_utils import resolve_profile_base
```

Pipeline interne (`resolve_profile_base`) :

1. Chargement du profil YAML : `configs/profiles/<profile>.yml`.
2. Merge avec configs communes :

   * `corpora.yml` → résolution de `corpus_id` et du chemin TEI,
   * `hardware.yml` → preset matériel,
   * `balance.yml` → presets d’équilibrage.
3. Application des overrides CLI (`--override key=val`) :

   * propagate `corpus_id`, `train_prop`, `balance_strategy`, `balance_preset`, `hardware_preset`, `families`, `seed`, `max_train_docs_*`…
4. Alias convivials :

   * `balance_mode=weights` → `balance_strategy=class_weights`,
   * `balance_mode=oversample` → `balance_strategy=cap_docs`.
5. Ajout de meta :

   * `pipeline_version`, `profile`, `view`, etc.

Tous les scripts core récupèrent **exactement** la même structure `params` ensuite.

---

## 5. `core_prepare.py` – TEI → TSV + formats

### 5.1 Rôle

* Parse le TEI (`corpus.xml`),
* applique le mapping idéologie (via `label_map`),
* filtre / nettoie les docs,
* applique l’équilibrage (`balance_strategy` + `balance_preset`),
* split en `train` / `job`,
* sérialise :

  * `data/interim/<corpus>/<view>/train.tsv`, `job.tsv`,
  * formats spacy/sklearn/hf dans `data/processed/<corpus>/<view>/`,
  * `meta_view.json` + `meta_formats.json`.

### 5.2 Étapes principales

1. **Résolution des params** via `resolve_profile_base`.
2. **Lecture du corpus** TEI :

   * streaming des `<doc>` / `<text>` (implémentation dans `core_prepare.py`),
   * extraction de :

     * `id` (identifiant doc),
     * `text` (texte brut),
     * champs meta (site, date, etc.),
     * tags de vue (ex. champ d’idéologie utilisé par la `view`).
3. **Label mapping** :

   * charge `configs/label_maps/ideology_actors.yml` *et/ou* `ideology_global.yml` selon la vue.
   * applique la résolution auteur/domaine → label global (droite/gauche, etc.).
4. **Filtrage** :

   * min. longueur (ex `min_chars`),
   * suppression des docs sans label ou hors vue active.
5. **Équilibrage** (optionnel) via `apply_balance` :

   * `balance_strategy` ∈ {`none`, `cap_docs`, `cap_tokens`, `alpha_total`, `class_weights`},
   * presets dans `balance.yml` → `balance_preset`,
   * `alpha_total` = oversampling proportionnel / log-lissé (implémenté, refinable),
   * pour `class_weights`, les poids sont calculés et stockés dans `params["class_weights"]` pour les modèles (sklearn).
6. **Split train / job** :

   * proportion `train_prop` (par défaut 0.7 / 0.8 selon profil),
   * split stratifié (par label) si possible,
   * écrit `train.tsv` et `job.tsv` dans `data/interim/<corpus>/<view>/`.
7. **Formats modèles** :

   * construit le répertoire `data/processed/<corpus>/<view>/`,
   * fabrique les formats pour chaque famille déclarée dans le profil :

     * **spaCy** : DocBin shardés,
     * **sklearn** : TSV utilisé tel quel,
     * **hf** : JSONL / TSV convertible (stub prêt pour plus tard).
8. **Meta** :

   * `meta_view.json` : infos sur vue, distribution label, split, équilibrage,
   * `meta_formats.json` : où sont les formats par famille (chemins vers shards `.spacy`, etc.).

### 5.3 Formats spaCy : DocBin shardés (fix E870)

Pour éviter l’erreur spaCy `[E870] DocBin too large`, le pipeline ne crée **jamais** un seul `train.spacy` géant. Au lieu de ça :

* `core_prepare` produit une **arbo de shards** :

```bash
data/processed/<corpus>/<view>/spacy/
  train/
    part-00001.spacy
    part-00002.spacy
    ...
  job/
    part-00001.spacy
    ...
meta_formats.json   # liste des shards
```

Les paramètres hardware peuvent contrôler la taille cible des shards (`max_docs_spacy`, etc.) via `hardware.yml` + overrides.

---

## 6. `core_train.py` – entraînement des familles

### 6.1 Rôle

* Résoudre le profil + hardware,
* fixer éventuellement une seed globale (`apply_global_seed`),
* pour chaque famille active (`families`) :

  * entraîner chaque `model_id` déclaré dans `models.yml`,
  * sérialiser le modèle dans `models/<corpus>/<view>/<family>/<model_id>/`,
  * écrire un `meta_model.json` par modèle.

### 6.2 Boucle principale

Pseudocode simplifié de `main()` :

```python
args = parse_args()
params = resolve_profile_base(args.profile, args.override)

if args.verbose:
    debug_print_params(params)

seed_applied = apply_global_seed(params.get("seed"))
log("train", "seed", ...)

hw = params.get("hardware", {})
set_blas_threads(hw.get("blas_threads", 1))

families = params.get("families", []) or []
if args.only_family and args.only_family in families:
    families = [args.only_family]

models_to_train = []
# check
if "check" in families:
    models_to_train.append({"family": "check", "model_id": "check_default"})
# spacy
if "spacy" in families:
    for mid in params.get("models_spacy", []) or []:
        models_to_train.append({"family": "spacy", "model_id": mid})
# sklearn
if "sklearn" in families:
    for mid in params.get("models_sklearn", []) or []:
        models_to_train.append({"family": "sklearn", "model_id": mid})
# hf
if "hf" in families:
    for mid in params.get("models_hf", []) or []:
        models_to_train.append({"family": "hf", "model_id": mid})

for m in models_to_train:
    # dispatch vers train_spacy_model / train_sklearn_model / ...
```

### 6.3 Famille `check`

* Vue comme un **pseudo-modèle** :

  * re-parcourt le jeu d’entraînement (`train.tsv`),
  * calcule des stats de base (distribution de labels, nb docs),
  * écrit seulement `meta_model.json` (pas de modèle entraîné).

Utile pour :

* s’assurer que la vue et l’équilibrage sont corrects,
* avoir un log structuré dans `models/<corpus>/<view>/check/check_default/meta_model.json`.

### 6.4 Famille `spacy`

Entrée :

* shards DocBin produits par `core_prepare` (`spacy/train/*.spacy`, `spacy/job/*.spacy`),
* config spaCy dans `configs/spacy/*.cfg`,
* hyper-params & `model_id` dans `models.yml` (`models_spacy`).

Points clés :

* le cœur ne fusionne plus tous les DocBin en un seul :

  * il lit les shards **un par un**,
  * spaCy voit un corpus “virtuel” composé de tous les shards,
  * plus d’erreur `bytes object is too large`.
* meta :

  * `meta_model.json` décrit `model_id`, corpus, vue, nb docs, version spaCy, etc.

### 6.5 Famille `sklearn`

Entrée :

* `train.tsv` / `job.tsv` (même format que pour `check`),
* hyper-params des modèles TF-IDF / linéaire / arbres dans `models.yml`
  (`models_sklearn`).

Actuellement, la famille peut inclure (selon `models.yml`) :

* SVM linéaire / RBF (`tfidf_svm_quick`, etc.),
* Perceptron,
* Arbres de décision,
* Forêts aléatoires,
* (d’autres variantes faciles à ajouter).

Les **class weights** peuvent être auto-déduits à partir du profil d’équilibrage si `balance_strategy=class_weights`. Dans ce cas, `apply_balance` stocke les poids dans `params["class_weights"]`, à utiliser dans les modèles sklearn.

### 6.6 Famille `hf`

Actuellement :

* stub de base (structure dispatch / meta prête),
* l’entraînement complet reste un TODO (prévu V4.x).

---

## 7. `core_evaluate.py` – évaluation & rapports

### 7.1 Rôle

* Charger les modèles entraînés pour chaque famille / `model_id`,
* évaluer sur `job.tsv` (et shards spacy/hf),
* calculer les métriques,
* produire des rapports pour `reports/<corpus>/<view>/`.

### 7.2 Boucle principale

Le schéma est parallèle à `core_train` : même logique `families` / `models_*`, mais côté évaluation.

Pour chaque `(family, model_id)` :

* récupérer les données d’éval (texte + labels vrais),
* appliquer le modèle,
* calculer :

  * précision global, macro/micro-F1, etc.,
  * éventuellement matrice de confusion (selon implémentation),
* appeler `save_eval_outputs` qui écrit :

  * `metrics.json`,
  * `classification_report.txt`,
  * `meta_eval.json` (infos run + références vers le modèle).

### 7.3 Famille `check`

Ré-utilise le pseudo-modèle de stats : pas de prédictions, juste un résumé structuré du jeu d’éval (taille, distribution labels) dans `metrics.json`.

---

## 8. Idéologie : de `ideology.yml` → acteurs & vues dérivées

### 8.1 Principe général

L’idée :

1. Tu **annotes une seule fois** la structure conceptuelle dans `configs/label_maps/ideology.yml` :

   * catégories globales (ex : far_right, right, center, left, far_left),
   * clusters intra (droite / gauche),
   * éventuellement tags d’acteurs / domaines / sites.
2. Le pipeline construit un **squelette d’acteurs** à partir du corpus :

   * `make ideology_skeleton` → `scripts/pre/make_ideology_skeleton.py`,
   * produit `configs/label_maps/ideology_actors.yml` + `data/configs/actors_counts_<corpus>.tsv`.
3. Tu complètes/raffines `ideology.yml` (et/ou `ideology_actors.yml`),
4. Tu relances :

```bash
make ideology_from_yaml
# ou
make ideology_all   # skeleton + from_yaml
```

Ce qui appelle `derive_ideology_from_yaml.py` et régénère :

* `configs/label_maps/ideology_actors.yml` (acteurs enrichis),
* `configs/label_maps/ideology_global.yml`,
* `configs/label_maps/ideology_left_intra.yml`,
* `configs/label_maps/ideology_right_intra.yml`.

### 8.2 Pipeline idéologie (mermaid)

```mermaid
flowchart LR
    A[corpus.xml] --> B[make_ideology_skeleton.py]
    B --> C[ideology_actors.yml (squelette)]
    D[ideology.yml (référentiel)] --> E[derive_ideology_from_yaml.py]
    C --> E
    E --> F[ideology_actors.yml final]
    E --> G[ideology_global.yml]
    E --> H[ideology_left_intra.yml]
    E --> I[ideology_right_intra.yml]
```

Ensuite, les profils (`ideo_quick.yml`, `ideo_full.yml`) référencent simplement :

* `label_map: configs/label_maps/ideology_global.yml` (ou autre vue),
* `view: ideology_global` (ou `left_intra`, `right_intra`, …).

---

## 9. Profils & configs : comment tout se connecte

Pour tout le détail des clés de profil et des paramètres, voir `ref_V4_parameters.md`.
Ici on résume le wiring.

### 9.1 Profils (`configs/profiles/*.yml`)

Un profil typique (ex `ideo_quick.yml`) contient :

* `profile`: nom (`ideo_quick`),
* `description`: texte humain,
* `corpus_id`: ex `web1`,
* `view`: ex `ideology_global`,
* `families`: ex `["check", "spacy", "sklearn", "hf"]`,
* `models_spacy`: liste d’identifiants présents dans `models.yml`,
* `models_sklearn`: idem,
* `models_hf`: idem,
* `hardware_preset`: ex `small` / `medium` / `lab`,
* `train_prop`, `balance_strategy`, `balance_preset`, etc.

### 9.2 `corpora.yml`

Pour chaque `corpus_id` (ex `web1`) :

* `path`: `data/raw/web1/corpus.xml`,
* optionnel : `language`, `description`, etc.
  Le core ne suppose rien d’idéologique ici, juste une source TEI.

### 9.3 `hardware.yml`

Définit des presets comme :

* `small`: peu de threads BLAS, limites strictes `max_train_docs_*`,
* `lab`: plus de threads, plus de docs, etc.

`resolve_profile_base` applique le preset puis recopie les overrides top-level (`max_train_docs_spacy`, etc.) dans `params["hardware"]`.

### 9.4 `balance.yml`

Décrit :

* stratégies disponibles (`none`, `cap_docs`, `cap_tokens`, `alpha_total`, `class_weights`),
* presets nommés (`balance_preset`) avec :

  * per-label caps, oversample, offset,
  * `alpha`, `total_docs`, etc.

---

## 10. Méta-fichiers & reproductibilité

Chaque stage écrit ses méta-infos :

* `core_prepare` :

  * `data/interim/<corpus>/<view>/meta_view.json` :

    * profil, vue, corpus, splits, stats de labels, équilibrage, etc.
  * `data/processed/<corpus>/<view>/meta_formats.json` :

    * pour chaque famille, chemins vers formats (shards `.spacy`, etc.).
* `core_train` :

  * `models/<corpus>/<view>/<family>/<model_id>/meta_model.json` :

    * profil, vue, corpus, hyper-params, nb docs, seed, version du pipeline, etc.
* `core_evaluate` :

  * `reports/<corpus>/<view>/metrics.json`,
  * `reports/<corpus>/<view>/classification_report.txt`,
  * `reports/<corpus>/<view>/meta_eval.json` (profil, modèle, hardware, etc.).

Avec ces trois couches, tu peux toujours reconstruire :

* **quoi** a été entraîné/évalué,
* avec **quels** paramètres,
* sur **quel** corpus/vue,
* et **quand** (timestamp / version).

---

## 11. Recettes de dev

### 11.1 Ajouter un nouveau modèle sklearn

1. Ajouter un bloc dans `configs/common/models.yml` sous `sklearn`.
2. Référencer le nouvel `model_id` dans `models_sklearn` d’un profil (`ideo_quick.yml`).
3. Implémenter la logique associée dans `train_sklearn_model` / `eval_sklearn_model` si nécessaire.

### 11.2 Ajouter un modèle spaCy

1. Créer un `.cfg` dans `configs/spacy/`.
2. Déclarer le `model_id` dans `models.yml` → section `spacy`.
3. L’ajouter à `models_spacy` dans le profil ciblé.
4. S’assurer que `core_prepare` produit bien les shards spacy.

### 11.3 Ajouter une nouvelle vue idéologique

1. Étendre `ideology.yml` (nouveaux labels / regroupements).
2. Adapter `derive_ideology_from_yaml.py` si besoin pour produire un nouveau YAML (ex `ideology_five_way.yml`).
3. Ajouter un profil ou modifier un profil existant pour pointer vers ce nouveau `label_map` + `view`.

### 11.4 Ajouter un nouveau corpus

1. Ajouter une entrée dans `corpora.yml` (nouvel `corpus_id`).
2. Créer `data/raw/<corpus_id>/corpus.xml`.
3. Copier/adapter un profil (`ideo_quick.yml`) en changeant `corpus_id`.

---

## 12. Backlog V4.x (idées / TODO)

Quelques points volontairement laissés ouverts pour itérations futures :

* **Équilibrage** :

  * raffiner `alpha_total` (contrôle plus fin du lissage),
  * `cap_tokens` plus robuste (prise en compte du nb tokens spacy réel).
* **Famille HF** :

  * implémenter l’entraînement complet (CamemBERT, etc.),
  * gestion des gros corpora (shards JSONL, gradient accumulation).
* **Multi-modalité** :

  * brancher facilement d’autres sources que TEI (YouTube, etc.) via de nouveaux `corpus_id` et formats d’entrée (tout en conservant la même interface TSV vue/label).
