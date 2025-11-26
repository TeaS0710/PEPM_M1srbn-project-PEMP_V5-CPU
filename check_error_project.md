1. vérifier le split + les distributions de labels,
2. traquer les **doublons exacts**,
3. traquer les **quasi-doublons** avec vectorisation TF-IDF (ce qu’elle demande),
4. vérifier les biais de corpus (site/source → étiquette),
5. faire un sanity check “labels randomisés” pour écarter un bug du pipeline.

> Hypothèses :
> *dataset_id = `web1`*, *view = `ideology_global`*, colonne texte = `text`, colonne label = `label` (ou `ideology`).
> Si les noms diffèrent, tu adaptes juste `TEXT_COL` / `LABEL_COL` dans les blocs Python.

---

## 0. Point de départ

```bash
cd /chemin/vers/PEPM_M1srbn-project-PEMP_V5.5-CPU-main
source .venv/bin/activate

# (re)générer proprement les TSV
make run STAGE=prepare PROFILE=ideo_quick
```

---

## 1) Vérifier split + distributions de labels

### 1.1. Comptage de base + détection auto de la colonne label

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

DATASET_ID = "web1"
VIEW = "ideology_global"

base = Path("data/interim") / DATASET_ID / VIEW
train = pd.read_csv(base / "train.tsv", sep="\t")
job   = pd.read_csv(base / "job.tsv",   sep="\t")

print("train.shape :", train.shape)
print("job.shape   :", job.shape)

# Détection naïve de la colonne de label
LABEL_CANDIDATES = ["label", "ideology", "y", "target"]
LABEL_COL = None
for c in LABEL_CANDIDATES:
    if c in train.columns:
        LABEL_COL = c
        break

if LABEL_COL is None:
    raise SystemExit(f"Aucune colonne label trouvée parmi {LABEL_CANDIDATES}. "
                     f"Colonnes dispo : {list(train.columns)}")

print(f"\n[OK] Colonne de label utilisée : {LABEL_COL}\n")

for name, df in [("TRAIN", train), ("JOB", job)]:
    print(f"== {name} ==")
    counts = df[LABEL_COL].value_counts()
    props  = df[LABEL_COL].value_counts(normalize=True).round(3)
    print("Counts :")
    print(counts)
    print("Proportions :")
    print(props)
    print()
PY
```

👉 Ça te donne :

* nombre de lignes train/job (vérifier que ça colle avec `TRAIN_PROP=0.6`),
* distribution des labels → montre tout de suite si le corpus est très déséquilibré.

Tu peux déjà calculer à la main le **baseline majority** : proportion de la classe majoritaire dans `JOB`.

---

## 2) Doublons **exacts** (dans train, dans job, et entre les deux)

> Ici il faut connaître le nom de la colonne texte. Je pars sur `"text"`.
> Si c’est `"body"`, `"content"`, `"texte"`, tu modifies `TEXT_COL`.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

DATASET_ID = "web1"
VIEW = "ideology_global"
TEXT_COL = "text"   # ⚠️ adapte si besoin
LABEL_CANDIDATES = ["label", "ideology", "y", "target"]

base = Path("data/interim") / DATASET_ID / VIEW
train = pd.read_csv(base / "train.tsv", sep="\t")
job   = pd.read_csv(base / "job.tsv",   sep="\t")

# Label
LABEL_COL = None
for c in LABEL_CANDIDATES:
    if c in train.columns:
        LABEL_COL = c
        break
if LABEL_COL is None:
    raise SystemExit(f"Colonne label introuvable. Colonnes : {list(train.columns)}")

if TEXT_COL not in train.columns:
    raise SystemExit(f"Colonne texte '{TEXT_COL}' absente de train.tsv. "
                     f"Colonnes : {list(train.columns)}")

if TEXT_COL not in job.columns:
    raise SystemExit(f"Colonne texte '{TEXT_COL}' absente de job.tsv. "
                     f"Colonnes : {list(job.columns)}")

print(f"[INFO] LABEL_COL = {LABEL_COL}, TEXT_COL = {TEXT_COL}\n")

# Doublons *dans* chaque split
for name, df in [("TRAIN", train), ("JOB", job)]:
    dup_mask = df.duplicated(TEXT_COL, keep=False)
    n_dup = dup_mask.sum()
    print(f"{name}: {n_dup} doublons exacts (même texte) sur {len(df)} lignes")
    if n_dup:
        print(df.loc[dup_mask, [TEXT_COL, LABEL_COL]].head(5))
        print()

# Doublons *entre* train et job
train_texts = set(train[TEXT_COL])
job_texts   = set(job[TEXT_COL])
overlap = train_texts & job_texts

print(f"\nDoublons exacts TRAIN/JOB (même texte dans les deux) : {len(overlap)}")
if overlap:
    example = next(iter(overlap))
    print("\nExemple de texte en commun (tronqué) :")
    print(example[:400].replace("\n", " ") + "...")
PY
```

* Si tu as 0 ou presque → pas de fuite triviale par copie stricte.
* S’il y en a beaucoup → tu as un vrai argument “le corpus est truffé de doublons”.

---

## 3) **Quasi-doublons** avec vectorisation TF-IDF (ce que ta prof veut)

Ici on vectorise et on cherche des paires **très similaires** (`cosine > 0.9–0.95`) entre un sous-ensemble de train et job.
On se limite à ~2000 docs par split pour ne pas exploser la RAM.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_ID = "web1"
VIEW = "ideology_global"
TEXT_COL = "text"   # adapte si besoin
LABEL_CANDIDATES = ["label", "ideology", "y", "target"]

MAX_PER_SPLIT = 2000
SIM_THRESHOLD = 0.95   # baisse à 0.9 si tu veux plus large

base = Path("data/interim") / DATASET_ID / VIEW
debug_dir = Path("debug")
debug_dir.mkdir(exist_ok=True)

train = pd.read_csv(base / "train.tsv", sep="\t")
job   = pd.read_csv(base / "job.tsv",   sep="\t")

# Label
LABEL_COL = None
for c in LABEL_CANDIDATES:
    if c in train.columns:
        LABEL_COL = c
        break
if LABEL_COL is None:
    raise SystemExit(f"Colonne label introuvable. Colonnes : {list(train.columns)}")

if TEXT_COL not in train.columns or TEXT_COL not in job.columns:
    raise SystemExit(f"Colonne texte '{TEXT_COL}' absente. "
                     f"train cols={list(train.columns)}, job cols={list(job.columns)}")

print(f"[INFO] LABEL_COL = {LABEL_COL}, TEXT_COL = {TEXT_COL}")

# Échantillon (pour rester raisonnable en RAM)
train_sub = train.sample(
    n=min(MAX_PER_SPLIT, len(train)),
    random_state=0
).reset_index(drop=True)
job_sub = job.sample(
    n=min(MAX_PER_SPLIT, len(job)),
    random_state=1
).reset_index(drop=True)

print(f"[INFO] train_sub: {len(train_sub)} docs, job_sub: {len(job_sub)} docs")

# TF-IDF (on entraîne sur l'union pour partager le vocabulaire)
all_texts = pd.concat([train_sub[TEXT_COL], job_sub[TEXT_COL]], axis=0).fillna("")
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_all = vec.fit_transform(all_texts)

X_train = X_all[:len(train_sub)]
X_job   = X_all[len(train_sub):]

# Similarités job vs train
sim_matrix = (X_job @ X_train.T).toarray()  # (n_job x n_train)

pairs = []
for j_idx in range(sim_matrix.shape[0]):
    row = sim_matrix[j_idx]
    # indices triés par similarité décroissante
    good = np.where(row >= SIM_THRESHOLD)[0]
    for t_idx in good:
        sim = row[t_idx]
        pairs.append(
            (
                float(sim),
                "job", int(j_idx),
                "train", int(t_idx),
            )
        )

print(f"[INFO] Paires avec sim >= {SIM_THRESHOLD} : {len(pairs)}")

# On garde les meilleures
pairs_sorted = sorted(pairs, key=lambda x: -x[0])[:200]

rows = []
for sim, job_flag, j_idx, train_flag, t_idx in pairs_sorted:
    j_row = job_sub.iloc[j_idx]
    t_row = train_sub.iloc[t_idx]
    rows.append({
        "sim": round(sim, 4),
        "job_index": int(j_idx),
        "train_index": int(t_idx),
        "job_label": j_row[LABEL_COL],
        "train_label": t_row[LABEL_COL],
        "job_text": str(j_row[TEXT_COL])[:400].replace("\n", " "),
        "train_text": str(t_row[TEXT_COL])[:400].replace("\n", " "),
    })

df_pairs = pd.DataFrame(rows)
out_path = debug_dir / "near_duplicates_web1.tsv"
df_pairs.to_csv(out_path, sep="\t", index=False)
print(f"[OUT] {len(df_pairs)} paires quasi-doublons écrites dans {out_path}")
PY
```

Tu ouvres ensuite `debug/near_duplicates_web1.tsv` dans un tableur ou un éditeur, et tu montres à ta prof :

* s’il y a beaucoup de paires à 0.98–1.0 → corpus **rempli d’articles quasi identiques**,
* si les labels de ces quasi-doublons sont cohérents,
* s’il y a des cas où un *même* texte (ou quasi) a deux labels → problème d’annotation.

Ça, c’est exactement “vectoriser pour comparer les articles”.

---

## 4) Vérifier les **biais de corpus** (site/source → idéologie)

Si certains sites sont 100 % gauche ou 100 % droite, le modèle peut “tricher” en apprenant juste la source.

On regarde les crosstabs pour quelques colonnes métadonnées typiques (`corpus_id`, `source`, `site`, `modality`, etc.).

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

DATASET_ID = "web1"
VIEW = "ideology_global"
LABEL_CANDIDATES = ["label", "ideology", "y", "target"]

base = Path("data/interim") / DATASET_ID / VIEW
job = pd.read_csv(base / "job.tsv", sep="\t")

LABEL_COL = None
for c in LABEL_CANDIDATES:
    if c in job.columns:
        LABEL_COL = c
        break
if LABEL_COL is None:
    raise SystemExit(f"Colonne label introuvable. Colonnes : {list(job.columns)}")

META_CANDIDATES = ["corpus_id", "source", "site", "modality", "media", "channel"]

print(f"[INFO] LABEL_COL = {LABEL_COL}")
print("[INFO] Colonnes meta candidates présentes :",
      [c for c in META_CANDIDATES if c in job.columns])

for col in META_CANDIDATES:
    if col not in job.columns:
        continue
    print(f"\n=== Crosstab {col} x {LABEL_COL} (normalisé par ligne) ===")
    ct = pd.crosstab(job[col], job[LABEL_COL], normalize="index").round(3)
    print(ct)
PY
```

* Si tu vois des lignes genre “site_X : 0.99 left / 0.01 right”,
  → tu as un **biais de source énorme** que tu peux documenter.
* Tu peux aussi montrer que, sans même regarder les textes, un classifieur “au pif mais connaissant le site” aurait déjà de très bons scores.

C’est un argument solide pour expliquer des performances “trop bonnes”.

---

## 5) Sanity check ultime : **randomiser les labels** et re-entraîner

Ça, c’est pour tester si ton pipeline est sain : si tu casses texte→label, les scores doivent tomber au niveau du hasard.

```bash
# 5.1. Re-générer un train propre
make run STAGE=prepare PROFILE=ideo_quick

# 5.2. Randomiser les labels du train.tsv
python - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np

DATASET_ID = "web1"
VIEW = "ideology_global"
LABEL_CANDIDATES = ["label", "ideology", "y", "target"]

base = Path("data/interim") / DATASET_ID / VIEW
train_path  = base / "train.tsv"
backup_path = base / "train.original.tsv"

df = pd.read_csv(train_path, sep="\t")

LABEL_COL = None
for c in LABEL_CANDIDATES:
    if c in df.columns:
        LABEL_COL = c
        break
if LABEL_COL is None:
    raise SystemExit(f"Colonne label introuvable. Colonnes : {list(df.columns)}")

print(f"[INFO] LABEL_COL = {LABEL_COL}")

# Backup de l'original (une seule fois)
if not backup_path.exists():
    df.to_csv(backup_path, sep="\t", index=False)
    print(f"[BACKUP] train original sauvegardé dans {backup_path}")
else:
    print(f"[BACKUP] {backup_path} existe déjà (non réécrit)")

print("\n[AVANT] Répartition des labels :")
print(df[LABEL_COL].value_counts())

# Randomisation des labels (en gardant la distribution)
rng = np.random.RandomState(42)
shuffled = df[LABEL_COL].sample(frac=1.0, random_state=rng).reset_index(drop=True)
df[LABEL_COL] = shuffled

print("\n[APRES] Répartition des labels (doit être similaire mais mélangée) :")
print(df[LABEL_COL].value_counts())

df.to_csv(train_path, sep="\t", index=False)
print(f"\n[WRITE] train.tsv écrasé avec labels randomisés")
PY

# 5.3. Train + evaluate sur ces données pourries
make run STAGE=train    PROFILE=ideo_quick
make run STAGE=evaluate PROFILE=ideo_quick
```

Ensuite tu ouvres un `metrics.json` / `classification_report.txt` :

* Si accuracy + macro-F1 tombent vers le **hasard**, ton pipeline d’éval est **sain**.
* Si tu restes à des scores “bons” → il y a forcément une fuite ou un bug.

---

## 6) Comment vendre ça à ta prof

Avec **ces blocs** tu peux arriver en disant :

> 1. J’ai vérifié les splits et les distributions de labels (commande 1).
> 2. J’ai testé les doublons exacts (commande 2).
> 3. J’ai vectorisé le corpus et extrait les quasi-doublons (commande 3) → fichier `debug/near_duplicates_web1.tsv`.
> 4. J’ai mesuré les biais de source (commande 4) : certains sites sont quasi mono-idéologie.
> 5. J’ai randomisé les labels (commande 5) : les performances s’effondrent → donc le pipeline ne triche pas.

Et tu peux conclure ensuite :

* soit “le corpus est vraiment très biaisé / facile (lexique + sources)” → résultats élevés mais explicables,
* soit tu mets le doigt sur un vrai bug (doublons massifs, labels contradictoires, etc.),
* soit… oui, tu as vraiment bien bossé, mais tu peux le démontrer proprement
