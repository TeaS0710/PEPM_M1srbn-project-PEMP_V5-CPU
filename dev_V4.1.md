# V4.1 – Pistes de développement ultérieur

Ce document esquisse les évolutions prioritaires ou exploratoires au-delà de la V4 actuelle.

## 1. Robustesse et extensibilité court terme
- **Clustering/semisupervisé** :
  - Nouvelle famille `cluster` (KMeans/HDBSCAN) branchée sur les TSV `train/job` pour explorer des corpus non étiquetés.
  - Export des embeddings (spaCy/HF) en cache pour réutiliser les features sans recalcule.
- **Upgrade HF** :
  - Support GPU optionnel, accumulation de gradients et mixed-precision configurables via `models.yml` + `hardware.yml`.
  - Grille d'optimisation légère (learning rate, max_length) appliquée via `OVERRIDES`.
- **Monitoring des distributions** :
  - Scripts de drift/shift entre train/job et entre versions de corpus, avec métriques stockées dans `reports/`.
  - Alertes simples (JSON/CSV) et visualisations rapides (histogrammes label/longueur) pour détecter les dérives.
- **Packaging léger** :
  - Publication d'un `pyproject.toml` minimal et d'un binaire CLI unique (`pepm`) en conservant l'interface Make.
  - Prévoir des wheel CPU only pour faciliter l'installation offline.

## 2. Orchestration et MLOps
- **Sweeps reproductibles** : cibles Make ou scripts `run_grid.py` alimentés par des matrices d'`OVERRIDES` (hydra-like) et stockage systématique des seeds dans les metas.
- **Registry de modèles** : index `reports/index.json` listant les entraînements (profil, seed, famille, métriques clés) pour faciliter les comparaisons et l'automatisation des sélections.
- **Artefacts compressés** : option pour archiver `meta_*`, checkpoints et logs par run (tar/gzip) afin de faciliter le transfert entre machines ou l'upload vers un bucket.

## 3. Qualité des données
- **Normalisation linguistique** : pipeline de détection de langue, détection de doublons et filtres de toxicité/off-topic avec stats intégrées dans `meta_view`.
- **Augmentations contrôlées** : règles simples (synonymes, masquage, permutation) activables par profil pour enrichir le train sans polluer le job ; traçabilité des taux d'augmentation.
- **Labeling assisté** : boucle semi-automatique pour générer/valider des `label_maps` dérivés (actors, crawl) et pousser les suggestions dans `configs/label_maps/`.

## 4. Visualisation et reporting
- **Dashboards légers** : génération automatique de notebooks HTML (via `papermill` ou `nbconvert`) synthétisant métriques, matrices de confusion et dérive des labels.
- **Explorateur de corpus** : vue interactive (Streamlit/Gradio) branchée sur les TSV intermédiaires pour filtrer par acteur, label, longueur, etc.

## 5. Compatibilité et migration
- **Import V1/V2** : scripts pour convertir les sorties historiques (TSV, JSON) vers la structure V4 sans repasser par le prepare complet, avec validation automatique des champs attendus.
- **Migration V4 → V4.1** : guide de checklist (hardware, modèles, label maps) pour garder la reproductibilité lors de l'upgrade et inclure des tests smoke Make.

## 6. Notes de maintenance (mini-corpus acteurs)
- Nettoyage du pipeline idéo : suppression du mode legacy dans `core_prepare` et prise en charge unique du YAML `ideology_actors.yml` (résolution par domaine puis crawl).
- Génération du squelette acteurs enrichi (domaines) via `make ideology_skeleton` + labellisation minimale pour `web1` afin d'avoir deux classes actives.
- Smoke-test reproductible : `scripts/tests/test_midlevel_demo.py` exécute `prepare → train → evaluate` sur le mini corpus TEI web1 (sklearn) et vérifie la présence des artefacts.

