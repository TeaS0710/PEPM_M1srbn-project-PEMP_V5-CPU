PYTHON                  ?= python3
ifneq ("$(wildcard .venv/bin/python)","")
PYTHON                  :=.venv/bin/python
endif

PROFILE                 ?= ideo_quick

# Corpus / vue par défaut (overridables en ligne de commande)
CORPUS_ID               ?= web1
VIEW                    ?= ideology_global

TRAIN_PROP              ?=
BALANCE_STRATEGY        ?=
BALANCE_PRESET          ?=
HARDWARE_PRESET         ?=
FAMILIES                ?=
SEED                    ?=
MAX_DOCS_SKLEARN        ?=
MAX_DOCS_SPACY          ?=
MAX_DOCS_HF             ?=

OVERRIDES               ?=
FAMILY                  ?=

CORPUS_XML              ?= data/raw/$(CORPUS_ID)/corpus.xml
IDEO_MAP_OUT            ?= configs/label_maps/ideology_actors.yml
IDEO_REPORT_OUT         ?= data/configs/actors_counts_$(CORPUS_ID).tsv
MIN_CHARS_IDEO          ?= 200
TOP_VARIANTS_IDEO       ?= 5

export PYTHONPATH       :=.

AUTO_OVERRIDES          :=

ifneq ($(strip $(CORPUS_ID)),)
AUTO_OVERRIDES          +=corpus_id=$(CORPUS_ID)
endif
ifneq ($(strip $(TRAIN_PROP)),)
AUTO_OVERRIDES          +=train_prop=$(TRAIN_PROP)
endif
ifneq ($(strip $(BALANCE_STRATEGY)),)
AUTO_OVERRIDES          +=balance_strategy=$(BALANCE_STRATEGY)
endif
ifneq ($(strip $(BALANCE_PRESET)),)
AUTO_OVERRIDES          +=balance_preset=$(BALANCE_PRESET)
endif
ifneq ($(strip $(HARDWARE_PRESET)),)
AUTO_OVERRIDES          +=hardware_preset=$(HARDWARE_PRESET)
endif
ifneq ($(strip $(FAMILIES)),)
AUTO_OVERRIDES          +=families=$(FAMILIES)
endif
ifneq ($(strip $(SEED)),)
AUTO_OVERRIDES          +=seed=$(SEED)
endif
ifneq ($(strip $(MAX_DOCS_SKLEARN)),)
AUTO_OVERRIDES          +=max_train_docs_sklearn=$(MAX_DOCS_SKLEARN)
endif
ifneq ($(strip $(MAX_DOCS_SPACY)),)
AUTO_OVERRIDES          +=max_train_docs_spacy=$(MAX_DOCS_SPACY)
endif
ifneq ($(strip $(MAX_DOCS_HF)),)
AUTO_OVERRIDES          +=max_train_docs_hf=$(MAX_DOCS_HF)
endif
ALL_OVERRIDES		:=$(AUTO_OVERRIDES) $(OVERRIDES)

OVR_FLAGS		=$(foreach o,$(ALL_OVERRIDES),--override $(o))

FAMILY_FLAG		=$(if $(FAMILY),--only-family $(FAMILY),)

.PHONY: help \
list_profiles \
run \
check \
prepare prepare_dry \
train evaluate pipeline \
ideology_skeleton \
init_dirs venv install setup \
check_scripts diagnostics \
sysinfo \
clean

help:
	@echo "Pipeline V4 (config-first)"
	@echo ""
	@echo "Point d'entrée universel :"
	@echo "  make run STAGE=pipeline PROFILE=ideo_quick"
	@echo "  make run STAGE=prepare PROFILE=ideo_quick CORPUS_ID=web2 TRAIN_PROP=0.7"
	@echo ""
	@echo "Variables principales :"
	@echo "  PROFILE          -> profil YAML (configs/profiles/\$$PROFILE.yml)"
	@echo "  STAGE            -> check | prepare | prepare_dry | train | evaluate | pipeline"
	@echo "  FAMILY           -> spacy | sklearn | hf | check (pour train/evaluate)"
	@echo ""
	@echo "Knobs haut niveau (transformés en --override key=val) :"
	@echo "  CORPUS_ID        -> corpus_id"
	@echo "  TRAIN_PROP       -> train_prop"
	@echo "  BALANCE_STRATEGY -> balance_strategy"
	@echo "  BALANCE_PRESET   -> balance_preset (preset dans balance.yml)"
	@echo "  HARDWARE_PRESET  -> hardware_preset (preset dans hardware.yml)"
	@echo "  FAMILIES         -> families (ex: spacy,sklearn)"
	@echo "  SEED             -> seed"
	@echo "  MAX_DOCS_SKLEARN -> max_train_docs_sklearn"
	@echo "  MAX_DOCS_SPACY   -> max_train_docs_spacy"
	@echo "  MAX_DOCS_HF      -> max_train_docs_hf"
	@echo ""
	@echo "Overrides bruts :"
	@echo "  OVERRIDES        -> liste libre de \"cle=val\" (ex: OVERRIDES=\"debug_mode=true\")"
	@echo "  Exemples idéologie : OVERRIDES=\"ideology.view=five_way\""
	@echo "                     OVERRIDES=\"ideology.unknown_actors.policy=keep\""
	@echo ""
	@echo "Mise en place :"
	@echo "  make setup                      # venv (si besoin) + install deps + arbo + check global"
	@echo ""
	@echo "Cibles directes :"
	@echo "  make check PROFILE=...          # vérif globale (imports + scripts + profil)"
	@echo "  make prepare PROFILE=...        # TEI -> TSV (+ formats spacy)"
	@echo "  make train PROFILE=... FAMILY=sklearn"
	@echo "  make evaluate PROFILE=... FAMILY=sklearn"
	@echo "  make pipeline PROFILE=...       # check + prepare + train + evaluate"
	@echo ""

venv:
	@if [ ! -x ".venv/bin/python" ]; then \
	echo "[venv] Création de l'environnement virtuel .venv..."; \
	python3 -m venv .venv; \
	.venv/bin/python -m pip install -U pip; \
	else \
	echo "[venv] Environnement .venv déjà présent."; \
	fi

install: venv
	@echo "[install] Installation des dépendances Python (requirements.txt)..."
	$(PYTHON) -m pip install -r requirements.txt

init_dirs:
	mkdir -p data/raw/$(CORPUS_ID)
	mkdir -p data/interim/$(CORPUS_ID)/$(VIEW)
	mkdir -p data/processed/$(CORPUS_ID)/$(VIEW)
	mkdir -p models/$(CORPUS_ID)/$(VIEW)
	mkdir -p reports/$(CORPUS_ID)/$(VIEW)
	mkdir -p logs
	@echo "[init_dirs] Arborescence de base créée pour CORPUS_ID=$(CORPUS_ID), VIEW=$(VIEW)."
	@echo "  - Place ton corpus TEI dans: data/raw/$(CORPUS_ID)/corpus.xml"

setup: install init_dirs check
	@echo "[setup] Terminé."
	@echo "  - Vérifie que ton corpus TEI est bien dans data/raw/web1/corpus.xml"
	@echo "  - Puis lance par exemple :"
	@echo "      make run STAGE=pipeline PROFILE=$(PROFILE)"

sysinfo:
	$(PYTHON) scripts/tools/sysinfo.py

check_scripts:
	@echo "[check_scripts] Compilation de tous les scripts Python (py_compile)..."
	@find scripts -name '*.py' -print0 | xargs -0 -n1 $(PYTHON) -m py_compile
	@echo "[check_scripts] OK : syntaxe Python valide pour tous les scripts."

diagnostics:
	$(PYTHON) scripts/pre/pre_check_env.py --profile $(PROFILE) || true

check: diagnostics check_scripts
	@echo "[check] Vérification du profil $(PROFILE)..."
	$(PYTHON) scripts/pre/pre_check_config.py \
	--profile $(PROFILE) \
	$(OVR_FLAGS) \
	--verbose
	@echo "[check] OK : environnement + scripts + profil $(PROFILE) validés."


list_profiles:
	@echo "Profils disponibles (configs/profiles/*.yml) :"
	@ls configs/profiles/*.yml | sed 's|configs/profiles/||;s|\.yml||'


prepare:
	@echo "[prepare] Profil: $(PROFILE)"
	$(PYTHON) scripts/core/core_prepare.py \
	--profile $(PROFILE) \
	$(OVR_FLAGS) \
	--verbose

prepare_dry:
	@echo "[prepare_dry] Profil: $(PROFILE) (dry-run)"
	$(PYTHON) scripts/core/core_prepare.py \
	--profile $(PROFILE) \
	$(OVR_FLAGS) \
	--dry-run \
	--verbose


train:
	@echo "[train] Profil: $(PROFILE)  Famille: $(FAMILY)"
	$(PYTHON) scripts/core/core_train.py \
	--profile $(PROFILE) \
	$(OVR_FLAGS) \
	$(FAMILY_FLAG) \
	--verbose


evaluate:
	@echo "[evaluate] Profil: $(PROFILE)  Famille: $(FAMILY)"
	$(PYTHON) scripts/core/core_evaluate.py \
	--profile $(PROFILE) \
	$(OVR_FLAGS) \
	$(FAMILY_FLAG) \
	--verbose


pipeline: check prepare train evaluate

STAGE ?= pipeline

run:
	@echo "[run] STAGE=$(STAGE) PROFILE=$(PROFILE) FAMILY=$(FAMILY)"
	@if [ "$(STAGE)" = "check" ]; then \
	$(MAKE) check PROFILE=$(PROFILE) OVERRIDES="$(OVERRIDES)" \
	CORPUS_ID="$(CORPUS_ID)" TRAIN_PROP="$(TRAIN_PROP)" \
	BALANCE_STRATEGY="$(BALANCE_STRATEGY)" BALANCE_PRESET="$(BALANCE_PRESET)" \
	HARDWARE_PRESET="$(HARDWARE_PRESET)" FAMILIES="$(FAMILIES)" \
	SEED="$(SEED)" MAX_DOCS_SKLEARN="$(MAX_DOCS_SKLEARN)" \
	MAX_DOCS_SPACY="$(MAX_DOCS_SPACY)" MAX_DOCS_HF="$(MAX_DOCS_HF)"; \
	elif [ "$(STAGE)" = "prepare" ]; then \
	$(MAKE) prepare PROFILE=$(PROFILE) FAMILY="$(FAMILY)" \
	OVERRIDES="$(OVERRIDES)" CORPUS_ID="$(CORPUS_ID)" TRAIN_PROP="$(TRAIN_PROP)" \
	BALANCE_STRATEGY="$(BALANCE_STRATEGY)" BALANCE_PRESET="$(BALANCE_PRESET)" \
	HARDWARE_PRESET="$(HARDWARE_PRESET)" FAMILIES="$(FAMILIES)" \
	SEED="$(SEED)" MAX_DOCS_SKLEARN="$(MAX_DOCS_SKLEARN)" \
	MAX_DOCS_SPACY="$(MAX_DOCS_SPACY)" MAX_DOCS_HF="$(MAX_DOCS_HF)"; \
	elif [ "$(STAGE)" = "prepare_dry" ]; then \
	$(MAKE) prepare_dry PROFILE=$(PROFILE) FAMILY="$(FAMILY)" \
	OVERRIDES="$(OVERRIDES)" CORPUS_ID="$(CORPUS_ID)" TRAIN_PROP="$(TRAIN_PROP)" \
	BALANCE_STRATEGY="$(BALANCE_STRATEGY)" BALANCE_PRESET="$(BALANCE_PRESET)" \
	HARDWARE_PRESET="$(HARDWARE_PRESET)" FAMILIES="$(FAMILIES)" \
	SEED="$(SEED)" MAX_DOCS_SKLEARN="$(MAX_DOCS_SKLEARN)" \
	MAX_DOCS_SPACY="$(MAX_DOCS_SPACY)" MAX_DOCS_HF="$(MAX_DOCS_HF)"; \
	elif [ "$(STAGE)" = "train" ]; then \
	$(MAKE) train PROFILE=$(PROFILE) FAMILY="$(FAMILY)" \
	OVERRIDES="$(OVERRIDES)" CORPUS_ID="$(CORPUS_ID)" TRAIN_PROP="$(TRAIN_PROP)" \
	BALANCE_STRATEGY="$(BALANCE_STRATEGY)" BALANCE_PRESET="$(BALANCE_PRESET)" \
	HARDWARE_PRESET="$(HARDWARE_PRESET)" FAMILIES="$(FAMILIES)" \
	SEED="$(SEED)" MAX_DOCS_SKLEARN="$(MAX_DOCS_SKLEARN)" \
	MAX_DOCS_SPACY="$(MAX_DOCS_SPACY)" MAX_DOCS_HF="$(MAX_DOCS_HF)"; \
	elif [ "$(STAGE)" = "evaluate" ]; then \
	$(MAKE) evaluate PROFILE=$(PROFILE) FAMILY="$(FAMILY)" \
	OVERRIDES="$(OVERRIDES)" CORPUS_ID="$(CORPUS_ID)" TRAIN_PROP="$(TRAIN_PROP)" \
	BALANCE_STRATEGY="$(BALANCE_STRATEGY)" BALANCE_PRESET="$(BALANCE_PRESET)" \
	HARDWARE_PRESET="$(HARDWARE_PRESET)" FAMILIES="$(FAMILIES)" \
	SEED="$(SEED)" MAX_DOCS_SKLEARN="$(MAX_DOCS_SKLEARN)" \
	MAX_DOCS_SPACY="$(MAX_DOCS_SPACY)" MAX_DOCS_HF="$(MAX_DOCS_HF)"; \
	elif [ "$(STAGE)" = "pipeline" ] || [ -z "$(STAGE)" ]; then \
	$(MAKE) pipeline PROFILE=$(PROFILE) FAMILY="$(FAMILY)" \
	OVERRIDES="$(OVERRIDES)" CORPUS_ID="$(CORPUS_ID)" TRAIN_PROP="$(TRAIN_PROP)" \
	BALANCE_STRATEGY="$(BALANCE_STRATEGY)" BALANCE_PRESET="$(BALANCE_PRESET)" \
	HARDWARE_PRESET="$(HARDWARE_PRESET)" FAMILIES="$(FAMILIES)" \
	SEED="$(SEED)" MAX_DOCS_SKLEARN="$(MAX_DOCS_SKLEARN)" \
	MAX_DOCS_SPACY="$(MAX_DOCS_SPACY)" MAX_DOCS_HF="$(MAX_DOCS_HF)"; \
	else \
echo "[run] Erreur : STAGE=$(STAGE) inconnu. Utiliser check|prepare|prepare_dry|train|evaluate|pipeline."; \
	exit 1; \
	fi

ideology_skeleton:
	$(PYTHON) scripts/pre/make_ideology_skeleton.py \
	--corpus $(CORPUS_XML) \
	--out-yaml $(IDEO_MAP_OUT) \
	--out-report $(IDEO_REPORT_OUT) \
	--min-chars $(MIN_CHARS_IDEO) \
	--top-variants $(TOP_VARIANTS_IDEO)
	@echo "YAML squelette ideologie écrit dans $(IDEO_MAP_OUT)"
	@echo "Rapport d'acteurs écrit dans $(IDEO_REPORT_OUT)"

clean:
	@echo "[clean] Suppression des fichiers intermédiaires, modèles et rapports..."
	rm -rf data/interim/* \
	data/processed/* \
	models/* \
	reports/*
	@echo "[clean] OK"
