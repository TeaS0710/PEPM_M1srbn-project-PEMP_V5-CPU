teas@teas-laptop13:~/programmes/PEPM_M1srbn-project-PEMP_V4-CPU-main$ tree
.
├── configs
│   ├── common
│   │   ├── balance.yml
│   │   ├── corpora.yml
│   │   ├── hardware.yml
│   │   └── models.yml
│   ├── label_maps
│   │   ├── ideology_actors.yml
│   │   └── ideology_global.yml
│   ├── profiles
│   │   ├── check_only.yml
│   │   ├── crawl_full.yml
│   │   ├── crawl_quick.yml
│   │   ├── custom.yml
│   │   ├── ideo_full.yml
│   │   └── ideo_quick.yml
│   └── spacy
│       ├── textcat_bow_base.cfg
│       └── textcat_cnn_base.cfg
├── data
│   ├── configs
│   │   └── actors_counts_web1.tsv
│   ├── interim
│   │   └── web1
│   │       └── ideology_global
│   ├── processed
│   │   └── web1
│   │       └── ideology_global
│   └── raw
│       └── web1
│           └── corpus.xml
├── dev_V4.md
├── logs
├── makefile
├── (modélisation intégrée à dev_V4.md)
├── models
│   └── web1
│       └── ideology_global
├── old_version
│   ├── PEPM_M1srbn-project-PEMP_V2-CPU-main.zip
│   ├── PEPM_M1srbn-project-PEMP_V3-CPU.zip
│   └── PEPM_M1srbn-project-PEPM_V1.1_CPU.zip
├── patch.md
├── patch_todo.md
├── README.md
├── ref_V4_parameters.md
├── reports
│   └── web1
│       └── ideology_global
├── requirements.txt
└── scripts
    ├── core
    │   ├── core_evaluate.py
    │   ├── core_prepare.py
    │   ├── core_train.py
    │   ├── core_utils.py
    │   ├── __init__.py
    │   └── __pycache__
    │       ├── core_evaluate.cpython-312.pyc
    │       ├── core_prepare.cpython-312.pyc
    │       ├── core_train.cpython-312.pyc
    │       ├── core_utils.cpython-312.pyc
    │       └── __init__.cpython-312.pyc
    ├── experiments
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-312.pyc
    │   │   └── run_grid.cpython-312.pyc
    │   └── run_grid.py
    ├── __init__.py
    ├── post
    │   ├── __init__.py
    │   ├── post_aggregate_metrics.py
    │   └── __pycache__
    │       ├── __init__.cpython-312.pyc
    │       └── post_aggregate_metrics.cpython-312.pyc
    ├── pre
    │   ├── __init__.py
    │   ├── make_ideology_skeleton.py
    │   ├── pre_check_config.py
    │   ├── pre_check_env.py
    │   └── __pycache__
    │       ├── __init__.cpython-312.pyc
    │       ├── make_ideology_skeleton.cpython-312.pyc
    │       ├── pre_check_config.cpython-312.pyc
    │       └── pre_check_env.cpython-312.pyc
    ├── __pycache__
    │   └── __init__.cpython-312.pyc
    └── tools
        ├── corpus_stats.py
        └── sysinfo.py
