# labram_and_xai

This project implements the fine-tuning of the Large Brain Model (LaBraM) and explainability tools for it

## Phase 1 - Fine-tuning LaBraM
- Make sure to prepare the dataset to be compatible with the LaBraM input. Examples can be found in `src/labram_ft/dataset_maker/`.
- Select/create a suitable dataloader in `src/labram_ft/dataset_maker/utils.py`
- Run the main script `src/labram_ft/dataset_maker/run_class_finetuning.py`

## Phase 2 - XAI
### Attention maps
- If attention pooling was used during fine-tuning, you can run `src/xai_attention_maps/visualize_attention.py` for displaying the attention attributed to different patches by the model. This is a simple first interpretability step.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```
