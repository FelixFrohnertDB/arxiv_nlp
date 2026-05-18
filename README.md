# Predicting Concept Combinations 

This repository investigates using word embeddings learned from arXiv abstracts to predict future concept pair combinations within quantum physics literature. The goal is to discover underlying semantic relationships and map emerging trends in quantum research.

[![MLST](https://img.shields.io/badge/MLST-10.1088/2632--2153/adb00a-blue.svg)](https://doi.org/10.1088/2632-2153/adb00a)

---

##  Project Workflow & Notebooks

| File | Phase | Description |
| --- | --- | --- |
| `1_process_arxiv_data.ipynb` | **Data Prep** | Loads and preprocesses raw arXiv abstracts. |
| `2_extract_concepts.ipynb` | **Extraction** | Extracts specific quantum physics concepts from text. |
| `3_train_embedding.ipynb` | **Modeling** | Trains word embeddings to capture semantic relationships. |
| `4_train_prediction.ipynb` | **Prediction** | Trains a neural network classifier to predict concept co-occurrence. |
| `5_baseline_1.ipynb` to `_5.ipynb` | **Evaluation** | Implements five distinct baseline models for benchmarking. |
| `6_additional_plots.ipynb` | **Analysis** | Generates final data visualizations and performance plots. |

---

##  Dataset

The model relies on the open-source arXiv dataset containing millions of scholarly articles.

>  **Download Link:** You can download the required data directly from the [Cornell University arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).

---

##  Setup & Installation

Get the environment ready by installing the package:

```bash
pip install -e .
```
