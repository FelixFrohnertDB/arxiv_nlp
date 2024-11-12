# Overview
This project investigates using word embeddings learned from arXiv abstracts in the quantum physics domain for predicting concept pair combinations.
The workflow includes data processing, concept extraction, embedding training, predictive model training, and baseline analysis. 
The goal is to understand concept trends and relationships in quantum literature.

### File Descriptions
1_process_arxiv_data.ipynb: 
- Loads and preprocesses arXiv abstracts. 
The arxiv data can be downloaded from https://www.kaggle.com/datasets/Cornell-University/arxiv

2_extract_concepts.ipynb: 
- Extracts quantum physics concepts.

Embedding and Model Training
3_train_embedding.ipynb: 
- Trains word embeddings on the processed abstracts to capture semantic relationships between extracted concepts.

4_train_prediction.ipynb: 
- Trains a neural network classifier to predict co-occurrence of concept pairs.


5_baseline_1.ipynb to 5_baseline_5.ipynb: 
- Implement various baseline models to compare against the primary embedding model. 

Additional Analysis
6_additional_plots.ipynb: 
- Generates additional visualizations.
