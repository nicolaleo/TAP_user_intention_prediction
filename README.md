# User intention prediction for trigger-action programming rule

This repository takes inspiration from the paper "User intention prediction for trigger-action programming rule using multi-view representation learning" https://www.sciencedirect.com/science/article/abs/pii/S0957417424030653?via%3Dihub and want to build a representation model for user intention prediction in a TAP (Trigger-action programming) framework settings.

The task is a typical multilabel classification problem.

# Environment Setup

pip install -r requirements.txt

# Dataset
The reference dataset is the IFTTT If-This-Then-That dataset  (https://zenodo.org/records/5572861 also available on kaggle https://www.kaggle.com/code/hrs2kr/analysis-on-ifttt-dataset).
The main refernce file is the Step4_Single_Trigger_IoT_Rules.csv that contains the target column "goal".

# Approach
A step by step methodology oriented to clarity and pragmatism

-[01_dataset_analysis_and_preparation](01_dataset_analysis_and_preparation.ipynb) EDA analysis and some preprocessing steps 

-[02_text_embedding_creation_and_representation](02_text_embedding_creation_and_representation.ipynb) text embedding creation (sentence-transformers library) and UMAP representation

-[02_text_embedding_creation_and_representation_trasformers](02_text_embedding_creation_and_representation_transformers.ipynb) text embedding creation (transformers library) and UMAP representation

-[02_text_embedding_creation_and_representation_finetuned](02_text_embedding_creation_and_representation_finetuned.ipynb) text embedding creation (transformers library) and UMAP representation using a finetuned model

-[02_text_embedding_creation_and_representation_LLM](02_text_embedding_creation_and_representation_LLM.ipynb) text embedding creation (via ollama) and UMAP representation using a decoder style model

-[03_dataset_manipulation_for_multilabel_classification](03_dataset_manipulation_for_multilabel_classification.ipynb) produce a dataset suitable for a multilabel classification task 

## Modeling
Starting from a baseline model the aim of the project is to build AI models of increasing complexity to achieve the best performance scores.
- **Using only the textual representation of the rule  (text embeddings as features) and add a custom classifier on top**
  - [04_train_features_extractor](04_train_features_extractor.ipynb) train a classifier on top the embedding representation
  - [enhanced text] add other dataset columns to the "name" column
  - [05_embedding_model_finetuning](05_embedding_model_finetuning.ipynb) finetune the base embedding model with a Contrastive Loss (using the "goal" attribute for example). Only the simplest approach (anchor, positive) + MultipleNegativesRankingLoss implemented and evaluated
- **Model Finetuning**  
   - [Only last layer]
   - [Some layers]
   - [PEF method eg LORA]
- **Decoder style LLM classificator**   
    - [06_LLM_classificator](06_LLM_classificator.ipynb) use prompting techniques to test an LLM as a classificator - Llamav3.2 1 billion parameter model 
    - [Work with encoder style representation and decoder style] try to merge the two approaches
- **Graph Neural Network modeling**
    - [07_kg_creation](07_kg_creation.ipynb) create the kg representation for the dataset [the generated graph file](out/IFTTT_graph.json)
    - [Create the graph representation and apply a GNN approach on this] extract the graph representation embeddings of the set of rules
- **Multi-view representation learning**   
    - [Representation Fusion] Merge different representation

# Results

|      | Approach   | Model    | Train Accuracy | Train F1-micro | Train F1-macro | Test Accuracy | Test F1-micro | Test F1-macro | 
| ---- | -----------| -------- | -------------- | -------------- | -------------- | --------------| ------------- | ------------- | 
|  1   |  Features extractor  |  bert-base-uncased + Logistic Regression   |  85.38%   |     0.92        |      0.94       |     52.60%       |    0.66      |     0.65     |
|  2   |  Features extractor  |  all-mpnet-base-v2 + Logistic Regression   |  73.59%   |     0.83        |      0.81       |     57.81%       |    0.71      |     0.69     |
|  3   |  Features extractor  |  ModernBERT-base + Logistic Regression   |  79.02%   |     0.88        |      0.88       |     47.14%       |    0.62      |     0.56     |
|  4   |  Features extractor  |  **all-mpnet-base-v2 finetuned + Logistic Regression**   |  71.76%   |     0.82        |      0.67       |     65.89%       |    0.77      |     0.64     |
|  5   |  Features extractor  |  llama3.2 1 billion + Logistic Regression   |  18.53%   |     0.31        |      0.18       |     14.71%       |    0.25      |     0.13     |






# UMAP 2d representations
### bert-base-uncased
<img src="figures/bert_uncased_umap_2d.png" width=1000>

### all-mpnet-base-v2
<img src="figures/all-mpnet-base-v2_umap_2d.png" width=1000>

### ModernBERT-base
<img src="figures/modernBert_base_umap_2d.png" width=1000>

### all-mpnet-base-v2 finetuned (the best clustered ones so far)
<img src="figures/all-mpnet-base-v2_finetuned_umap_2d.png" width=1000>

### llama3.2 1 billion
<img src="figures/llama3.2.1_umap_2d.png" width=1000>

# Comments
## Features extractor Approach
From the representation of UMAP embeddings it is evident how the all-mpnet-base-v2 model is able to group in a stronger way textual representations of rules with the same goal. This better "clustered representation" helps the cascaded classifier which therefore obtains better results (test-set).

The same logic is followed by the all-mpnet-base-v2-finetuned model which has better performance. It is the best in its category so far. It is clear from the UMAP representation and metrics results that the embedding finetuning step with the simple MultipleNegativesRankingLoss is able to improve the model overall performance.


The results of using this decoder style embedding model lllama3.2 1 billion are quite bad in comparison to the other encoder style models.
This is evident from the 2d umap representation of the embeddings which clearly appear less "clustered" than the other models in relation to the classes to be discriminated.