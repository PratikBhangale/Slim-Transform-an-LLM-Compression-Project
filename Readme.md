# Project Title: LLM Size Reduction Techniques

## Overview
This project focuses on exploring and implementing techniques to optimize large language models (LLMs) for enhanced efficiency while retaining performance. Key approaches include dataset preprocessing, knowledge distillation, pruning (structured and unstructured), and quantization. These methods aim to reduce model size, computational requirements, and latency, making LLMs more practical for diverse applications.

---

## Setup Instructions
1. Install Python 3.8 or later, along with the dependencies specified in `requirements.txt`.
2. Set up the environment using Anaconda and Jupyter Notebook.
3. To access the full project, download it from github using the following link:
   - `https://github.com/PratikBhangale/Slim-Transform-Compressing-LLM-Models.git`
4. Ensure GPU support is enabled for efficient training and testing of models.

---

## Repository Structure

### Dataset Preprocessing (`Dataset_Preprocessing.ipynb`)
- **Purpose**: Prepares and cleans datasets for training, pruning, and evaluation.
- **Features**:
  - Data cleaning, normalization, and feature extraction.
  - Splitting data into training, validation, and testing subsets.
  - Optimized storage of processed datasets for reuse.

### Knowledge Distillation (`Knowledge_Distillation.ipynb`)
- **Purpose**: Transfers knowledge from a larger teacher model to a smaller student model.
- **Features**:
  - Balances soft label distillation and true label predictions.
  - Enhances performance of the smaller, distilled model.

### Pruning Techniques
1. **Magnitude Pruning (`Magnitude_Pruning.ipynb`)**:
   - Removes weights with the smallest magnitude.
2. **Structured Pruning (`structured_pruning.ipynb`)**:
   - Prunes entire neurons, filters, or layers.
3. **Unstructured Pruning (`unstruct_pruning.ipynb`)**:
   - Sparsely prunes individual weights.
4. **Quadratic Assignment Problem (QAP) Pruning (`QAP.ipynb`)**:
   - Optimizes pruning using QAP-based methodologies.

### Testing and Evaluation (`Testing.ipynb`)
- **Purpose**: Evaluates pruned models using key metrics such as METEOR score and average perplexity.
- **Comparison of Model Performance**:
  - LLama 3.2 (1b): METEOR = 0.39, Perplexity = 4.39
  - LLama 3.1 (8b): METEOR = 0.17, Perplexity = 2.20
  - Distilled LLama (1b): METEOR = 0.26, Perplexity = 2161.37
  - Pruned LLama (1b): METEOR = 0.18, Perplexity = 2.21
  - Selective Quantization LLama (1b): METEOR = 0.33, Perplexity = 5.03

---

## Results Summary
The project demonstrates:
- **Knowledge Distillation**: Significant reduction in model complexity.
- **Pruning Techniques**: Smaller model sizes with minimal performance degradation.
- **Quantization**: Effective compression strategies for deployment.

---

## Future Work
- Exploring advanced quantization methods.
- Extending optimization techniques to multi-modal datasets.
- Automating hyperparameter tuning for better generalization.
