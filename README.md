# cancer-research
Identifying major topics in cancer research from 1 million publications over the last 20 years.

Description: This project aims to retrieve a million medical publications from PubMed in the field of cancer and use Natural Language Processing and unsupervised Topic Modeling to map the major research areas. Tools:
- Biopython Entrez API
- NLTK
- Gensim
- PyLDAVis

## Order of code for this project:
1. get_data.py (Retrieve data from PubMed)
2. preprocess.py (Preprocess data using NLP)
3. build_models.py (Build Latent Dirichlet Allocation (LDA) models to get topics and keep the best ones)
4. visualize_model.py (Visualize one model at a time)
5. postprocess.py (Get meaningful insights from model)