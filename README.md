# Topic Modeling for Breast Cancer Research
This project aims to develop a method to **quickly understand the landscape of breast cancer research**. This approach could be applied to other domains of research. 

Although online databases such as PubMed and Google Scholar provide access to research papers, it can be challenging to understand the major frontiers in a field of research. People who are new to a field or subfields may face a "cold-start" problem in which they lack keywords to begin researching. 

Topic Modeling is an unsupervised machine learning algorithm that discovers topics across a set of documents. The benefits of this approach are in providing:
* a comprehensive set of prominent keywords used in different subfields
* an ability to measure which subfields might be more popular
* an ability to measure which subfields might be trending.

See my [blog post](https://edmondchensj.github.io/2018/08/09/breast-cancer-trends/) for the results.

## Usage
### 1. Data Retrieval
* Run `python code/get_data.py`
* Note: By default, NCBI restricts the rate of downloads to 3 requests per second. To speed up data retrieval, register for an API key, which will increase the rate to 10 requests per second. See [here](https://www.ncbi.nlm.nih.gov/books/NBK25497/) for more information. 

### 2. Preprocessing
* Run `python code/preprocess.py`
* Note: A big factor affecting the topic model results is the list of additional stop words. You would want to customize your own list based on the domain of research. 

### 3. Topic Modeling (LDA and TFIDF)
* Run `python code/build_models.py`

### 4a. Preliminary Visualization
* Run `python code/visualize_models.py`
* This generates a coherence graph to select the optimum number of topics, a list of wordclouds that will be required for step 4b, and a grid of wordclouds that combines all topics together. 
* The *helper_files* folder contains a .png file to mask wordclouds in a circle shape. 

### 4b. Postprocessing
* Run `python code/postprocess.py`
* This generates two key charts that show the distribution and trends of each topic. 

## Acknowledgements
Data is provided by the US National Library of Medicine (NLM) via the PubMed database.