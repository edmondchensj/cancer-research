# Topic Modeling for Breast Cancer Research
This project has been published in my blog [here](https://edmondchensj.github.io/2018/08/09/breast-cancer-trends/).

## Introduction
This project aims to develop a method to **quickly and comprehensively understand the landscape of breast cancer research**. This approach could be applied to other domains of research. 

While online databases such as PubMed and Google Scholar provide unprecendented access to research papers, it can be challenging to understand the major frontiers in a field of research. People who are new to a field or subfields may face a "cold-start" problem in which they lack keywords to begin researching. 

Topic Modeling is an unsupervised machine learning algorithm that discovers topics across a set of documents. The benefits of this approach are in providing:
* a comprehensive set of prominent keywords used in different subfields
* an ability to measure which subfields might be more popular
* an ability to measure which subfields might be trending.

See my [blog post](https://edmondchensj.github.io/2018/08/09/breast-cancer-trends/) for the results.

## Usage
The entire process can be broken down into 4 parts, which has to be run in correct order. 

### 1. Data Retrieval
* Run `python code/get_data.py`
* Note: By default, NCBI restricts the rate of downloads to 3 requests per second. To speed up data retrieval, register for an API key, which will increase the rate to 10 requests per second. See [here](https://www.ncbi.nlm.nih.gov/books/NBK25497/) for more information. 

### 2. Preprocessing
* Run `python code/preprocess.py`
* Note: A big factor affecting the topic model results is the list of additional stop words. You would want to customize your own list based on the domain of research. 

### 3. Build LDA Model
* Run `python code/build_models.py`

### 4. Preliminary Visualization
* This outputs a coherence graph to select the optimum number of topics and a list of wordclouds that will be required for step 5. 
* Run `python code/visualize_models.py`

### 5. Postprocessing and final Visualization
* Run `python code/postprocess.py`

## Misc
The raw data that I retrieved from PubMed is shown in the **data** folder. The **helper_files** folder contains a .png file to mask wordclouds in a circle shape. 