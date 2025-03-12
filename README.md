# The-Medi-Chat

# Predicting your prognosis based on the symptoms given: THE MediBot 

A Machine Learning Project by Armand Araujo, Misha Hedman, Sami Chowdhury, and Dhwani Patel

# Executive Summary  

Our project goal is to create a sophisticated medical symptom checker that leverages user-reported symptoms to determine potential illnesses. We created a classifier trained off a dataset of disease-symptom pairs processed using TF-IDF, and then implemented that model into a gradio app that can be interfaced via speech-to-text or traditional text. This app would then classify the prognosis based on user entered symptomsg-edge diagnostic software. 

# Table of Contents
- [Installation & Usage](#installation--usage)
    - [File Structure](#file-structure)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
- [Data Collection & Preprocessing](#data-collection--preprocessing)
    - [Initial Exploration](#initial-exploration)
    - [Preprocessing and Cleaning](#preprocessing-and-cleaning)
    - [Machine Learning Model Methodology](#machine-learning-model-methodology)
- [Limitations and Considerations](#limitations-and-considerations)
- [Summary of Findings](#summary-of-findings)
- [Potential Next Steps](#potential-next-steps)
- [Contributers](#contributors)
- [Acknowledgments](#acknowledgments)
- [Repository Structure](#repository-structure)


# Installation & Usage  


## File Structure


```
├── Resources/
├──── disease_symptoms.csv
├── .gitattributes
├── the-medichat-bot.ipynb
├── README.md
```


## Prerequisites

- Latest version of Python

- The following Python Libraries must be installed:
    1. **pandas** : needed for general data frame management
    2. **regex ("re" library)**: needed for regex operations 
    3. **numpy** : needed for numerical operations
    4. **nltk** : needed to import stop words and word tokenization
    5. **sklearn** : needed for preprocessing the data, creating the machine learning classification models, and optimizing the models
    6. **tensorflow** : needed for creating tensorflow models and processing the data
    7. **joblib** : needed for saving and loading the models
    8. **gradio**: needed for running the final application that hosts the chatbot
    9. **SpeechRecognition** : needed to interact with the gradio app through voice
    10. **gtts** : needed for text to speech for the chatbot
    11. **whisper** : load speech model for speech to text

- Git version control system

- Internet connection for data downloads


## Setup    

- Clone this repository: https://github.com/Armand57araujo/the-medi-chat.git

- Install required packages mentioned in the prerequisites

- Launch Jupyter Notebook

- Download required datasets and ensure it is in the **Resources/** directory. The data (in French) is located [at this link](https://www.kaggle.com/datasets/amaelbogne/medical-symptoms).

- Verify file integrity using provided checksums


# Data Collection & Preprocessing

## Initial Exploration



## Preprocessing and Cleaning


## Machine Learning Model Methodology

# Analysis & Results



# Challenges Encountered



# Limitations and Considerations



# Summary of Findings



# Potential Next Steps:

1. Data Expansion and Diversity:
- To make the model more resilient, use a wider variety of datasets (such as electronic health records).
- Consider merging data from multiple nations to get a more diverse range of symptom descriptions.
2. Advanced Modeling Techniques:
- Examine deep learning models (such as BERT and transformers) to improve your understanding of natural language.
- To provide a more complete diagnosis, take into account multimodal learning, which combines text, audio, and maybe visual data.
3. Clinical Integration and Real-Time Personalization:
- Create mechanisms for integrating data in real time to deliver personalized health information.
- Collaborate with medical specialists to assess clinical findings and make recommendations for diagnosis.
- Prior to implementation, address privacy, legal, and ethical issues.

# Contributors  
- [Armand Araujo](https://github.com/Armand57araujo): Project Lead, Data Sourcing, Data Cleaning, Data Preprocessing
- [Misha Hedman](https://github.com/MishaHedman) : Project Lead, Data Sourcing, Data Modeling
- [Sami Chowdhury](https://github.com/SamiC2): Data Modeling, App Building
- [Dhwani Patel](https://github.com/dhwani0619): Data Visualization


# Acknowledgments
- Data providers: 

# Repository Structure

```
├── Resources/
├──── disease_symptoms.csv
├── .gitattributes
├── the-medichat-bot.ipynb
├── README.md
```
