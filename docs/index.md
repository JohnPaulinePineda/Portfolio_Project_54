***
# Model Deployment : Estimating Lung Cancer Probabilities From Demographic Factors, Clinical Symptoms And Behavioral Indicators

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *August 31, 2024*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Predictive Model Development](#1.6)
        * [1.6.1 Pre-Modelling Data Preparation](#1.6.1)
        * [1.6.2 Data Splitting](#1.6.2)
        * [1.6.3 Modelling Pipeline Development](#1.6.3)
            * [1.6.3.1 Individual Classifier](#1.6.3.1)
            * [1.6.3.2 Stacked Classifier](#1.6.3.2)
        * [1.6.4 Model Fitting using Original Training Data | Hyperparameter Tuning | Validation](#1.6.4)
            * [1.6.4.1 Individual Classifier](#1.6.4.1)
            * [1.6.4.2 Stacked Classifier](#1.6.4.2)
        * [1.6.5 Model Fitting using Upsampled Training Data | Hyperparameter Tuning | Validation](#1.6.5)
            * [1.6.5.1 Individual Classifier](#1.6.5.1)
            * [1.6.5.2 Stacked Classifier](#1.6.5.2)
        * [1.6.6 Model Fitting using Downsampled Training Data | Hyperparameter Tuning | Validation](#1.6.6)
            * [1.6.6.1 Individual Classifier](#1.6.6.1)
            * [1.6.6.2 Stacked Classifier](#1.6.6.2)
        * [1.6.7 Model Selection](#1.6.7)
        * [1.6.8 Model Testing](#1.6.8)
        * [1.6.9 Model Inference](#1.6.9)
    * [1.7 Predictive Model Deployment Using Streamlit and Streamlit Community Cloud](#1.7)
        * [1.7.1 Model Prediction Application Code Development](#1.7.1)
        * [1.7.2 User Interface Application Code Development](#1.7.2)
        * [1.7.3 Web Application](#1.7.3)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project implements the **Logistic Regression Model** as an independent learner and as a meta-learner of a stacking ensemble model with **Decision Trees**, **Random Forest**, and **Support Vector Machine** classifier algorithms using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> to estimate probability of a dichotomous categorical response variable by modelling the relationship between one or more predictor variables and a binary outcome. The resulting predictions derived from the candidate models were evaluated using the **F1 Score** that ensures both false positives and false negatives are considered, providing a more balanced view of model classification performance. Resampling approaches including **Synthetic Minority Oversampling Technique** and **Condensed Nearest Neighbors** for imbalanced classification problems were applied by augmenting the dataset used for model training based on its inherent characteristics to achieve a more reasonably balanced distribution between the majority and minority classes. Additionally, **Class Weights** were also implemented by amplifying the loss contributed by the minority class and diminishing the loss from the majority class, forcing the model to focus more on correctly predicting the minority class. Penalties including **Least Absolute Shrinkage and Selection Operator** and **Ridge Regularization** were evaluated to impose constraints on the model coefficient updates. The final model was deployed as a prototype application with a web interface via **Streamlit**. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document. 

[Machine Learning Classification Models](http://appliedpredictivemodeling.com/) are algorithms that learn to assign predefined categories or labels to input data based on patterns and relationships identified during the training phase. Classification is a supervised learning task, meaning the models are trained on a labeled dataset where the correct output (class or label) is known for each input. Once trained, these models can predict the class of new, unseen instances.

[Binary Classification Learning](http://appliedpredictivemodeling.com/) refers to a predictive modelling problem where only two class labels are predicted for a given sample of input data. These models use the training data set and calculate how to best map instances of input data to the specific class labels. Typically, binary classification tasks involve one class that is the normal state (assigned the class label 0) and another class that is the abnormal state (assigned the class label 1). It is common to structure a binary classification task with a model that predicts a Bernoulli probability distribution for each instance. The Bernoulli distribution is a discrete probability distribution that covers a case where an event will have a binary outcome as either a 0 or 1. For a binary classification, this means that the model predicts a probability of an instance belonging to class 1, or the abnormal state.

[Imbalanced Class Learning](http://appliedpredictivemodeling.com/) refers to the process of building and training models to predict a dichotomous categorical response in scenarios where the two classes are not equally represented in the dataset. This imbalance can cause challenges in training machine learning models, leading to biased predictions that favor the majority class or misleading estimation of model performance using the accuracy metric. Several strategies can be employed to effectively handle class imbalance including resampling, class weighting, cost-sensitive learning, and choosing appropriate metrics. in effect, models can be trained to perform well on both the minority and majority classes, ensuring more reliable and fair predictions.

[Regularization Methods](http://appliedpredictivemodeling.com/), in the context of binary classification using Logistic Regression, are primarily used to prevent overfitting and improve the model's generalization to new data. Overfitting occurs when a model is too complex and learns not only the underlying pattern in the data but also the noise. This leads to poor performance on unseen data. Regularization introduces a penalty for large coefficients in the model, which helps in controlling the model complexity. In Logistic Regression, this is done by adding a regularization term to the loss function, which penalizes large values of the coefficients. This forces the model to keep the coefficients small, thereby reducing the likelihood of overfitting. Addiitonally, by penalizing the complexity of the model through the regularization term, regularization methods also help the model generalize better to unseen data. This is because the model is less likely to overfit the training data and more likely to capture the true underlying pattern.

[Streamlit](https://streamlit.io/) is an open-source Python library that simplifies the creation and deployment of web applications for machine learning and data science projects. It allows developers and data scientists to turn Python scripts into interactive web apps quickly without requiring extensive web development knowledge. Streamlit seamlessly integrates with popular Python libraries such as Pandas, Matplotlib, Plotly, and TensorFlow, allowing one to leverage existing data processing and visualization tools within the application. Streamlit apps can be easily deployed on various platforms, including Streamlit Community Cloud, Heroku, or any cloud service that supports Python web applications.

[Streamlit Community Cloud](https://streamlit.io/cloud), formerly known as Streamlit Sharing, is a free cloud-based platform provided by Streamlit that allows users to easily deploy and share Streamlit apps online. It is particularly popular among data scientists, machine learning engineers, and developers for quickly showcasing projects, creating interactive demos, and sharing data-driven applications with a wider audience without needing to manage server infrastructure. Significant features include free hosting (Streamlit Community Cloud provides free hosting for Streamlit apps, making it accessible for users who want to share their work without incurring hosting costs), easy deployment (users can connect their GitHub repository to Streamlit Community Cloud, and the app is automatically deployed from the repository), continuous deployment (if the code in the connected GitHub repository is updated, the app is automatically redeployed with the latest changes), 
sharing capabilities (once deployed, apps can be shared with others via a simple URL, making it easy for collaborators, stakeholders, or the general public to access and interact with the app), built-in authentication (users can restrict access to their apps using GitHub-based authentication, allowing control over who can view and interact with the app), and community support (the platform is supported by a community of users and developers who share knowledge, templates, and best practices for building and deploying Streamlit apps).


## 1.1. Data Background <a class="anchor" id="1.1"></a>

An open [Lung Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer/data) from [Kaggle](https://www.kaggle.com/) (with all credits attributed to [Nancy Al Aswad](https://www.kaggle.com/nancyalaswad90)) was used for the analysis as consolidated from the following primary source: 
1. Research Paper entitled **Optimal Discriminant Plane for a Small Number of Samples and Design Method of Classifier on the Plane** from the [Pattern Recognition](https://www.sciencedirect.com/science/article/abs/pii/003132039190074F) Journal

This study hypothesized that demographic factors, clinical symptoms, and behavioral indicators influence lung cancer probabilities between patients.

The dichotomous categorical variable for the study is:
* <span style="color: #FF0000">LUNG_CANCER</span> - Lung cancer status of the patient (YES, lung cancer cases | NO, non-lung cancer case)

The predictor variables for the study are:
* <span style="color: #FF0000">GENDER</span> - Patient's sex (M, Male | F, Female)
* <span style="color: #FF0000">AGE</span> - Patient's age (Years)
* <span style="color: #FF0000">SMOKING</span> - Behavioral indication of smoking (1, Absent | 2, Present)
* <span style="color: #FF0000">YELLOW_FINGERS</span> - Clinical symptom of yellowing of fingers (1, Absent | 2, Present)
* <span style="color: #FF0000">ANXIETY</span> - Behavioral indication of experiencing anxiety (1, Absent | 2, Present)
* <span style="color: #FF0000">PEER_PRESSURE</span> - Behavioral indication of experiencing peer pressure (1, Absent | 2, Present)
* <span style="color: #FF0000">CHRONIC_DISEASE</span> - Clinical symptom of chronic diseases (1, Absent | 2, Present)
* <span style="color: #FF0000">FATIGUE</span> - Clinical symptom of chronic fatigue (1, Absent | 2, Present)
* <span style="color: #FF0000">ALLERGY</span> - Clinical symptom of allergies (1, Absent | 2, Present)
* <span style="color: #FF0000">WHEEZING</span> - Clinical symptom of wheezing (1, Absent | 2, Present)
* <span style="color: #FF0000">ALCOHOL_CONSUMING</span> - Behavioral indication of consuming alcohol (1, Absent | 2, Present)
* <span style="color: #FF0000">COUGHING</span> - Clinical symptom of wheezing (1, Absent | 2, Present)
* <span style="color: #FF0000">SHORTNESS_OF_BREATH</span> - Clinical symptom of shortness of breath (1, Absent | 2, Present)
* <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span> - Clinical symptom of difficulty in swallowing (1, Absent | 2, Present)
* <span style="color: #FF0000">CHEST_PAIN</span> - Clinical symptom of chest pain (1, Absent | 2, Present)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The dataset is comprised of:
    * **309 rows** (observations)
    * **16 columns** (variables)
        * **1/16 target** (categorical)
             * <span style="color: #FF0000">LUNG_CANCER</span>
        * **1/16 predictor** (numeric)
             * <span style="color: #FF0000">AGE</span>
        * **14/16 predictors** (categorical)
             * <span style="color: #FF0000">GENDER</span>
             * <span style="color: #FF0000">SMOKING</span>
             * <span style="color: #FF0000">YELLOW_FINGERS</span>
             * <span style="color: #FF0000">ANXIETY</span>
             * <span style="color: #FF0000">PEER_PRESSURE</span>
             * <span style="color: #FF0000">CHRONIC_DISEASE</span>
             * <span style="color: #FF0000">FATIGUE</span>
             * <span style="color: #FF0000">ALLERGY</span>
             * <span style="color: #FF0000">WHEEZING</span>
             * <span style="color: #FF0000">ALCOHOL_CONSUMING </span>
             * <span style="color: #FF0000">COUGHING</span>
             * <span style="color: #FF0000">SHORTNESS_OF_BREATH</span>
             * <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span>
             * <span style="color: #FF0000">CHEST_PAIN</span>


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import itertools
import joblib
%matplotlib inline

from operator import add,mul,truediv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy import stats
from scipy.stats import pointbiserialr

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour

```


```python
##################################
# Defining file paths
##################################
DATASETS_ORIGINAL_PATH = r"datasets\original"
DATASETS_PREPROCESSED_PATH = r"datasets\preprocessed"
DATASETS_FINAL_PATH = r"datasets\final\complete"
DATASETS_FINAL_TRAIN_PATH = r"datasets\final\train"
DATASETS_FINAL_TRAIN_FEATURES_PATH = r"datasets\final\train\features"
DATASETS_FINAL_TRAIN_TARGET_PATH = r"datasets\final\train\target"
DATASETS_FINAL_VALIDATION_PATH = r"datasets\final\validation"
DATASETS_FINAL_VALIDATION_FEATURES_PATH = r"datasets\final\validation\features"
DATASETS_FINAL_VALIDATION_TARGET_PATH = r"datasets\final\validation\target"
DATASETS_FINAL_TEST_PATH = r"datasets\final\test"
DATASETS_FINAL_TEST_FEATURES_PATH = r"datasets\final\test\features"
DATASETS_FINAL_TEST_TARGET_PATH = r"datasets\final\test\target"
MODELS_PATH = r"models"
```


```python
##################################
# Loading the dataset
# from the DATASETS_ORIGINAL_PATH
##################################
lung_cancer = pd.read_csv(os.path.join("..", DATASETS_ORIGINAL_PATH, "lung_cancer.csv"))
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(lung_cancer.shape)
```

    Dataset Dimensions: 
    


    (309, 16)



```python
##################################
# Verifying the column names
##################################
print('Column Names: ')
display(lung_cancer.columns)
```

    Column Names: 
    


    Index(['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
           'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
           'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
           'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER'],
          dtype='object')



```python
##################################
# Removing trailing white spaces
# in column names
##################################
lung_cancer.columns = [x.strip() for x in lung_cancer.columns]
```


```python
##################################
# Standardizing the column names
##################################
lung_cancer.columns = ['GENDER', 
                       'AGE', 
                       'SMOKING', 
                       'YELLOW_FINGERS', 
                       'ANXIETY',
                       'PEER_PRESSURE', 
                       'CHRONIC_DISEASE', 
                       'FATIGUE', 
                       'ALLERGY', 
                       'WHEEZING',
                       'ALCOHOL_CONSUMING', 
                       'COUGHING', 
                       'SHORTNESS_OF_BREATH',
                       'SWALLOWING_DIFFICULTY', 
                       'CHEST_PAIN', 
                       'LUNG_CANCER']
```


```python
##################################
# Verifying the corrected column names
##################################
print('Column Names: ')
display(lung_cancer.columns)
```

    Column Names: 
    


    Index(['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
           'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
           'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
           'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER'],
          dtype='object')



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(lung_cancer.dtypes)
```

    Column Names and Data Types:
    


    GENDER                   object
    AGE                       int64
    SMOKING                   int64
    YELLOW_FINGERS            int64
    ANXIETY                   int64
    PEER_PRESSURE             int64
    CHRONIC_DISEASE           int64
    FATIGUE                   int64
    ALLERGY                   int64
    WHEEZING                  int64
    ALCOHOL_CONSUMING         int64
    COUGHING                  int64
    SHORTNESS_OF_BREATH       int64
    SWALLOWING_DIFFICULTY     int64
    CHEST_PAIN                int64
    LUNG_CANCER              object
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
lung_cancer.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GENDER</th>
      <th>AGE</th>
      <th>SMOKING</th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>CHRONIC_DISEASE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SHORTNESS_OF_BREATH</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
      <th>LUNG_CANCER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>69</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>74</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>59</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>63</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F</td>
      <td>63</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Setting the levels of the dichotomous categorical variables
# to boolean values
##################################
lung_cancer[['GENDER','LUNG_CANCER']] = lung_cancer[['GENDER','LUNG_CANCER']].astype('category')
lung_cancer['GENDER'] = lung_cancer['GENDER'].cat.set_categories(['F', 'M'], ordered=True)
lung_cancer['LUNG_CANCER'] = lung_cancer['LUNG_CANCER'].cat.set_categories(['NO', 'YES'], ordered=True)
int_columns = ['SMOKING',
               'YELLOW_FINGERS', 
               'ANXIETY',
               'PEER_PRESSURE', 
               'CHRONIC_DISEASE', 
               'FATIGUE', 
               'ALLERGY', 
               'WHEEZING',
               'ALCOHOL_CONSUMING', 
               'COUGHING', 
               'SHORTNESS_OF_BREATH',
               'SWALLOWING_DIFFICULTY', 
               'CHEST_PAIN', 
               'LUNG_CANCER']
lung_cancer[int_columns] = lung_cancer[int_columns].astype(object)
lung_cancer[int_columns] = lung_cancer[int_columns].replace({1: 'Absent', 2: 'Present'})
```


```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(lung_cancer.dtypes)
```

    Column Names and Data Types:
    


    GENDER                   category
    AGE                         int64
    SMOKING                    object
    YELLOW_FINGERS             object
    ANXIETY                    object
    PEER_PRESSURE              object
    CHRONIC_DISEASE            object
    FATIGUE                    object
    ALLERGY                    object
    WHEEZING                   object
    ALCOHOL_CONSUMING          object
    COUGHING                   object
    SHORTNESS_OF_BREATH        object
    SWALLOWING_DIFFICULTY      object
    CHEST_PAIN                 object
    LUNG_CANCER                object
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
lung_cancer.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GENDER</th>
      <th>AGE</th>
      <th>SMOKING</th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>CHRONIC_DISEASE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SHORTNESS_OF_BREATH</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
      <th>LUNG_CANCER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>69</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>74</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>59</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>63</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F</td>
      <td>63</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Performing a general exploration 
# of the numeric variables
##################################
print('Numeric Variable Summary:')
display(lung_cancer.describe(include='number').transpose())
```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AGE</th>
      <td>309.0</td>
      <td>62.673139</td>
      <td>8.210301</td>
      <td>21.0</td>
      <td>57.0</td>
      <td>62.0</td>
      <td>69.0</td>
      <td>87.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration 
# of the object and categorical variables
##################################
print('Categorical Variable Summary:')
display(lung_cancer.describe(include=['category','object']).transpose())
```

    Categorical Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GENDER</th>
      <td>309</td>
      <td>2</td>
      <td>M</td>
      <td>162</td>
    </tr>
    <tr>
      <th>SMOKING</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>174</td>
    </tr>
    <tr>
      <th>YELLOW_FINGERS</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>176</td>
    </tr>
    <tr>
      <th>ANXIETY</th>
      <td>309</td>
      <td>2</td>
      <td>Absent</td>
      <td>155</td>
    </tr>
    <tr>
      <th>PEER_PRESSURE</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>155</td>
    </tr>
    <tr>
      <th>CHRONIC_DISEASE</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>156</td>
    </tr>
    <tr>
      <th>FATIGUE</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>208</td>
    </tr>
    <tr>
      <th>ALLERGY</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>172</td>
    </tr>
    <tr>
      <th>WHEEZING</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>172</td>
    </tr>
    <tr>
      <th>ALCOHOL_CONSUMING</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>172</td>
    </tr>
    <tr>
      <th>COUGHING</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>179</td>
    </tr>
    <tr>
      <th>SHORTNESS_OF_BREATH</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>198</td>
    </tr>
    <tr>
      <th>SWALLOWING_DIFFICULTY</th>
      <td>309</td>
      <td>2</td>
      <td>Absent</td>
      <td>164</td>
    </tr>
    <tr>
      <th>CHEST_PAIN</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>172</td>
    </tr>
    <tr>
      <th>LUNG_CANCER</th>
      <td>309</td>
      <td>2</td>
      <td>YES</td>
      <td>270</td>
    </tr>
  </tbody>
</table>
</div>


## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. 33 duplicated rows observed. These cases were not removed considering that most variables are dichotomous categorical where duplicate values might be possible.
2. No missing data noted for any variable with Null.Count>0 and Fill.Rate<1.0.
3. No low variance observed for the numeric predictor with First.Second.Mode.Ratio>5.
4. No low variance observed for the numeric and categorical predictors with Unique.Count.Ratio>5.
5. Low variance observed for the target variable with Unique.Count.Ratio>5 indicating **class imbalance** that needs to be addressed for the downstream modelling process.
    * <span style="color: #FF0000">LUNG_CANCER</span>: Unique.Count.Ratio = +6.923
6. No high skewness observed for the numeric predictor with Skewness>3 or Skewness<(-3).


```python
##################################
# Counting the number of duplicated rows
##################################
lung_cancer.duplicated().sum()
```




    np.int64(33)




```python
##################################
# Displaying the duplicated rows
##################################
lung_cancer[lung_cancer.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GENDER</th>
      <th>AGE</th>
      <th>SMOKING</th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>CHRONIC_DISEASE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SHORTNESS_OF_BREATH</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
      <th>LUNG_CANCER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99</th>
      <td>M</td>
      <td>56</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>100</th>
      <td>M</td>
      <td>58</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>117</th>
      <td>F</td>
      <td>51</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>199</th>
      <td>F</td>
      <td>55</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>212</th>
      <td>M</td>
      <td>58</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>223</th>
      <td>M</td>
      <td>63</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>256</th>
      <td>M</td>
      <td>60</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>275</th>
      <td>M</td>
      <td>64</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>284</th>
      <td>M</td>
      <td>58</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>285</th>
      <td>F</td>
      <td>58</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>286</th>
      <td>F</td>
      <td>63</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>287</th>
      <td>F</td>
      <td>51</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>288</th>
      <td>F</td>
      <td>61</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>289</th>
      <td>F</td>
      <td>61</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>290</th>
      <td>M</td>
      <td>76</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>291</th>
      <td>M</td>
      <td>71</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>292</th>
      <td>M</td>
      <td>69</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>293</th>
      <td>F</td>
      <td>56</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>294</th>
      <td>M</td>
      <td>67</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>295</th>
      <td>F</td>
      <td>54</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>296</th>
      <td>M</td>
      <td>63</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>297</th>
      <td>F</td>
      <td>47</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>298</th>
      <td>M</td>
      <td>62</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>299</th>
      <td>M</td>
      <td>65</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>300</th>
      <td>F</td>
      <td>63</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>301</th>
      <td>M</td>
      <td>64</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>302</th>
      <td>F</td>
      <td>65</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>303</th>
      <td>M</td>
      <td>51</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>304</th>
      <td>F</td>
      <td>56</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>305</th>
      <td>M</td>
      <td>70</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>306</th>
      <td>M</td>
      <td>58</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>307</th>
      <td>M</td>
      <td>67</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>308</th>
      <td>M</td>
      <td>62</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>YES</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(lung_cancer.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(lung_cancer.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(lung_cancer)] * len(lung_cancer.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(lung_cancer.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(lung_cancer.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GENDER</td>
      <td>category</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AGE</td>
      <td>int64</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SMOKING</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>YELLOW_FINGERS</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ANXIETY</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PEER_PRESSURE</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CHRONIC_DISEASE</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>FATIGUE</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ALLERGY</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>WHEEZING</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ALCOHOL_CONSUMING</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>COUGHING</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SHORTNESS_OF_BREATH</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SWALLOWING_DIFFICULTY</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CHEST_PAIN</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LUNG_CANCER</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
print('Number of Columns with Missing Data:', str(len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])))
```

    Number of Columns with Missing Data: 0
    


```python
##################################
# Identifying the rows
# with Fill.Rate < 1.00
##################################
column_low_fill_rate = all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1.00)]
```


```python
##################################
# Gathering the metadata labels for each observation
##################################
row_metadata_list = lung_cancer.index.values.tolist()
```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(lung_cancer.columns)] * len(lung_cancer))
```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(lung_cancer.isna().sum(axis=1))
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)
```


```python
##################################
# Exploring the rows
# for missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_metadata_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>16</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>16</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>16</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>16</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>16</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>304</th>
      <td>304</td>
      <td>16</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>305</th>
      <td>305</td>
      <td>16</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>306</th>
      <td>306</td>
      <td>16</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>307</th>
      <td>307</td>
      <td>16</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>308</th>
      <td>308</td>
      <td>16</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>309 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Fill.Rate < 1.00
##################################
print('Number of Rows with Missing Data:',str(len(all_row_quality_summary[all_row_quality_summary['Missing.Rate']>0])))
```

    Number of Rows with Missing Data: 0
    


```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
lung_cancer_numeric = lung_cancer.select_dtypes(include=['number','int'])
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = lung_cancer_numeric.columns
```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = lung_cancer_numeric.min()
```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = lung_cancer_numeric.mean()
```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = lung_cancer_numeric.median()
```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = lung_cancer_numeric.max()
```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [lung_cancer[x].value_counts(dropna=True).index.tolist()[0] for x in lung_cancer_numeric]
```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [lung_cancer[x].value_counts(dropna=True).index.tolist()[1] for x in lung_cancer_numeric]
```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [lung_cancer_numeric[x].isin([lung_cancer[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in lung_cancer_numeric]
```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [lung_cancer_numeric[x].isin([lung_cancer[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in lung_cancer_numeric]
```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = lung_cancer_numeric.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(lung_cancer_numeric)] * len(lung_cancer_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = lung_cancer_numeric.skew()
```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = lung_cancer_numeric.kurtosis()
```


```python
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGE</td>
      <td>21</td>
      <td>62.673139</td>
      <td>62.0</td>
      <td>87</td>
      <td>64</td>
      <td>56</td>
      <td>20</td>
      <td>19</td>
      <td>1.052632</td>
      <td>39</td>
      <td>309</td>
      <td>0.126214</td>
      <td>-0.395086</td>
      <td>1.746558</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    0




```python
##################################
# Counting the number of numeric columns
# with Unique.Count.Ratio > 10.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0




```python
##################################
# Counting the number of numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))])
```




    0




```python
##################################
# Formulating the dataset
# with object or categorical column only
##################################
lung_cancer_object = lung_cancer.select_dtypes(include=['object','category'])
```


```python
##################################
# Gathering the variable names for the object or categorical column
##################################
categorical_variable_name_list = lung_cancer_object.columns
```


```python
##################################
# Gathering the first mode values for the object or categorical column
##################################
categorical_first_mode_list = [lung_cancer[x].value_counts().index.tolist()[0] for x in lung_cancer_object]
```


```python
##################################
# Gathering the second mode values for each object or categorical column
##################################
categorical_second_mode_list = [lung_cancer[x].value_counts().index.tolist()[1] for x in lung_cancer_object]
```


```python
##################################
# Gathering the count of first mode values for each object or categorical column
##################################
categorical_first_mode_count_list = [lung_cancer_object[x].isin([lung_cancer[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in lung_cancer_object]
```


```python
##################################
# Gathering the count of second mode values for each object or categorical column
##################################
categorical_second_mode_count_list = [lung_cancer_object[x].isin([lung_cancer[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in lung_cancer_object]
```


```python
##################################
# Gathering the first mode to second mode ratio for each object or categorical column
##################################
categorical_first_second_mode_ratio_list = map(truediv, categorical_first_mode_count_list, categorical_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each object or categorical column
##################################
categorical_unique_count_list = lung_cancer_object.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each object or categorical column
##################################
categorical_row_count_list = list([len(lung_cancer_object)] * len(lung_cancer_object.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object or categorical column
##################################
categorical_unique_count_ratio_list = map(truediv, categorical_unique_count_list, categorical_row_count_list)
```


```python
categorical_column_quality_summary = pd.DataFrame(zip(categorical_variable_name_list,
                                                 categorical_first_mode_list,
                                                 categorical_second_mode_list,
                                                 categorical_first_mode_count_list,
                                                 categorical_second_mode_count_list,
                                                 categorical_first_second_mode_ratio_list,
                                                 categorical_unique_count_list,
                                                 categorical_row_count_list,
                                                 categorical_unique_count_ratio_list), 
                                        columns=['Categorical.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(categorical_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GENDER</td>
      <td>M</td>
      <td>F</td>
      <td>162</td>
      <td>147</td>
      <td>1.102041</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SMOKING</td>
      <td>Present</td>
      <td>Absent</td>
      <td>174</td>
      <td>135</td>
      <td>1.288889</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YELLOW_FINGERS</td>
      <td>Present</td>
      <td>Absent</td>
      <td>176</td>
      <td>133</td>
      <td>1.323308</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ANXIETY</td>
      <td>Absent</td>
      <td>Present</td>
      <td>155</td>
      <td>154</td>
      <td>1.006494</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PEER_PRESSURE</td>
      <td>Present</td>
      <td>Absent</td>
      <td>155</td>
      <td>154</td>
      <td>1.006494</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CHRONIC_DISEASE</td>
      <td>Present</td>
      <td>Absent</td>
      <td>156</td>
      <td>153</td>
      <td>1.019608</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>6</th>
      <td>FATIGUE</td>
      <td>Present</td>
      <td>Absent</td>
      <td>208</td>
      <td>101</td>
      <td>2.059406</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ALLERGY</td>
      <td>Present</td>
      <td>Absent</td>
      <td>172</td>
      <td>137</td>
      <td>1.255474</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WHEEZING</td>
      <td>Present</td>
      <td>Absent</td>
      <td>172</td>
      <td>137</td>
      <td>1.255474</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ALCOHOL_CONSUMING</td>
      <td>Present</td>
      <td>Absent</td>
      <td>172</td>
      <td>137</td>
      <td>1.255474</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>10</th>
      <td>COUGHING</td>
      <td>Present</td>
      <td>Absent</td>
      <td>179</td>
      <td>130</td>
      <td>1.376923</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SHORTNESS_OF_BREATH</td>
      <td>Present</td>
      <td>Absent</td>
      <td>198</td>
      <td>111</td>
      <td>1.783784</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SWALLOWING_DIFFICULTY</td>
      <td>Absent</td>
      <td>Present</td>
      <td>164</td>
      <td>145</td>
      <td>1.131034</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CHEST_PAIN</td>
      <td>Present</td>
      <td>Absent</td>
      <td>172</td>
      <td>137</td>
      <td>1.255474</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LUNG_CANCER</td>
      <td>YES</td>
      <td>NO</td>
      <td>270</td>
      <td>39</td>
      <td>6.923077</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of object or categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    1




```python
##################################
# Identifying the object or categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
display(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)].sort_values(by=['First.Second.Mode.Ratio'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>LUNG_CANCER</td>
      <td>YES</td>
      <td>NO</td>
      <td>270</td>
      <td>39</td>
      <td>6.923077</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of object or categorical columns
# with Unique.Count.Ratio > 10.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>

1. No data transformation and scaling applied to the numeric predictor due to the minimal number of outliers and normal skewness values.
2. All dichotomous categorical predictors were one-hot encoded for the correlation analysis process.
3. All variables were retained since majority reported sufficiently moderate correlation with no excessive multicollinearity.
    * Minimal correlation observed between the predictors using the point-biserial coefficient for evaluating numeric and dichotomous categorical variables.
    * Minimal correlation observed between the predictors using the phi coefficient for evaluating both dichotomous categorical variables.
4. Among pairwise combinations of variables in the training subset, sufficiently high correlation values were observed but with no excessive multicollinearity noted:
    * <span style="color: #FF0000">ANXIETY</span> and <span style="color: #FF0000">YELLOW_FINGERS</span>: Phi.Coefficient = +0.570
    * <span style="color: #FF0000">ANXIETY</span> and <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span>: Phi.Coefficient = +0.490
    * <span style="color: #FF0000">SHORTNESS_OF_BREATH</span> and <span style="color: #FF0000">FATIGUE</span>: Phi.Coefficient = +0.440
    * <span style="color: #FF0000">COUGHING</span> and <span style="color: #FF0000">WHEEZING</span>: Phi.Coefficient = +0.370
    * <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span> and <span style="color: #FF0000">PEER_PRESSURE</span>: Phi.Coefficient = +0.370



```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
lung_cancer_numeric = lung_cancer.select_dtypes(include=['number','int'])
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = lung_cancer_numeric.columns
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = lung_cancer_numeric.skew()
```


```python
##################################
# Computing the interquartile range
# for all columns
##################################
lung_cancer_numeric_q1 = lung_cancer_numeric.quantile(0.25)
lung_cancer_numeric_q3 = lung_cancer_numeric.quantile(0.75)
lung_cancer_numeric_iqr = lung_cancer_numeric_q3 - lung_cancer_numeric_q1
```


```python
##################################
# Gathering the outlier count for each numeric column
# based on the interquartile range criterion
##################################
numeric_outlier_count_list = ((lung_cancer_numeric < (lung_cancer_numeric_q1 - 1.5 * lung_cancer_numeric_iqr)) | (lung_cancer_numeric > (lung_cancer_numeric_q3 + 1.5 * lung_cancer_numeric_iqr))).sum()
```


```python
##################################
# Gathering the number of observations for each column
##################################
numeric_row_count_list = list([len(lung_cancer_numeric)] * len(lung_cancer_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
numeric_outlier_ratio_list = map(truediv, numeric_outlier_count_list, numeric_row_count_list)
```


```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
numeric_column_outlier_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                  numeric_skewness_list,
                                                  numeric_outlier_count_list,
                                                  numeric_row_count_list,
                                                  numeric_outlier_ratio_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(numeric_column_outlier_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGE</td>
      <td>-0.395086</td>
      <td>2</td>
      <td>309</td>
      <td>0.006472</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the individual boxplots
# for all numeric columns
##################################
for column in lung_cancer_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=lung_cancer_numeric, x=column)
```


    
![png](output_82_0.png)
    



```python
##################################
# Creating a dataset copy and
# converting all values to numeric (integer or float)
# for correlation analysis
##################################
pd.set_option('future.no_silent_downcasting', True)
lung_cancer_correlation = lung_cancer.copy()
lung_cancer_correlation_object = lung_cancer_correlation.iloc[:,2:15].columns
lung_cancer_correlation[lung_cancer_correlation_object] = lung_cancer_correlation[lung_cancer_correlation_object].replace({'Absent': 0, 'Present': 1})
lung_cancer_correlation = lung_cancer_correlation.drop(['GENDER','LUNG_CANCER'], axis=1)
lung_cancer_correlation['AGE'] = lung_cancer_correlation['AGE'].astype(float)
object_cols_to_convert = lung_cancer_correlation.columns[1:]
for col in object_cols_to_convert:
    if lung_cancer_correlation[col].dtype == 'object':
        lung_cancer_correlation[col] = lung_cancer_correlation[col].astype(int)
display(lung_cancer_correlation)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>SMOKING</th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>CHRONIC_DISEASE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SHORTNESS_OF_BREATH</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>74.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>59.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>63.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>63.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>304</th>
      <td>56.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>305</th>
      <td>70.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>306</th>
      <td>58.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>307</th>
      <td>67.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>308</th>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>309 rows × 14 columns</p>
</div>



```python
##################################
# Initializing the correlation matrix
##################################
lung_cancer_correlation_matrix = pd.DataFrame(np.zeros((len(lung_cancer_correlation.columns), len(lung_cancer_correlation.columns))),
                                              columns=lung_cancer_correlation.columns,
                                              index=lung_cancer_correlation.columns)
```


```python
##################################
# Calculating different types
# of correlation coefficients
# per variable type
##################################
for i in range(len(lung_cancer_correlation.columns)):
    for j in range(i, len(lung_cancer_correlation.columns)):
        if i == j:
            lung_cancer_correlation_matrix.iloc[i, j] = 1.0
        else:
            if lung_cancer_correlation.dtypes.iloc[i] == 'int64' and lung_cancer_correlation.dtypes.iloc[j] == 'int64':
                # Phi coefficient for two binary variables
                corr = lung_cancer_correlation.iloc[:, i].corr(lung_cancer_correlation.iloc[:, j])
            elif lung_cancer_correlation.dtypes.iloc[i] == 'float64' or lung_cancer_correlation.dtypes.iloc[j] == 'int64':
                # Point-biserial correlation for one continuous and one binary variable
                continuous_var = lung_cancer_correlation.iloc[:, i] if lung_cancer_correlation.dtypes.iloc[i] == 'float64' else lung_cancer_correlation.iloc[:, j]
                binary_var = lung_cancer_correlation.iloc[:, j] if lung_cancer_correlation.dtypes.iloc[j] == 'int64' else lung_cancer_correlation.iloc[:, i]
                corr, _ = pointbiserialr(continuous_var, binary_var)
            else:
                # Pearson correlation for two continuous variables
                corr = lung_cancer_correlation.iloc[:, i].corr(lung_cancer_correlation.iloc[:, j])
            lung_cancer_correlation_matrix.iloc[i, j] = corr
            lung_cancer_correlation_matrix.iloc[j, i] = corr

```


```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric and categorical columns
##################################
plt.figure(figsize=(17, 8))
sns.heatmap(lung_cancer_correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()
```


    
![png](output_86_0.png)
    


## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. The lung cancer prevalence estimated for the overall dataset is 87.38%, indicating **class imbalance**.
2. Higher counts for the following categorical predictors are associated with better differentiation between <span style="color: #FF0000">LUNG_CANCER=Yes</span> and <span style="color: #FF0000">LUNG_CANCER=No</span>: 
    * <span style="color: #FF0000">YELLOW_FINGERS</span>
    * <span style="color: #FF0000">ANXIETY</span>
    * <span style="color: #FF0000">PEER_PRESSURE</span>
    * <span style="color: #FF0000">CHRONIC_DISEASE</span>
    * <span style="color: #FF0000">FATIGUE</span>
    * <span style="color: #FF0000">ALLERGY</span>
    * <span style="color: #FF0000">WHEEZING</span>
    * <span style="color: #FF0000">ALCOHOL_CONSUMING </span>
    * <span style="color: #FF0000">COUGHING</span>
    * <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span>
    * <span style="color: #FF0000">CHEST_PAIN</span> 
        


```python
##################################
# Estimating the lung cancer prevalence
##################################
print('Lung Cancer Prevalence: ')
display(lung_cancer['LUNG_CANCER'].value_counts(normalize = True))
```

    Lung Cancer Prevalence: 
    


    LUNG_CANCER
    YES    0.873786
    NO     0.126214
    Name: proportion, dtype: float64



```python
##################################
# Segregating the target
# and predictor variables
##################################
lung_cancer_predictors = lung_cancer.iloc[:,:-1].columns
lung_cancer_predictors_numeric = lung_cancer.iloc[:,:-1].loc[:,lung_cancer.iloc[:,:-1].columns == 'AGE'].columns
lung_cancer_predictors_categorical = lung_cancer.iloc[:,:-1].loc[:,lung_cancer.iloc[:,:-1].columns != 'AGE'].columns
```


```python
##################################
# Segregating the target variable
# and numeric predictors
##################################
boxplot_y_variable = 'LUNG_CANCER'
boxplot_x_variable = lung_cancer_predictors_numeric.values[0]
```


```python
##################################
# Evaluating the numeric predictors
# against the target variable
##################################
plt.figure(figsize=(7, 5))
plt.boxplot([group[boxplot_x_variable] for name, group in lung_cancer.groupby(boxplot_y_variable, observed=True)])
plt.title(f'{boxplot_y_variable} Versus {boxplot_x_variable}')
plt.xlabel(boxplot_y_variable)
plt.ylabel(boxplot_x_variable)
plt.xticks(range(1, len(lung_cancer[boxplot_y_variable].unique()) + 1), ['No', 'Yes'])
plt.show()
```


    
![png](output_92_0.png)
    



```python
##################################
# Segregating the target variable
# and categorical predictors
##################################
proportion_y_variables = lung_cancer_predictors_categorical
proportion_x_variable = 'LUNG_CANCER'
```


```python
##################################
# Defining the number of 
# rows and columns for the subplots
##################################
num_rows = 7
num_cols = 2

##################################
# Formulating the subplot structure
##################################
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 40))

##################################
# Flattening the multi-row and
# multi-column axes
##################################
axes = axes.ravel()

##################################
# Formulating the individual stacked column plots
# for all categorical columns
##################################
for i, y_variable in enumerate(proportion_y_variables):
    ax = axes[i]
    category_counts = lung_cancer.groupby([proportion_x_variable, y_variable], observed=True).size().unstack(fill_value=0)
    category_proportions = category_counts.div(category_counts.sum(axis=1), axis=0)
    category_proportions.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'{proportion_x_variable} Versus {y_variable}')
    ax.set_xlabel(proportion_x_variable)
    ax.set_ylabel('PROPORTIONS')
    ax.legend(loc="lower center")

##################################
# Adjusting the subplot layout
##################################
plt.tight_layout()

##################################
# Presenting the subplots
##################################
plt.show()
```


    
![png](output_94_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

1. The relationship between the numeric predictor to the <span style="color: #FF0000">LUNG_CANCER</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: Difference in the means between groups Yes and No is equal to zero  
    * **Alternative**: Difference in the means between groups Yes and No is not equal to zero   
2. There is no sufficient evidence to conclude of a statistically significant difference between the means of the numeric measurements obtained from the <span style="color: #FF0000">LUNG_CANCER</span> groups in 1 numeric predictor given its low t-test statistic value with reported high p-value above the significance level of 0.05.
    * <span style="color: #FF0000">AGE</span>: T.Test.Statistic=-1.574, T.Test.PValue=0.116
3. The relationship between the categorical predictors to the <span style="color: #FF0000">LUNG_CANCER</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: The categorical predictor is independent of the target variable 
    * **Alternative**: The categorical predictor is dependent on the target variable   
4. There is sufficient evidence to conclude of a statistically significant relationship between the individual categories and the <span style="color: #FF0000">LUNG_CANCER</span> groups in 9 categorical predictors given their high chisquare statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">ALLERGY</span>: ChiSquare.Test.Statistic=31.238, ChiSquare.Test.PValue=0.000
    * <span style="color: #FF0000">ALCOHOL_CONSUMING</span>: ChiSquare.Test.Statistic=24.005, ChiSquare.Test.PValue=0.000   
    * <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span>: ChiSquare.Test.Statistic=19.307, ChiSquare.Test.PValue=0.000 
    * <span style="color: #FF0000">WHEEZING</span>: ChiSquare.Test.Statistic=17.723, ChiSquare.Test.PValue=0.000
    * <span style="color: #FF0000">COUGHING</span>: ChiSquare.Test.Statistic=17.606, ChiSquare.Test.PValue=0.000
    * <span style="color: #FF0000">CHEST_PAIN</span>: ChiSquare.Test.Statistic=10.083, ChiSquare.Test.PValue=0.001   
    * <span style="color: #FF0000">PEER_PRESSURE</span>: ChiSquare.Test.Statistic=9.641, ChiSquare.Test.PValue=0.001 
    * <span style="color: #FF0000">YELLOW_FINGERS</span>: ChiSquare.Test.Statistic=9.088, ChiSquare.Test.PValue=0.002
    * <span style="color: #FF0000">FATIGUE</span>: ChiSquare.Test.Statistic=6.081, ChiSquare.Test.PValue=0.013 
    * <span style="color: #FF0000">ANXIETY</span>: ChiSquare.Test.Statistic=5.648, ChiSquare.Test.PValue=0.017 
5. There is no sufficient evidence to conclude of a statistically significant relationship between the individual categories and the <span style="color: #FF0000">LUNG_CANCER</span> groups in 4 categorical predictors given their low chisquare statistic values with reported high p-values greater than the significance level of 0.05.    
    * <span style="color: #FF0000">CHRONIC_DISEASE</span>: ChiSquare.Test.Statistic=3.161, ChiSquare.Test.PValue=0.075
    * <span style="color: #FF0000">GENDER</span>: ChiSquare.Test.Statistic=1.021, ChiSquare.Test.PValue=0.312
    * <span style="color: #FF0000">SHORTNESS_OF_BREATH</span>: ChiSquare.Test.Statistic=0.790, ChiSquare.Test.PValue=0.373   
    * <span style="color: #FF0000">SMOKING</span>: ChiSquare.Test.Statistic=0.722, ChiSquare.Test.PValue=0.395


```python
##################################
# Computing the t-test 
# statistic and p-values
# between the target variable
# and numeric predictor columns
##################################
lung_cancer_numeric_ttest_target = {}
lung_cancer_numeric = lung_cancer.loc[:,(lung_cancer.columns == 'AGE') | (lung_cancer.columns == 'LUNG_CANCER')]
lung_cancer_numeric_columns = lung_cancer_predictors_numeric
for numeric_column in lung_cancer_numeric_columns:
    group_0 = lung_cancer_numeric[lung_cancer_numeric.loc[:,'LUNG_CANCER']=='NO']
    group_1 = lung_cancer_numeric[lung_cancer_numeric.loc[:,'LUNG_CANCER']=='YES']
    lung_cancer_numeric_ttest_target['LUNG_CANCER_' + numeric_column] = stats.ttest_ind(
        group_0[numeric_column], 
        group_1[numeric_column], 
        equal_var=True)
```


```python
##################################
# Formulating the pairwise ttest summary
# between the target variable
# and numeric predictor columns
##################################
lung_cancer_numeric_summary = lung_cancer_numeric.from_dict(lung_cancer_numeric_ttest_target, orient='index')
lung_cancer_numeric_summary.columns = ['T.Test.Statistic', 'T.Test.PValue']
display(lung_cancer_numeric_summary.sort_values(by=['T.Test.PValue'], ascending=True).head(len(lung_cancer_predictors_numeric)))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T.Test.Statistic</th>
      <th>T.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LUNG_CANCER_AGE</th>
      <td>-1.573857</td>
      <td>0.11655</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Computing the chisquare
# statistic and p-values
# between the target variable
# and categorical predictor columns
##################################
lung_cancer_categorical_chisquare_target = {}
lung_cancer_categorical = lung_cancer.loc[:,(lung_cancer.columns != 'AGE') | (lung_cancer.columns == 'LUNG_CANCER')]
lung_cancer_categorical_columns = lung_cancer_predictors_categorical
for categorical_column in lung_cancer_categorical_columns:
    contingency_table = pd.crosstab(lung_cancer_categorical[categorical_column], 
                                    lung_cancer_categorical['LUNG_CANCER'])
    lung_cancer_categorical_chisquare_target['LUNG_CANCER_' + categorical_column] = stats.chi2_contingency(
        contingency_table)[0:2]
```


```python
##################################
# Formulating the pairwise chisquare summary
# between the target variable
# and categorical predictor columns
##################################
lung_cancer_categorical_summary = lung_cancer_categorical.from_dict(lung_cancer_categorical_chisquare_target, orient='index')
lung_cancer_categorical_summary.columns = ['ChiSquare.Test.Statistic', 'ChiSquare.Test.PValue']
display(lung_cancer_categorical_summary.sort_values(by=['ChiSquare.Test.PValue'], ascending=True).head(len(lung_cancer_predictors_categorical)))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ChiSquare.Test.Statistic</th>
      <th>ChiSquare.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LUNG_CANCER_ALLERGY</th>
      <td>31.238952</td>
      <td>2.281422e-08</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_ALCOHOL_CONSUMING</th>
      <td>24.005406</td>
      <td>9.606559e-07</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_SWALLOWING_DIFFICULTY</th>
      <td>19.307277</td>
      <td>1.112814e-05</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_WHEEZING</th>
      <td>17.723096</td>
      <td>2.555055e-05</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_COUGHING</th>
      <td>17.606122</td>
      <td>2.717123e-05</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_CHEST_PAIN</th>
      <td>10.083198</td>
      <td>1.496275e-03</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_PEER_PRESSURE</th>
      <td>9.641594</td>
      <td>1.902201e-03</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_YELLOW_FINGERS</th>
      <td>9.088186</td>
      <td>2.572659e-03</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_FATIGUE</th>
      <td>6.081100</td>
      <td>1.366356e-02</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_ANXIETY</th>
      <td>5.648390</td>
      <td>1.747141e-02</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_CHRONIC_DISEASE</th>
      <td>3.161200</td>
      <td>7.540772e-02</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_GENDER</th>
      <td>1.021545</td>
      <td>3.121527e-01</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_SHORTNESS_OF_BREATH</th>
      <td>0.790604</td>
      <td>3.739175e-01</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_SMOKING</th>
      <td>0.722513</td>
      <td>3.953209e-01</td>
    </tr>
  </tbody>
</table>
</div>


## 1.6. Predictive Model Development <a class="anchor" id="1.6"></a>

### 1.6.1 Pre-Modelling Data Preparation <a class="anchor" id="1.6.1"></a>

1. All dichotomous categorical predictors and the target variable were one-hot encoded for the downstream modelling process. 
2. Predictors determined with insufficient association with the <span style="color: #FF0000">LUNG_CANCER</span> target variables were excluded for the subsequent modelling steps.
    * <span style="color: #FF0000">AGE</span>: T.Test.Statistic=-1.574, T.Test.PValue=0.116
    * <span style="color: #FF0000">CHRONIC_DISEASE</span>: ChiSquare.Test.Statistic=3.161, ChiSquare.Test.PValue=0.075
    * <span style="color: #FF0000">GENDER</span>: ChiSquare.Test.Statistic=1.021, ChiSquare.Test.PValue=0.312
    * <span style="color: #FF0000">SHORTNESS_OF_BREATH</span>: ChiSquare.Test.Statistic=0.790, ChiSquare.Test.PValue=0.373   
    * <span style="color: #FF0000">SMOKING</span>: ChiSquare.Test.Statistic=0.722, ChiSquare.Test.PValue=0.395


```python
##################################
# Creating a dataset copy and
# transforming all values to numeric
# prior to data splitting and modelling
##################################
pd.set_option('future.no_silent_downcasting', True)
lung_cancer_transformed = lung_cancer.copy()
lung_cancer_transformed_object = lung_cancer_transformed.iloc[:,2:15].columns
lung_cancer_transformed['GENDER'] = lung_cancer_transformed['GENDER'].astype('category')
lung_cancer_transformed['GENDER'] = lung_cancer_transformed['GENDER'].cat.rename_categories({'F': 0, 'M': 1})
lung_cancer_transformed['LUNG_CANCER'] = lung_cancer_transformed['LUNG_CANCER'].astype('category')
lung_cancer_transformed['LUNG_CANCER'] = lung_cancer_transformed['LUNG_CANCER'].cat.rename_categories({'NO': 0, 'YES': 1})
lung_cancer_transformed[lung_cancer_transformed_object] = lung_cancer_transformed[lung_cancer_transformed_object].replace({'Absent': 0, 'Present': 1})
display(lung_cancer_transformed)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GENDER</th>
      <th>AGE</th>
      <th>SMOKING</th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>CHRONIC_DISEASE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SHORTNESS_OF_BREATH</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
      <th>LUNG_CANCER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>69</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>74</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>59</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>63</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>63</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>304</th>
      <td>0</td>
      <td>56</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>305</th>
      <td>1</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>306</th>
      <td>1</td>
      <td>58</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>307</th>
      <td>1</td>
      <td>67</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>308</th>
      <td>1</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>309 rows × 16 columns</p>
</div>



```python
##################################
# Saving the tranformed data
# to the DATASETS_PREPROCESSED_PATH
##################################
lung_cancer_transformed.to_csv(os.path.join("..", DATASETS_PREPROCESSED_PATH, "lung_cancer_transformed.csv"), index=False)
```


```python
##################################
# Filtering out predictors that did not exhibit 
# sufficient discrimination of the target variable
# Saving the tranformed data
# to the DATASETS_PREPROCESSED_PATH
##################################
lung_cancer_filtered = lung_cancer_transformed.drop(['GENDER','CHRONIC_DISEASE', 'SHORTNESS_OF_BREATH', 'SMOKING', 'AGE'], axis=1)
lung_cancer_filtered.to_csv(os.path.join("..", DATASETS_FINAL_PATH, "lung_cancer_final.csv"), index=False)
display(lung_cancer_filtered)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
      <th>LUNG_CANCER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>304</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>305</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>306</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>307</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>308</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>309 rows × 11 columns</p>
</div>


### 1.6.2 Data Splitting <a class="anchor" id="1.6.2"></a>

1. The preprocessed dataset was divided into three subsets using a fixed random seed:
    * **test data**: 25% of the original data with class stratification applied
    * **train data (initial)**: 75% of the original data with class stratification applied
        * **train data (final)**: 75% of the **train (initial)** data with class stratification applied
        * **validation data**: 25% of the **train (initial)** data with class stratification applied
2. Resampling (upsampling and downsampling) algorithms were applied on the **train data (final)** to evaluate the effects of remedial actions against class imbalance. 
3. Models were developed from the original, upsampled and downsampled **train data (final)**. Using the same dataset, a subset of models with optimal hyperparameters were selected, based on cross-validation.
4. Among candidate models with optimal hyperparameters, the final model were selected based on performance on the **validation data**. 
5. Performance of the selected final model (and other candidate models for post-model selection comparison) were evaluated using the **test data**. 
6. The preprocessed data is comprised of:
    * **309 rows** (observations)
        * **270 LUNG_CANCER=Yes**: 87.38%
        * **39 LUNG_CANCER=No**: 12.82%
    * **11 columns** (variables)
        * **1/11 target** (categorical)
             * <span style="color: #FF0000">LUNG_CANCER</span>
        * **10/11 predictors** (categorical)
             * <span style="color: #FF0000">YELLOW_FINGERS</span>
             * <span style="color: #FF0000">ANXIETY</span>
             * <span style="color: #FF0000">PEER_PRESSURE</span>
             * <span style="color: #FF0000">FATIGUE</span>
             * <span style="color: #FF0000">ALLERGY</span>
             * <span style="color: #FF0000">WHEEZING</span>
             * <span style="color: #FF0000">ALCOHOL_CONSUMING </span>
             * <span style="color: #FF0000">COUGHING</span>
             * <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span>
             * <span style="color: #FF0000">CHEST_PAIN</span>
7. The **train data (final)** subset is comprised of:
    * **173 rows** (observations)
        * **151 LUNG_CANCER=Yes**: 87.28%
        * **22 LUNG_CANCER=No**: 12.72%
    * **11 columns** (variables)
8. The **validation data** subset is comprised of:
    * **58 rows** (observations)
        * **51 LUNG_CANCER=Yes**: 87.93%
        * **7 LUNG_CANCER=No**: 12.07%
    * **11 columns** (variables)
9. The **train data (final)** subset with **SMOTE-upsampled** minority class(**LUNG_CANCER=No**) is comprised of:
    * **302 rows** (observations)
        * **151 LUNG_CANCER=Yes**: 50.00%
        * **151 LUNG_CANCER=No**: 50.00%
    * **11 columns** (variables)
10. The **train data (final)** subset with **CNN-downsampled** minority class(**LUNG_CANCER=Yes**) is comprised of:
    * **173 rows** (observations)
        * **39 LUNG_CANCER=Yes**: 63.93%
        * **22 LUNG_CANCER=No**: 36.07%
    * **11 columns** (variables)


```python
##################################
# Creating a dataset copy
# of the filtered data
##################################
lung_cancer_final = lung_cancer_filtered.copy()
```


```python
##################################
# Performing a general exploration
# of the final dataset
##################################
print('Final Dataset Dimensions: ')
display(lung_cancer_final.shape)
```

    Final Dataset Dimensions: 
    


    (309, 11)



```python
print('Target Variable Breakdown: ')
lung_cancer_breakdown = lung_cancer_final.groupby('LUNG_CANCER', observed=True).size().reset_index(name='Count')
lung_cancer_breakdown['Percentage'] = (lung_cancer_breakdown['Count'] / len(lung_cancer_final)) * 100
display(lung_cancer_breakdown)
```

    Target Variable Breakdown: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LUNG_CANCER</th>
      <th>Count</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>39</td>
      <td>12.621359</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>270</td>
      <td>87.378641</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the train and test data
# from the final dataset
# by applying stratification and
# using a 70-30 ratio
##################################
lung_cancer_train_initial, lung_cancer_test = train_test_split(lung_cancer_final, 
                                                               test_size=0.25, 
                                                               stratify=lung_cancer_final['LUNG_CANCER'], 
                                                               random_state=88888888)
```


```python
##################################
# Performing a general exploration
# of the initial training dataset
##################################
X_train_initial = lung_cancer_train_initial.drop('LUNG_CANCER', axis = 1)
y_train_initial = lung_cancer_train_initial['LUNG_CANCER']
print('Initial Training Dataset Dimensions: ')
display(X_train_initial.shape)
display(y_train_initial.shape)
print('Initial Training Target Variable Breakdown: ')
display(y_train_initial.value_counts())
print('Initial Training Target Variable Proportion: ')
display(y_train_initial.value_counts(normalize = True))
```

    Initial Training Dataset Dimensions: 
    


    (231, 10)



    (231,)


    Initial Training Target Variable Breakdown: 
    


    LUNG_CANCER
    1    202
    0     29
    Name: count, dtype: int64


    Initial Training Target Variable Proportion: 
    


    LUNG_CANCER
    1    0.874459
    0    0.125541
    Name: proportion, dtype: float64



```python
##################################
# Performing a general exploration
# of the test dataset
##################################
X_test = lung_cancer_test.drop('LUNG_CANCER', axis = 1)
y_test = lung_cancer_test['LUNG_CANCER']
print('Test Dataset Dimensions: ')
display(X_test.shape)
display(y_test.shape)
print('Test Target Variable Breakdown: ')
display(y_test.value_counts())
print('Test Target Variable Proportion: ')
display(y_test.value_counts(normalize = True))
```

    Test Dataset Dimensions: 
    


    (78, 10)



    (78,)


    Test Target Variable Breakdown: 
    


    LUNG_CANCER
    1    68
    0    10
    Name: count, dtype: int64


    Test Target Variable Proportion: 
    


    LUNG_CANCER
    1    0.871795
    0    0.128205
    Name: proportion, dtype: float64



```python
##################################
# Formulating the train and validation data
# from the train dataset
# by applying stratification and
# using a 70-30 ratio
##################################
lung_cancer_train, lung_cancer_validation = train_test_split(lung_cancer_train_initial, 
                                                             test_size=0.25, 
                                                             stratify=lung_cancer_train_initial['LUNG_CANCER'], 
                                                             random_state=88888888)
```


```python
##################################
# Performing a general exploration
# of the final training dataset
##################################
X_train = lung_cancer_train.drop('LUNG_CANCER', axis = 1)
y_train = lung_cancer_train['LUNG_CANCER']
print('Final Training Dataset Dimensions: ')
display(X_train.shape)
display(y_train.shape)
print('Final Training Target Variable Breakdown: ')
display(y_train.value_counts())
print('Final Training Target Variable Proportion: ')
display(y_train.value_counts(normalize = True))
```

    Final Training Dataset Dimensions: 
    


    (173, 10)



    (173,)


    Final Training Target Variable Breakdown: 
    


    LUNG_CANCER
    1    151
    0     22
    Name: count, dtype: int64


    Final Training Target Variable Proportion: 
    


    LUNG_CANCER
    1    0.872832
    0    0.127168
    Name: proportion, dtype: float64



```python
##################################
# Performing a general exploration
# of the validation dataset
##################################
X_validation = lung_cancer_validation.drop('LUNG_CANCER', axis = 1)
y_validation = lung_cancer_validation['LUNG_CANCER']
print('Validation Dataset Dimensions: ')
display(X_validation.shape)
display(y_validation.shape)
print('Validation Target Variable Breakdown: ')
display(y_validation.value_counts())
print('Validation Target Variable Proportion: ')
display(y_validation.value_counts(normalize = True))
```

    Validation Dataset Dimensions: 
    


    (58, 10)



    (58,)


    Validation Target Variable Breakdown: 
    


    LUNG_CANCER
    1    51
    0     7
    Name: count, dtype: int64


    Validation Target Variable Proportion: 
    


    LUNG_CANCER
    1    0.87931
    0    0.12069
    Name: proportion, dtype: float64



```python
##################################
# Initiating an oversampling instance
# on the training data using
# Synthetic Minority Oversampling Technique
##################################
smote = SMOTE(random_state = 88888888)
X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
print('SMOTE-Upsampled Training Dataset Dimensions: ')
display(X_train_smote.shape)
display(y_train_smote.shape)
print('SMOTE-Upsampled Training Target Variable Breakdown: ')
display(y_train_smote.value_counts())
print('SMOTE-Upsampled Training Target Variable Proportion: ')
display(y_train_smote.value_counts(normalize = True))
```

    SMOTE-Upsampled Training Dataset Dimensions: 
    


    (302, 10)



    (302,)


    SMOTE-Upsampled Training Target Variable Breakdown: 
    


    LUNG_CANCER
    0    151
    1    151
    Name: count, dtype: int64


    SMOTE-Upsampled Training Target Variable Proportion: 
    


    LUNG_CANCER
    0    0.5
    1    0.5
    Name: proportion, dtype: float64



```python
##################################
# Initiating an undersampling instance
# on the training data using
# Condense Nearest Neighbors
##################################
cnn = CondensedNearestNeighbour(random_state = 88888888, n_neighbors=3)
X_train_cnn, y_train_cnn = cnn.fit_resample(X_train,y_train)
print('Downsampled Training Dataset Dimensions: ')
display(X_train_cnn.shape)
display(y_train_cnn.shape)
print('Downsampled Training Target Variable Breakdown: ')
display(y_train_cnn.value_counts())
print('Downsampled Training Target Variable Proportion: ')
display(y_train_cnn.value_counts(normalize = True))
```

    Downsampled Training Dataset Dimensions: 
    


    (61, 10)



    (61,)


    Downsampled Training Target Variable Breakdown: 
    


    LUNG_CANCER
    1    39
    0    22
    Name: count, dtype: int64


    Downsampled Training Target Variable Proportion: 
    


    LUNG_CANCER
    1    0.639344
    0    0.360656
    Name: proportion, dtype: float64



```python
##################################
# Saving the training data
# to the DATASETS_FINAL_TRAIN_PATH
# and DATASETS_FINAL_TRAIN_FEATURES_PATH
# and DATASETS_FINAL_TRAIN_TARGET_PATH
##################################
lung_cancer_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_PATH, "lung_cancer_train.csv"), index=False)
X_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_FEATURES_PATH, "X_train.csv"), index=False)
y_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_TARGET_PATH, "y_train.csv"), index=False)
X_train_smote.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_FEATURES_PATH, "X_train_smote.csv"), index=False)
y_train_smote.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_TARGET_PATH, "y_train_smote.csv"), index=False)
X_train_cnn.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_FEATURES_PATH, "X_train_cnn.csv"), index=False)
y_train_cnn.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_TARGET_PATH, "y_train_cnn.csv"), index=False)
```


```python
##################################
# Saving the validation data
# to the DATASETS_FINAL_VALIDATION_PATH
# and DATASETS_FINAL_VALIDATION_FEATURE_PATH
# and DATASETS_FINAL_VALIDATION_TARGET_PATH
##################################
lung_cancer_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_PATH, "lung_cancer_validation.csv"), index=False)
X_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_FEATURES_PATH, "X_validation.csv"), index=False)
y_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_TARGET_PATH, "y_validation.csv"), index=False)
```


```python
##################################
# Saving the test data
# to the DATASETS_FINAL_TEST_PATH
# and DATASETS_FINAL_TEST_FEATURES_PATH
# and DATASETS_FINAL_TEST_TARGET_PATH
##################################
lung_cancer_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_PATH, "lung_cancer_test.csv"), index=False)
X_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_FEATURES_PATH, "X_test.csv"), index=False)
y_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_TARGET_PATH, "y_test.csv"), index=False)
```

### 1.6.3 Modelling Pipeline Development <a class="anchor" id="1.6.3"></a>

#### 1.6.3.1 Individual Classifier <a class="anchor" id="1.6.3.1"></a>

[Logistic Regression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=360300) models the relationship between the probability of an event (among two outcome levels) by having the log-odds of the event be a linear combination of a set of predictors weighted by their respective parameter estimates. The parameters are estimated via maximum likelihood estimation by testing different values through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimates. Given the optimal parameters, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability.

[Class Weights](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) are used to assign different levels of importance to different classes when the distribution of instances across different classes in a classification problem is not equal. By assigning higher weights to the minority class, the model is encouraged to give more attention to correctly predicting instances from the minority class. Class weights are incorporated into the loss function during training. The loss for each instance is multiplied by its corresponding class weight. This means that misclassifying an instance from the minority class will have a greater impact on the overall loss than misclassifying an instance from the majority class. The use of class weights helps balance the influence of each class during training, mitigating the impact of class imbalance. It provides a way to focus the learning process on the classes that are underrepresented in the training data.

[Hyperparameter Tuning](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) is an iterative process that involves experimenting with different hyperparameter combinations, evaluating the model's performance, and refining the hyperparameter values to achieve the best possible performance on new, unseen data - aimed at building effective and well-generalizing machine learning models. A model's performance depends not only on the learned parameters (weights) during training but also on hyperparameters, which are external configuration settings that cannot be learned from the data.

1. A modelling pipeline using an individual classifier was implemented.
    * [Logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API with 5 hyperparameters:
        * <span style="color: #FF0000">penalty</span> = penalty norm made to vary between L1, L2 and none
        * <span style="color: #FF0000">class_weight</span> = weights associated with classes held constant at a value  equal to balanced or none, as applicable
        * <span style="color: #FF0000">solver</span> = algorithm used in the optimization problem held constant at a value equal to saga
        * <span style="color: #FF0000">max_iter</span> = maximum number of iterations taken for the solvers to converge held constant at a value of 500
        * <span style="color: #FF0000">random_state</span> = random instance to shuffle the data for the solver algorithm held constant at a value of 88888888
2. Hyperparameter tuning was conducted using the 5-fold cross-validation method with optimal model performance determined using the F1 score. 


```python
##################################
# Defining the modelling pipeline
# using the logistic regression structure
##################################
individual_pipeline = Pipeline([('individual_model', LogisticRegression(solver='saga', 
                                                             random_state=88888888, 
                                                             max_iter=5000))])
```


```python
##################################
# Defining the hyperparameters for grid search
# including the regularization penalties
# and class weights for unbalanced class
##################################
individual_unbalanced_class_hyperparameter_grid = {'individual_model__penalty': ['l1', 'l2', None],
                                                   'individual_model__class_weight': ['balanced']}
```


```python
##################################
# Setting up the GridSearchCV with 5-fold cross-validation
# and using F1 score as the model evaluation metric
##################################
individual_unbalanced_class_grid_search = GridSearchCV(estimator=individual_pipeline,
                                                       param_grid=individual_unbalanced_class_hyperparameter_grid,
                                                       scoring='f1',
                                                       cv=5, 
                                                       n_jobs=-1,
                                                       verbose=1)
```


```python
##################################
# Defining the hyperparameters for grid search
# including the regularization penalties
# and class weights for unbalanced class
##################################
individual_balanced_class_hyperparameter_grid = {'individual_model__penalty': ['l1', 'l2', None],
                                                 'individual_model__class_weight': [None]}
```


```python
##################################
# Setting up the GridSearchCV with 5-fold cross-validation
# and using F1 score as the model evaluation metric
##################################
individual_balanced_class_grid_search = GridSearchCV(estimator=individual_pipeline,
                                                     param_grid=individual_balanced_class_hyperparameter_grid,
                                                     scoring='f1',
                                                     cv=5, 
                                                     n_jobs=-1,
                                                     verbose=1)
```

#### 1.6.3.2 Stacked Classifier <a class="anchor" id="1.6.3.2"></a>

[Logistic Regression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=360300) models the relationship between the probability of an event (among two outcome levels) by having the log-odds of the event be a linear combination of a set of predictors weighted by their respective parameter estimates. The parameters are estimated via maximum likelihood estimation by testing different values through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimates. Given the optimal parameters, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability.

[Decision Trees](https://www.semanticscholar.org/paper/Classification-and-Regression-Trees-Breiman-Friedman/8017699564136f93af21575810d557dba1ee6fc6) create a model that predicts the class label of a sample based on input features. A decision tree consists of nodes that represent decisions or choices, edges which connect nodes and represent the possible outcomes of a decision and leaf (or terminal) nodes which represent the final decision or the predicted class label. The decision-making process involves feature selection (at each internal node, the algorithm decides which feature to split on based on a certain criterion including gini impurity or entropy), splitting criteria (the splitting criteria aim to find the feature and its corresponding threshold that best separates the data into different classes. The goal is to increase homogeneity within each resulting subset), recursive splitting (the process of feature selection and splitting continues recursively, creating a tree structure. The dataset is partitioned at each internal node based on the chosen feature, and the process repeats for each subset) and stopping criteria (the recursion stops when a certain condition is met, known as a stopping criterion. Common stopping criteria include a maximum depth for the tree, a minimum number of samples required to split a node, or a minimum number of samples in a leaf node.)

[Random Forest](https://link.springer.com/article/10.1023/A:1010933404324) is an ensemble learning method made up of a large set of small decision trees called estimators, with each producing its own prediction. The random forest model aggregates the predictions of the estimators to produce a more accurate prediction. The algorithm involves bootstrap aggregating (where smaller subsets of the training data are repeatedly subsampled with replacement), random subspacing (where a subset of features are sampled and used to train each individual estimator), estimator training (where unpruned decision trees are formulated for each estimator) and inference by aggregating the predictions of all estimators.

[Support Vector Machine](https://dl.acm.org/doi/10.1145/130385.130401) plots each observation in an N-dimensional space corresponding to the number of features in the data set and finds a hyperplane that maximally separates the different classes by a maximally large margin (which is defined as the distance between the hyperplane and the closest data points from each class). The algorithm applies kernel transformation by mapping non-linearly separable data using the similarities between the points in a high-dimensional feature space for improved discrimination.

[Hyperparameter Tuning](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) is an iterative process that involves experimenting with different hyperparameter combinations, evaluating the model's performance, and refining the hyperparameter values to achieve the best possible performance on new, unseen data - aimed at building effective and well-generalizing machine learning models. A model's performance depends not only on the learned parameters (weights) during training but also on hyperparameters, which are external configuration settings that cannot be learned from the data. 

[Model Stacking](https://www.manning.com/books/ensemble-methods-for-machine-learning) - also known as stacked generalization, is an ensemble approach which involves creating a variety of base learners and using them to create intermediate predictions, one for each learned model. A meta-model is incorporated that gains knowledge of the same target from intermediate predictions. Unlike bagging, in stacking, the models are typically different (e.g. not all decision trees) and fit on the same dataset (e.g. instead of samples of the training dataset). Unlike boosting, in stacking, a single model is used to learn how to best combine the predictions from the contributing models (e.g. instead of a sequence of models that correct the predictions of prior models). Stacking is appropriate when the predictions made by the base learners or the errors in predictions made by the models have minimal correlation. Achieving an improvement in performance is dependent upon the choice of base learners and whether they are sufficiently skillful in their predictions.

1. A modelling pipeline using a stacking classifier was implemented.
    * **Meta-learner**: [Logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API with 5 hyperparameters:
        * <span style="color: #FF0000">penalty</span> = penalty norm made to vary between L1, L2 and none
        * <span style="color: #FF0000">class_weight</span> = weights associated with classes held constant at a value  equal to balanced or none, as applicable
        * <span style="color: #FF0000">solver</span> = algorithm used in the optimization problem held constant at a value equal to saga
        * <span style="color: #FF0000">max_iter</span> = maximum number of iterations taken for the solvers to converge held constant at 500
        * <span style="color: #FF0000">random_state</span> = random instance to shuffle the data for the solver algorithm held constant at 88888888
    * **Base learner**: [Decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API with 5 hyperparameters:
        * <span style="color: #FF0000">max_depth</span> = maximum depth of the tree made to vary between 3 and 5
        * <span style="color: #FF0000">class_weight</span> = weights associated with classes held constant at a value  equal to balanced or none, as applicable
        * <span style="color: #FF0000">criterion</span> = function to measure the quality of a split held constant at a value equal to entropy
        * <span style="color: #FF0000">min_samples_leaf</span> = minimum number of samples required to split an internal node held constant at 3
        * <span style="color: #FF0000">random_state</span> = random instance for feature permutation process of the algorithm held constant at 88888888
    * **Base learner**: [Random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API with 6 hyperparameters:
        * <span style="color: #FF0000">max_depth</span> = maximum depth of the tree made to vary between 3 and 5
        * <span style="color: #FF0000">class_weight</span> = weights associated with classes held constant at a value equal to balanced or none, as applicable
        * <span style="color: #FF0000">criterion</span> = function to measure the quality of a split held constant at a value equal to entropy
        * <span style="color: #FF0000">max_features</span> = number of features to consider when looking for the best split held constant at a value equal to sqrt
        * <span style="color: #FF0000">min_samples_leaf</span> = minimum number of samples required to split an internal node held constant at 3
        * <span style="color: #FF0000">random_state</span> = random instance for controlling the bootstrapping of the samples and feature sampling of the algorithm held constant at 88888888
    * **Base learner**: [Support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API with 5 hyperparameters:
        * <span style="color: #FF0000">C</span> = inverse of regularization strength made to vary between 1.0 and 0.5
        * <span style="color: #FF0000">class_weight</span> = weights associated with classes held constant at a value  equal to balanced or none, as applicable
        * <span style="color: #FF0000">kernel</span> = kernel type to be used in the algorithm made held constant at a value equal to linear
        * <span style="color: #FF0000">probability</span> = setting to enable probability estimates held constant at a value equal to true
        * <span style="color: #FF0000">random_state</span> = random instance for controling data shuffle for probability estimation of the algorithm held constant at 88888888        
2. Hyperparameter tuning was conducted using the 5-fold cross-validation method with optimal model performance determined using the F1 score. 


```python
##################################
# Defining the base learners
# for the stacked classifier
# composed of decision tree,
# random forest, and support vector machine
##################################
stacked_unbalanced_class_base_learners = [('dt', DecisionTreeClassifier(class_weight='balanced',
                                                                         criterion='entropy',
                                                                         min_samples_leaf=3,
                                                                         random_state=88888888)),
                                           ('rf', RandomForestClassifier(class_weight='balanced',
                                                                         criterion='entropy',
                                                                         max_features='sqrt',
                                                                         min_samples_leaf=3,
                                                                         random_state=88888888)),
                                           ('svm', SVC(class_weight='balanced',
                                                       probability=True,
                                                       kernel='linear',
                                                       random_state=88888888))]
```


```python
##################################
# Defining the meta-learner
# using the logistic regression structure
##################################
stacked_unbalanced_class_meta_learner = LogisticRegression(solver='saga', 
                                                           random_state=88888888,
                                                           max_iter=5000)
```


```python
##################################
# Defining the stacking model
# using the logistic regression structure
##################################
stacked_unbalanced_class_model = StackingClassifier(estimators=stacked_unbalanced_class_base_learners,
                                                    final_estimator=stacked_unbalanced_class_meta_learner)
```


```python
##################################
# Defining the modelling pipeline
# for the stacked classifier
# composed of decision tree,
# random forest, and support vector machine
# using the logistic regression structure
##################################
stacked_unbalanced_class_pipeline = Pipeline([('stacked_model', stacked_unbalanced_class_model)])
```


```python
##################################
# Defining the hyperparameters for grid search
# including the regularization penalties
# and class weights for unbalanced class
##################################
stacked_unbalanced_class_hyperparameter_grid = {'stacked_model__dt__max_depth': [3, 5],
                                                'stacked_model__rf__max_depth': [3, 5],
                                                'stacked_model__svm__C': [0.50, 1.00],
                                                'stacked_model__final_estimator__penalty': ['l1', 'l2', None],
                                                'stacked_model__final_estimator__class_weight': ['balanced']}
```


```python
##################################
# Setting up the GridSearchCV with 5-fold cross-validation
# and using F1 score as the model evaluation metric
##################################
stacked_unbalanced_class_grid_search = GridSearchCV(estimator=stacked_unbalanced_class_pipeline,
                                                    param_grid=stacked_unbalanced_class_hyperparameter_grid,
                                                    scoring='f1',
                                                    cv=5,
                                                    n_jobs=-1,
                                                    verbose=1)
```


```python
##################################
# Defining the base learners
# for the stacked classifier
# composed of decision tree,
# random forest, and support vector machine
##################################
stacked_balanced_class_base_learners = [('dt', DecisionTreeClassifier(class_weight=None,
                                                                         criterion='entropy',
                                                                         min_samples_leaf=3,
                                                                         random_state=88888888)),
                                           ('rf', RandomForestClassifier(class_weight=None,
                                                                         criterion='entropy',
                                                                         max_features='sqrt',
                                                                         min_samples_leaf=3,
                                                                         random_state=88888888)),
                                           ('svm', SVC(class_weight=None,
                                                       probability=True,
                                                       kernel='linear',
                                                       random_state=88888888))]
```


```python
##################################
# Defining the meta-learner
# using the logistic regression structure
##################################
stacked_balanced_class_meta_learner = LogisticRegression(solver='saga', 
                                                           random_state=88888888,
                                                           max_iter=5000)
```


```python
##################################
# Defining the stacking model
# using the logistic regression structure
##################################
stacked_balanced_class_model = StackingClassifier(estimators=stacked_balanced_class_base_learners,
                                                    final_estimator=stacked_balanced_class_meta_learner)
```


```python
##################################
# Defining the modelling pipeline
# for the stacked classifier
# composed of decision tree,
# random forest, and support vector machine
# using the logistic regression structure
##################################
stacked_balanced_class_pipeline = Pipeline([('stacked_model', stacked_balanced_class_model)])
```


```python
##################################
# Defining the hyperparameters for grid search
# including the regularization penalties
# and class weights for balanced class
##################################
stacked_balanced_class_hyperparameter_grid = {'stacked_model__dt__max_depth': [3, 5],
                                                'stacked_model__rf__max_depth': [3, 5],
                                                'stacked_model__svm__C': [0.50, 1.00],
                                                'stacked_model__final_estimator__penalty': ['l1', 'l2', None],
                                                'stacked_model__final_estimator__class_weight': [None]}
```


```python
##################################
# Setting up the GridSearchCV with 5-fold cross-validation
# and using F1 score as the model evaluation metric
##################################
stacked_balanced_class_grid_search = GridSearchCV(estimator=stacked_balanced_class_pipeline,
                                                    param_grid=stacked_balanced_class_hyperparameter_grid,
                                                    scoring='f1',
                                                    cv=5,
                                                    n_jobs=-1,
                                                    verbose=1)
```

### 1.6.4 Model Fitting using Original Training Data | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.4"></a>

#### 1.6.4.1 Individual Classifier <a class="anchor" id="1.6.4.1"></a>

[Logistic Regression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=360300) models the relationship between the probability of an event (among two outcome levels) by having the log-odds of the event be a linear combination of a set of predictors weighted by their respective parameter estimates. The parameters are estimated via maximum likelihood estimation by testing different values through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimates. Given the optimal parameters, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability.

[Class Weights](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) are used to assign different levels of importance to different classes when the distribution of instances across different classes in a classification problem is not equal. By assigning higher weights to the minority class, the model is encouraged to give more attention to correctly predicting instances from the minority class. Class weights are incorporated into the loss function during training. The loss for each instance is multiplied by its corresponding class weight. This means that misclassifying an instance from the minority class will have a greater impact on the overall loss than misclassifying an instance from the majority class. The use of class weights helps balance the influence of each class during training, mitigating the impact of class imbalance. It provides a way to focus the learning process on the classes that are underrepresented in the training data.

[Hyperparameter Tuning](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) is an iterative process that involves experimenting with different hyperparameter combinations, evaluating the model's performance, and refining the hyperparameter values to achieve the best possible performance on new, unseen data - aimed at building effective and well-generalizing machine learning models. A model's performance depends not only on the learned parameters (weights) during training but also on hyperparameters, which are external configuration settings that cannot be learned from the data. 

1. The optimal [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (**individual classifier**) determined from the 5-fold cross-validation of **train data (final)** contained the following hyperparameters:
    * <span style="color: #FF0000">penalty</span> = L2
    * <span style="color: #FF0000">class_weight</span> = balanced
    * <span style="color: #FF0000">solver</span> = saga
    * <span style="color: #FF0000">max_iter</span> = 500
    * <span style="color: #FF0000">random_state</span> = 88888888
2. The **F1 scores** estimated for the different data subsets were as follows:
    * **train data (final)** = 0.9306
    * **train data (cross-validated)** = 0.9116
    * **validation data** = 0.9495
3. Moderate overfitting noted based on the considerable difference in the apparent and cross-validated **F1 scores**.


```python
##################################
# Fitting the model on the 
# original training data
##################################
individual_unbalanced_class_grid_search.fit(X_train, y_train)
```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits
    




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;individual_model&#x27;,
                 LogisticRegression(class_weight=&#x27;balanced&#x27;, max_iter=5000,
                                    random_state=88888888, solver=&#x27;saga&#x27;))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(class_weight=&#x27;balanced&#x27;, max_iter=5000,
                   random_state=88888888, solver=&#x27;saga&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Identifying the best model
##################################
individual_unbalanced_class_best_model_original = individual_unbalanced_class_grid_search.best_estimator_
```


```python
##################################
# Evaluating the F1 scores
# on the training, cross-validation, and validation data
##################################
individual_unbalanced_class_best_model_original_f1_cv = individual_unbalanced_class_grid_search.best_score_
individual_unbalanced_class_best_model_original_f1_train = f1_score(y_train, individual_unbalanced_class_best_model_original.predict(X_train))
individual_unbalanced_class_best_model_original_f1_validation = f1_score(y_validation, individual_unbalanced_class_best_model_original.predict(X_validation))
```


```python
##################################
# Identifying the optimal model
##################################
print('Best Individual Model using the Original Train Data: ')
print(f"Best Individual Model Parameters: {individual_unbalanced_class_grid_search.best_params_}")
```

    Best Individual Model using the Original Train Data: 
    Best Individual Model Parameters: {'individual_model__class_weight': 'balanced', 'individual_model__penalty': 'l2'}
    


```python
##################################
# Summarizing the F1 score results
# on the training and cross-validated data
# to assess overfitting optimism
##################################
print(f"F1 Score on Cross-Validated Data: {individual_unbalanced_class_best_model_original_f1_cv:.4f}")
print(f"F1 Score on Training Data: {individual_unbalanced_class_best_model_original_f1_train:.4f}")
print("\nClassification Report on Training Data:\n", classification_report(y_train, individual_unbalanced_class_best_model_original.predict(X_train)))
```

    F1 Score on Cross-Validated Data: 0.9116
    F1 Score on Training Data: 0.9306
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
               0       0.53      0.86      0.66        22
               1       0.98      0.89      0.93       151
    
        accuracy                           0.88       173
       macro avg       0.75      0.88      0.79       173
    weighted avg       0.92      0.88      0.90       173
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the training data
##################################
cm_raw = confusion_matrix(y_train, individual_unbalanced_class_best_model_original.predict(X_train))
cm_normalized = confusion_matrix(y_train, individual_unbalanced_class_best_model_original.predict(X_train), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Individual Model on Training Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Individual Model on Training Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_147_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
##################################
print(f"F1 Score on Validation Data: {individual_unbalanced_class_best_model_original_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_validation, individual_unbalanced_class_best_model_original.predict(X_validation)))
```

    F1 Score on Validation Data: 0.9495
    
    Classification Report on Validation Data:
                   precision    recall  f1-score   support
    
               0       0.60      0.86      0.71         7
               1       0.98      0.92      0.95        51
    
        accuracy                           0.91        58
       macro avg       0.79      0.89      0.83        58
    weighted avg       0.93      0.91      0.92        58
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the validation data
##################################
cm_raw = confusion_matrix(y_validation, individual_unbalanced_class_best_model_original.predict(X_validation))
cm_normalized = confusion_matrix(y_validation, individual_unbalanced_class_best_model_original.predict(X_validation), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Individual Model on Validation Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Individual Model on Validation Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_149_0.png)
    



```python
##################################
# Obtaining the logit values (log-odds)
# from the decision function for training data
##################################
individual_unbalanced_class_best_model_original_logit_values = individual_unbalanced_class_best_model_original.decision_function(X_train)
```


```python
##################################
# Obtaining the estimated probabilities 
# for the positive class (LUNG_CANCER=YES) for training data
##################################
individual_unbalanced_class_best_model_original_probabilities = individual_unbalanced_class_best_model_original.predict_proba(X_train)[:, 1]
```


```python
##################################
# Sorting the values to generate
# a smoother curve
##################################
individual_unbalanced_class_best_model_original_sorted_indices = np.argsort(individual_unbalanced_class_best_model_original_logit_values)
individual_unbalanced_class_best_model_original_logit_values_sorted = individual_unbalanced_class_best_model_original_logit_values[individual_unbalanced_class_best_model_original_sorted_indices]
individual_unbalanced_class_best_model_original_probabilities_sorted = individual_unbalanced_class_best_model_original_probabilities[individual_unbalanced_class_best_model_original_sorted_indices]
```


```python
##################################
# Plotting the estimated logistic curve
# using the logit values
# and estimated probabilities
# obtained from the training data
##################################
plt.figure(figsize=(17, 8))
plt.plot(individual_unbalanced_class_best_model_original_logit_values_sorted, 
         individual_unbalanced_class_best_model_original_probabilities_sorted, label='Classification Model Logistic Curve', color='black')
plt.ylim(-0.05, 1.05)
plt.xlim(-8.00, 8.00)
target_0_indices = y_train == 0
target_1_indices = y_train == 1
plt.scatter(individual_unbalanced_class_best_model_original_logit_values[target_0_indices], 
            individual_unbalanced_class_best_model_original_probabilities[target_0_indices], 
            color='blue', alpha=0.40, s=100, marker= 'o', edgecolor='k', label='LUNG_CANCER=NO')
plt.scatter(individual_unbalanced_class_best_model_original_logit_values[target_1_indices], 
            individual_unbalanced_class_best_model_original_probabilities[target_1_indices], 
            color='red', alpha=0.40, s=100, marker='o', edgecolor='k', label='LUNG_CANCER=YES')
plt.axhline(0.5, color='green', linestyle='--', label='Classification Model Threshold')
plt.title('Logistic Curve (Original Training Data): Individual Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_153_0.png)
    



```python
##################################
# Saving the best individual model
# developed from the original training data
################################## 
joblib.dump(individual_unbalanced_class_best_model_original, 
            os.path.join("..", MODELS_PATH, "individual_unbalanced_class_best_model_original.pkl"))
```




    ['..\\models\\individual_unbalanced_class_best_model_original.pkl']



#### 1.6.4.2 Stacked Classifier <a class="anchor" id="1.6.4.2"></a>

[Logistic Regression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=360300) models the relationship between the probability of an event (among two outcome levels) by having the log-odds of the event be a linear combination of a set of predictors weighted by their respective parameter estimates. The parameters are estimated via maximum likelihood estimation by testing different values through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimates. Given the optimal parameters, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability.

[Decision Trees](https://www.semanticscholar.org/paper/Classification-and-Regression-Trees-Breiman-Friedman/8017699564136f93af21575810d557dba1ee6fc6) create a model that predicts the class label of a sample based on input features. A decision tree consists of nodes that represent decisions or choices, edges which connect nodes and represent the possible outcomes of a decision and leaf (or terminal) nodes which represent the final decision or the predicted class label. The decision-making process involves feature selection (at each internal node, the algorithm decides which feature to split on based on a certain criterion including gini impurity or entropy), splitting criteria (the splitting criteria aim to find the feature and its corresponding threshold that best separates the data into different classes. The goal is to increase homogeneity within each resulting subset), recursive splitting (the process of feature selection and splitting continues recursively, creating a tree structure. The dataset is partitioned at each internal node based on the chosen feature, and the process repeats for each subset) and stopping criteria (the recursion stops when a certain condition is met, known as a stopping criterion. Common stopping criteria include a maximum depth for the tree, a minimum number of samples required to split a node, or a minimum number of samples in a leaf node.)

[Random Forest](https://link.springer.com/article/10.1023/A:1010933404324) is an ensemble learning method made up of a large set of small decision trees called estimators, with each producing its own prediction. The random forest model aggregates the predictions of the estimators to produce a more accurate prediction. The algorithm involves bootstrap aggregating (where smaller subsets of the training data are repeatedly subsampled with replacement), random subspacing (where a subset of features are sampled and used to train each individual estimator), estimator training (where unpruned decision trees are formulated for each estimator) and inference by aggregating the predictions of all estimators.

[Support Vector Machine](https://dl.acm.org/doi/10.1145/130385.130401) plots each observation in an N-dimensional space corresponding to the number of features in the data set and finds a hyperplane that maximally separates the different classes by a maximally large margin (which is defined as the distance between the hyperplane and the closest data points from each class). The algorithm applies kernel transformation by mapping non-linearly separable data using the similarities between the points in a high-dimensional feature space for improved discrimination.

[Class Weights](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) are used to assign different levels of importance to different classes when the distribution of instances across different classes in a classification problem is not equal. By assigning higher weights to the minority class, the model is encouraged to give more attention to correctly predicting instances from the minority class. Class weights are incorporated into the loss function during training. The loss for each instance is multiplied by its corresponding class weight. This means that misclassifying an instance from the minority class will have a greater impact on the overall loss than misclassifying an instance from the majority class. The use of class weights helps balance the influence of each class during training, mitigating the impact of class imbalance. It provides a way to focus the learning process on the classes that are underrepresented in the training data.

[Hyperparameter Tuning](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) is an iterative process that involves experimenting with different hyperparameter combinations, evaluating the model's performance, and refining the hyperparameter values to achieve the best possible performance on new, unseen data - aimed at building effective and well-generalizing machine learning models. A model's performance depends not only on the learned parameters (weights) during training but also on hyperparameters, which are external configuration settings that cannot be learned from the data. 

[Model Stacking](https://www.manning.com/books/ensemble-methods-for-machine-learning) - also known as stacked generalization, is an ensemble approach which involves creating a variety of base learners and using them to create intermediate predictions, one for each learned model. A meta-model is incorporated that gains knowledge of the same target from intermediate predictions. Unlike bagging, in stacking, the models are typically different (e.g. not all decision trees) and fit on the same dataset (e.g. instead of samples of the training dataset). Unlike boosting, in stacking, a single model is used to learn how to best combine the predictions from the contributing models (e.g. instead of a sequence of models that correct the predictions of prior models). Stacking is appropriate when the predictions made by the base learners or the errors in predictions made by the models have minimal correlation. Achieving an improvement in performance is dependent upon the choice of base learners and whether they are sufficiently skillful in their predictions.

1. The optimal [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) (**base learner**) determined from the 5-fold cross-validation of **train data (final)** contained the following hyperparameters:
    * <span style="color: #FF0000">max_depth</span> = 3
    * <span style="color: #FF0000">class_weight</span> = balanced
    * <span style="color: #FF0000">criterion</span> = entropy
    * <span style="color: #FF0000">min_samples_leaf</span> = 3
    * <span style="color: #FF0000">random_state</span> = 88888888
2. The optimal [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#) (**base learner**) determined from the 5-fold cross-validation of **train data (final)** contained the following hyperparameters:
    * <span style="color: #FF0000">max_depth</span> = 5
    * <span style="color: #FF0000">class_weight</span> = balanced
    * <span style="color: #FF0000">criterion</span> = entropy
    * <span style="color: #FF0000">max_features</span> = sqrt
    * <span style="color: #FF0000">min_samples_leaf</span> = 3
    * <span style="color: #FF0000">random_state</span> = 88888888
3. The optimal [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) (**base learner**) determined from the 5-fold cross-validation of **train data (final)** contained the following hyperparameters:
    * <span style="color: #FF0000">C</span> = 0.50
    * <span style="color: #FF0000">class_weight</span> = balanced
    * <span style="color: #FF0000">kernel</span> = linear
    * <span style="color: #FF0000">probability</span> = true
    * <span style="color: #FF0000">random_state</span> = 88888888  
4. The optimal [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (**meta-learner**) determined from the 5-fold cross-validation of **train data (final)** contained the following hyperparameters:
    * <span style="color: #FF0000">penalty</span> = L1
    * <span style="color: #FF0000">class_weight</span> = balanced
    * <span style="color: #FF0000">solver</span> = saga
    * <span style="color: #FF0000">max_iter</span> = 500
    * <span style="color: #FF0000">random_state</span> = 88888888
5. The **F1 scores** estimated for the different data subsets were as follows:
    * **train data (final)** = 0.9404
    * **train data (cross-validated)** = 0.9125
    * **validation data** = 0.9149
6. Moderate overfitting noted based on the considerable difference in the apparent and cross-validated **F1 scores**.


```python
##################################
# Fitting the model on the 
# original training data
##################################
stacked_unbalanced_class_grid_search.fit(X_train, y_train)
```

    Fitting 5 folds for each of 24 candidates, totalling 120 fits
    




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;stacked_model&#x27;,
                                        StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                                                        DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;,
                                                                                               criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;rf&#x27;,
                                                                        RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                                                                               criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;svm&#x27;,
                                                                        SVC(class_weight=&#x27;b...
                                                           final_estimator=LogisticRegression(max_iter=5000,
                                                                                              random_state=88888888,
                                                                                              solver=&#x27;saga&#x27;)))]),
             n_jobs=-1,
             param_grid={&#x27;stacked_model__dt__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__final_estimator__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;stacked_model__final_estimator__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;,
                                                                     None],
                         &#x27;stacked_model__rf__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__svm__C&#x27;: [0.5, 1.0]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;stacked_model&#x27;,
                                        StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                                                        DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;,
                                                                                               criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;rf&#x27;,
                                                                        RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                                                                               criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;svm&#x27;,
                                                                        SVC(class_weight=&#x27;b...
                                                           final_estimator=LogisticRegression(max_iter=5000,
                                                                                              random_state=88888888,
                                                                                              solver=&#x27;saga&#x27;)))]),
             n_jobs=-1,
             param_grid={&#x27;stacked_model__dt__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__final_estimator__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;stacked_model__final_estimator__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;,
                                                                     None],
                         &#x27;stacked_model__rf__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__svm__C&#x27;: [0.5, 1.0]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;stacked_model&#x27;,
                 StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                                 DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;,
                                                                        criterion=&#x27;entropy&#x27;,
                                                                        max_depth=3,
                                                                        min_samples_leaf=3,
                                                                        random_state=88888888)),
                                                (&#x27;rf&#x27;,
                                                 RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                                                        criterion=&#x27;entropy&#x27;,
                                                                        max_depth=5,
                                                                        min_samples_leaf=3,
                                                                        random_state=88888888)),
                                                (&#x27;svm&#x27;,
                                                 SVC(C=0.5,
                                                     class_weight=&#x27;balanced&#x27;,
                                                     kernel=&#x27;linear&#x27;,
                                                     probability=True,
                                                     random_state=88888888))],
                                    final_estimator=LogisticRegression(class_weight=&#x27;balanced&#x27;,
                                                                       max_iter=5000,
                                                                       penalty=&#x27;l1&#x27;,
                                                                       random_state=88888888,
                                                                       solver=&#x27;saga&#x27;)))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;stacked_model: StackingClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.StackingClassifier.html">?<span>Documentation for stacked_model: StackingClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;,
                                                       criterion=&#x27;entropy&#x27;,
                                                       max_depth=3,
                                                       min_samples_leaf=3,
                                                       random_state=88888888)),
                               (&#x27;rf&#x27;,
                                RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                                       criterion=&#x27;entropy&#x27;,
                                                       max_depth=5,
                                                       min_samples_leaf=3,
                                                       random_state=88888888)),
                               (&#x27;svm&#x27;,
                                SVC(C=0.5, class_weight=&#x27;balanced&#x27;,
                                    kernel=&#x27;linear&#x27;, probability=True,
                                    random_state=88888888))],
                   final_estimator=LogisticRegression(class_weight=&#x27;balanced&#x27;,
                                                      max_iter=5000,
                                                      penalty=&#x27;l1&#x27;,
                                                      random_state=88888888,
                                                      solver=&#x27;saga&#x27;))</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>dt</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;DecisionTreeClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeClassifier.html">?<span>Documentation for DecisionTreeClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;, criterion=&#x27;entropy&#x27;,
                       max_depth=3, min_samples_leaf=3, random_state=88888888)</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>rf</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(class_weight=&#x27;balanced&#x27;, criterion=&#x27;entropy&#x27;,
                       max_depth=5, min_samples_leaf=3, random_state=88888888)</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>svm</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SVC<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html">?<span>Documentation for SVC</span></a></label><div class="sk-toggleable__content fitted"><pre>SVC(C=0.5, class_weight=&#x27;balanced&#x27;, kernel=&#x27;linear&#x27;, probability=True,
    random_state=88888888)</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>final_estimator</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(class_weight=&#x27;balanced&#x27;, max_iter=5000, penalty=&#x27;l1&#x27;,
                   random_state=88888888, solver=&#x27;saga&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Identifying the best model
##################################
stacked_unbalanced_class_best_model_original = stacked_unbalanced_class_grid_search.best_estimator_
```


```python
##################################
# Evaluating the F1 scores
# on the training, cross-validation, and validation data
##################################
stacked_unbalanced_class_best_model_original_f1_cv = stacked_unbalanced_class_grid_search.best_score_
stacked_unbalanced_class_best_model_original_f1_train = f1_score(y_train, stacked_unbalanced_class_best_model_original.predict(X_train))
stacked_unbalanced_class_best_model_original_f1_validation = f1_score(y_validation, stacked_unbalanced_class_best_model_original.predict(X_validation))
```


```python
##################################
# Identifying the optimal model
##################################
print('Best Stacked Model using the Original Train Data: ')
print(f"Best Stacked Model Parameters: {stacked_unbalanced_class_grid_search.best_params_}")
```

    Best Stacked Model using the Original Train Data: 
    Best Stacked Model Parameters: {'stacked_model__dt__max_depth': 3, 'stacked_model__final_estimator__class_weight': 'balanced', 'stacked_model__final_estimator__penalty': 'l1', 'stacked_model__rf__max_depth': 5, 'stacked_model__svm__C': 0.5}
    


```python
##################################
# Summarizing the F1 score results
# on the training and cross-validated data
# to assess overfitting optimism
##################################
print(f"F1 Score on Cross-Validated Data: {stacked_unbalanced_class_best_model_original_f1_cv:.4f}")
print(f"F1 Score on Training Data: {stacked_unbalanced_class_best_model_original_f1_train:.4f}")
print("\nClassification Report on Training Data:\n", classification_report(y_train, stacked_unbalanced_class_best_model_original.predict(X_train)))
```

    F1 Score on Cross-Validated Data: 0.9125
    F1 Score on Training Data: 0.9404
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
               0       0.56      1.00      0.72        22
               1       1.00      0.89      0.94       151
    
        accuracy                           0.90       173
       macro avg       0.78      0.94      0.83       173
    weighted avg       0.94      0.90      0.91       173
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the training data
##################################
cm_raw = confusion_matrix(y_train, stacked_unbalanced_class_best_model_original.predict(X_train))
cm_normalized = confusion_matrix(y_train, stacked_unbalanced_class_best_model_original.predict(X_train), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Stacked Model on Training Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Stacked Model on Training Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_161_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
##################################
print(f"F1 Score on Validation Data: {stacked_unbalanced_class_best_model_original_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_validation, stacked_unbalanced_class_best_model_original.predict(X_validation)))
```

    F1 Score on Validation Data: 0.9149
    
    Classification Report on Validation Data:
                   precision    recall  f1-score   support
    
               0       0.47      1.00      0.64         7
               1       1.00      0.84      0.91        51
    
        accuracy                           0.86        58
       macro avg       0.73      0.92      0.78        58
    weighted avg       0.94      0.86      0.88        58
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the validation data
##################################
cm_raw = confusion_matrix(y_validation, stacked_unbalanced_class_best_model_original.predict(X_validation))
cm_normalized = confusion_matrix(y_validation, stacked_unbalanced_class_best_model_original.predict(X_validation), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Stacked Model on Validation Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Stacked Model on Validation Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_163_0.png)
    



```python
##################################
# Obtaining the logit values (log-odds)
# from the decision function for training data
##################################
stacked_unbalanced_class_best_model_original_logit_values = stacked_unbalanced_class_best_model_original.decision_function(X_train)
```


```python
##################################
# Obtaining the estimated probabilities 
# for the positive class (LUNG_CANCER=YES) for training data
##################################
stacked_unbalanced_class_best_model_original_probabilities = stacked_unbalanced_class_best_model_original.predict_proba(X_train)[:, 1]
```


```python
##################################
# Sorting the values to generate
# a smoother curve
##################################
stacked_unbalanced_class_best_model_original_sorted_indices = np.argsort(stacked_unbalanced_class_best_model_original_logit_values)
stacked_unbalanced_class_best_model_original_logit_values_sorted = stacked_unbalanced_class_best_model_original_logit_values[stacked_unbalanced_class_best_model_original_sorted_indices]
stacked_unbalanced_class_best_model_original_probabilities_sorted = stacked_unbalanced_class_best_model_original_probabilities[stacked_unbalanced_class_best_model_original_sorted_indices]
```


```python
##################################
# Plotting the estimated logistic curve
# using the logit values
# and estimated probabilities
# obtained from the training data
##################################
plt.figure(figsize=(17, 8))
plt.plot(stacked_unbalanced_class_best_model_original_logit_values_sorted, 
         stacked_unbalanced_class_best_model_original_probabilities_sorted, label='Classification Model Logistic Curve', color='black')
plt.ylim(-0.05, 1.05)
plt.xlim(-8.00, 8.00)
target_0_indices = y_train == 0
target_1_indices = y_train == 1
plt.scatter(stacked_unbalanced_class_best_model_original_logit_values[target_0_indices], 
            stacked_unbalanced_class_best_model_original_probabilities[target_0_indices], 
            color='blue', alpha=0.40, s=100, marker= 'o', edgecolor='k', label='LUNG_CANCER=NO')
plt.scatter(stacked_unbalanced_class_best_model_original_logit_values[target_1_indices], 
            stacked_unbalanced_class_best_model_original_probabilities[target_1_indices], 
            color='red', alpha=0.40, s=100, marker='o', edgecolor='k', label='LUNG_CANCER=YES')
plt.axhline(0.5, color='green', linestyle='--', label='Classification Model Threshold')
plt.title('Logistic Curve (Original Training Data): Stacked Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_167_0.png)
    



```python
##################################
# Saving the best stacked model
# developed from the original training data
################################## 
joblib.dump(stacked_unbalanced_class_best_model_original, 
            os.path.join("..", MODELS_PATH, "stacked_unbalanced_class_best_model_original.pkl"))
```




    ['..\\models\\stacked_unbalanced_class_best_model_original.pkl']



### 1.6.5 Model Fitting using Upsampled Training Data | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.5"></a>

#### 1.6.5.1 Individual Classifier <a class="anchor" id="1.6.5.1"></a>

[Synthetic Minority Oversampling Technique](https://dl.acm.org/doi/10.5555/1622407.1622416) is specifically designed to increase the representation of the minority class by generating new minority instances between existing instances. The new instances created are not just the copy of existing minority cases, instead for each minority class instance, the algorithm generates synthetic examples by creating linear combinations of the feature vectors between that instance and its k nearest neighbors. The synthetic samples are placed along the line segments connecting the original instance to its neighbors.

[Logistic Regression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=360300) models the relationship between the probability of an event (among two outcome levels) by having the log-odds of the event be a linear combination of a set of predictors weighted by their respective parameter estimates. The parameters are estimated via maximum likelihood estimation by testing different values through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimates. Given the optimal parameters, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability.

[Hyperparameter Tuning](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) is an iterative process that involves experimenting with different hyperparameter combinations, evaluating the model's performance, and refining the hyperparameter values to achieve the best possible performance on new, unseen data - aimed at building effective and well-generalizing machine learning models. A model's performance depends not only on the learned parameters (weights) during training but also on hyperparameters, which are external configuration settings that cannot be learned from the data.

1. The optimal [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (**individual classifier**) determined from the 5-fold cross-validation of **train data (SMOTE-upsampled)** contained the following hyperparameters:
    * <span style="color: #FF0000">penalty</span> = L1
    * <span style="color: #FF0000">class_weight</span> = none
    * <span style="color: #FF0000">solver</span> = saga
    * <span style="color: #FF0000">max_iter</span> = 500
    * <span style="color: #FF0000">random_state</span> = 88888888
2. The **F1 scores** estimated for the different data subsets were as follows:
    * **train data (SMOTE-upsampled)** = 0.9122
    * **train data (cross-validated)** = 0.9109
    * **validation data** = 0.9278
3. Minimal overfitting noted based on the small difference in the apparent and cross-validated **F1 scores**.


```python
##################################
# Fitting the model on the 
# upsampled training data
##################################
individual_balanced_class_grid_search.fit(X_train_smote, y_train_smote)
```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits
    




<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [None],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [None],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;individual_model&#x27;,
                 LogisticRegression(max_iter=5000, penalty=&#x27;l1&#x27;,
                                    random_state=88888888, solver=&#x27;saga&#x27;))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(max_iter=5000, penalty=&#x27;l1&#x27;, random_state=88888888,
                   solver=&#x27;saga&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Identifying the best model
##################################
individual_balanced_class_best_model_upsampled = individual_balanced_class_grid_search.best_estimator_
```


```python
##################################
# Evaluating the F1 scores
# on the training, cross-validation, and validation data
##################################
individual_balanced_class_best_model_upsampled_f1_cv = individual_balanced_class_grid_search.best_score_
individual_balanced_class_best_model_upsampled_f1_train_smote = f1_score(y_train_smote, individual_balanced_class_best_model_upsampled.predict(X_train_smote))
individual_balanced_class_best_model_upsampled_f1_validation = f1_score(y_validation, individual_balanced_class_best_model_upsampled.predict(X_validation))
```


```python
##################################
# Identifying the optimal model
##################################
print('Best Individual Model using the SMOTE-Upsampled Train Data: ')
print(f"Best Individual Model Parameters: {individual_balanced_class_grid_search.best_params_}")
```

    Best Individual Model using the SMOTE-Upsampled Train Data: 
    Best Individual Model Parameters: {'individual_model__class_weight': None, 'individual_model__penalty': 'l1'}
    


```python
##################################
# Summarizing the F1 score results
# on the training and cross-validated data
# to assess overfitting optimism
##################################
print(f"F1 Score on Cross-Validated Data: {individual_balanced_class_best_model_upsampled_f1_cv:.4f}")
print(f"F1 Score on Training Data: {individual_balanced_class_best_model_upsampled_f1_train_smote:.4f}")
print("\nClassification Report on Training Data:\n", classification_report(y_train_smote, individual_balanced_class_best_model_upsampled.predict(X_train_smote)))
```

    F1 Score on Cross-Validated Data: 0.9109
    F1 Score on Training Data: 0.9122
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
               0       0.90      0.93      0.92       151
               1       0.93      0.89      0.91       151
    
        accuracy                           0.91       302
       macro avg       0.91      0.91      0.91       302
    weighted avg       0.91      0.91      0.91       302
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the training data
##################################
cm_raw = confusion_matrix(y_train_smote, individual_balanced_class_best_model_upsampled.predict(X_train_smote))
cm_normalized = confusion_matrix(y_train_smote, individual_balanced_class_best_model_upsampled.predict(X_train_smote), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Individual Model on Training Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Individual Model on Training Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_176_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
##################################
print(f"F1 Score on Validation Data: {individual_balanced_class_best_model_upsampled_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_validation, individual_balanced_class_best_model_upsampled.predict(X_validation)))
```

    F1 Score on Validation Data: 0.9278
    
    Classification Report on Validation Data:
                   precision    recall  f1-score   support
    
               0       0.50      0.86      0.63         7
               1       0.98      0.88      0.93        51
    
        accuracy                           0.88        58
       macro avg       0.74      0.87      0.78        58
    weighted avg       0.92      0.88      0.89        58
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the validation data
##################################
cm_raw = confusion_matrix(y_validation, individual_balanced_class_best_model_upsampled.predict(X_validation))
cm_normalized = confusion_matrix(y_validation, individual_balanced_class_best_model_upsampled.predict(X_validation), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Individual Model on Validation Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Individual Model on Validation Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_178_0.png)
    



```python
##################################
# Obtaining the logit values (log-odds)
# from the decision function for training data
##################################
individual_balanced_class_best_model_upsampled_logit_values = individual_balanced_class_best_model_upsampled.decision_function(X_train_smote)
```


```python
##################################
# Obtaining the estimated probabilities 
# for the positive class (LUNG_CANCER=YES) for training data
##################################
individual_balanced_class_best_model_upsampled_probabilities = individual_balanced_class_best_model_upsampled.predict_proba(X_train_smote)[:, 1]
```


```python
##################################
# Sorting the values to generate
# a smoother curve
##################################
individual_balanced_class_best_model_upsampled_sorted_indices = np.argsort(individual_balanced_class_best_model_upsampled_logit_values)
individual_balanced_class_best_model_upsampled_logit_values_sorted = individual_balanced_class_best_model_upsampled_logit_values[individual_balanced_class_best_model_upsampled_sorted_indices]
individual_balanced_class_best_model_upsampled_probabilities_sorted = individual_balanced_class_best_model_upsampled_probabilities[individual_balanced_class_best_model_upsampled_sorted_indices]
```


```python
##################################
# Plotting the estimated logistic curve
# using the logit values
# and estimated probabilities
# obtained from the training data
##################################
plt.figure(figsize=(17, 8))
plt.plot(individual_balanced_class_best_model_upsampled_logit_values_sorted, 
         individual_balanced_class_best_model_upsampled_probabilities_sorted, label='Classification Model Logistic Curve', color='black')
plt.ylim(-0.05, 1.05)
plt.xlim(-8.00, 8.00)
target_0_indices = y_train_smote == 0
target_1_indices = y_train_smote == 1
plt.scatter(individual_balanced_class_best_model_upsampled_logit_values[target_0_indices], 
            individual_balanced_class_best_model_upsampled_probabilities[target_0_indices], 
            color='blue', alpha=0.40, s=100, marker= 'o', edgecolor='k', label='LUNG_CANCER=NO')
plt.scatter(individual_balanced_class_best_model_upsampled_logit_values[target_1_indices], 
            individual_balanced_class_best_model_upsampled_probabilities[target_1_indices], 
            color='red', alpha=0.40, s=100, marker='o', edgecolor='k', label='LUNG_CANCER=YES')
plt.axhline(0.5, color='green', linestyle='--', label='Classification Model Threshold')
plt.title('Logistic Curve (Upsampled Training Data): Individual Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_182_0.png)
    



```python
##################################
# Saving the best individual model
# developed from the upsampled training data
################################## 
joblib.dump(individual_balanced_class_best_model_upsampled, 
            os.path.join("..", MODELS_PATH, "individual_balanced_class_best_model_upsampled.pkl"))
```




    ['..\\models\\individual_balanced_class_best_model_upsampled.pkl']



### 1.6.5.2 Stacked Classifier <a class="anchor" id="1.6.5.2"></a>

[Synthetic Minority Oversampling Technique](https://dl.acm.org/doi/10.5555/1622407.1622416) is specifically designed to increase the representation of the minority class by generating new minority instances between existing instances. The new instances created are not just the copy of existing minority cases, instead for each minority class instance, the algorithm generates synthetic examples by creating linear combinations of the feature vectors between that instance and its k nearest neighbors. The synthetic samples are placed along the line segments connecting the original instance to its neighbors.

[Logistic Regression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=360300) models the relationship between the probability of an event (among two outcome levels) by having the log-odds of the event be a linear combination of a set of predictors weighted by their respective parameter estimates. The parameters are estimated via maximum likelihood estimation by testing different values through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimates. Given the optimal parameters, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability.

[Decision Trees](https://www.semanticscholar.org/paper/Classification-and-Regression-Trees-Breiman-Friedman/8017699564136f93af21575810d557dba1ee6fc6) create a model that predicts the class label of a sample based on input features. A decision tree consists of nodes that represent decisions or choices, edges which connect nodes and represent the possible outcomes of a decision and leaf (or terminal) nodes which represent the final decision or the predicted class label. The decision-making process involves feature selection (at each internal node, the algorithm decides which feature to split on based on a certain criterion including gini impurity or entropy), splitting criteria (the splitting criteria aim to find the feature and its corresponding threshold that best separates the data into different classes. The goal is to increase homogeneity within each resulting subset), recursive splitting (the process of feature selection and splitting continues recursively, creating a tree structure. The dataset is partitioned at each internal node based on the chosen feature, and the process repeats for each subset) and stopping criteria (the recursion stops when a certain condition is met, known as a stopping criterion. Common stopping criteria include a maximum depth for the tree, a minimum number of samples required to split a node, or a minimum number of samples in a leaf node.)

[Random Forest](https://link.springer.com/article/10.1023/A:1010933404324) is an ensemble learning method made up of a large set of small decision trees called estimators, with each producing its own prediction. The random forest model aggregates the predictions of the estimators to produce a more accurate prediction. The algorithm involves bootstrap aggregating (where smaller subsets of the training data are repeatedly subsampled with replacement), random subspacing (where a subset of features are sampled and used to train each individual estimator), estimator training (where unpruned decision trees are formulated for each estimator) and inference by aggregating the predictions of all estimators.

[Support Vector Machine](https://dl.acm.org/doi/10.1145/130385.130401) plots each observation in an N-dimensional space corresponding to the number of features in the data set and finds a hyperplane that maximally separates the different classes by a maximally large margin (which is defined as the distance between the hyperplane and the closest data points from each class). The algorithm applies kernel transformation by mapping non-linearly separable data using the similarities between the points in a high-dimensional feature space for improved discrimination.

[Hyperparameter Tuning](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) is an iterative process that involves experimenting with different hyperparameter combinations, evaluating the model's performance, and refining the hyperparameter values to achieve the best possible performance on new, unseen data - aimed at building effective and well-generalizing machine learning models. A model's performance depends not only on the learned parameters (weights) during training but also on hyperparameters, which are external configuration settings that cannot be learned from the data. 

[Model Stacking](https://www.manning.com/books/ensemble-methods-for-machine-learning) - also known as stacked generalization, is an ensemble approach which involves creating a variety of base learners and using them to create intermediate predictions, one for each learned model. A meta-model is incorporated that gains knowledge of the same target from intermediate predictions. Unlike bagging, in stacking, the models are typically different (e.g. not all decision trees) and fit on the same dataset (e.g. instead of samples of the training dataset). Unlike boosting, in stacking, a single model is used to learn how to best combine the predictions from the contributing models (e.g. instead of a sequence of models that correct the predictions of prior models). Stacking is appropriate when the predictions made by the base learners or the errors in predictions made by the models have minimal correlation. Achieving an improvement in performance is dependent upon the choice of base learners and whether they are sufficiently skillful in their predictions.

1. The optimal [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) (**base learner**) determined from the 5-fold cross-validation of **train data (SMOTE-upsampled)** contained the following hyperparameters:
    * <span style="color: #FF0000">max_depth</span> = 3
    * <span style="color: #FF0000">class_weight</span> = none
    * <span style="color: #FF0000">criterion</span> = entropy
    * <span style="color: #FF0000">min_samples_leaf</span> = 3
    * <span style="color: #FF0000">random_state</span> = 88888888
2. The optimal [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#) (**base learner**) determined from the 5-fold cross-validation of **train data (SMOTE-upsampled)** contained the following hyperparameters:
    * <span style="color: #FF0000">max_depth</span> = 5
    * <span style="color: #FF0000">class_weight</span> = none
    * <span style="color: #FF0000">criterion</span> = entropy
    * <span style="color: #FF0000">max_features</span> = sqrt
    * <span style="color: #FF0000">min_samples_leaf</span> = 3
    * <span style="color: #FF0000">random_state</span> = 88888888
3. The optimal [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) (**base learner**) determined from the 5-fold cross-validation of **train data (SMOTE-upsampled)** contained the following hyperparameters:
    * <span style="color: #FF0000">C</span> = 1.00
    * <span style="color: #FF0000">class_weight</span> = none
    * <span style="color: #FF0000">kernel</span> = linear
    * <span style="color: #FF0000">probability</span> = true
    * <span style="color: #FF0000">random_state</span> = 88888888  
4. The optimal [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (**meta-learner**) determined from the 5-fold cross-validation of **train data (SMOTE-upsampled)** contained the following hyperparameters:
    * <span style="color: #FF0000">penalty</span> = none
    * <span style="color: #FF0000">class_weight</span> = none
    * <span style="color: #FF0000">solver</span> = saga
    * <span style="color: #FF0000">max_iter</span> = 500
    * <span style="color: #FF0000">random_state</span> = 88888888
5. The **F1 scores** estimated for the different data subsets were as follows:
    * **train data (SMOTE-upsampled)** = 0.9568
    * **train data (cross-validated)** = 0.9489
    * **validation data** = 0.9615
6. Minimal overfitting noted based on the small difference in the apparent and cross-validated **F1 scores**.


```python
##################################
# Fitting the model on the 
# upsampled training data
##################################
stacked_balanced_class_grid_search.fit(X_train_smote, y_train_smote)
```

    Fitting 5 folds for each of 24 candidates, totalling 120 fits
    




<style>#sk-container-id-4 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-4 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;stacked_model&#x27;,
                                        StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                                                        DecisionTreeClassifier(criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;rf&#x27;,
                                                                        RandomForestClassifier(criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;svm&#x27;,
                                                                        SVC(kernel=&#x27;linear&#x27;,
                                                                            probability=True,
                                                                            random_state=88888888))],
                                                           final_estimator=LogisticRegression(max_iter=5000,
                                                                                              random_state=88888888,
                                                                                              solver=&#x27;saga&#x27;)))]),
             n_jobs=-1,
             param_grid={&#x27;stacked_model__dt__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__final_estimator__class_weight&#x27;: [None],
                         &#x27;stacked_model__final_estimator__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;,
                                                                     None],
                         &#x27;stacked_model__rf__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__svm__C&#x27;: [0.5, 1.0]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;stacked_model&#x27;,
                                        StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                                                        DecisionTreeClassifier(criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;rf&#x27;,
                                                                        RandomForestClassifier(criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;svm&#x27;,
                                                                        SVC(kernel=&#x27;linear&#x27;,
                                                                            probability=True,
                                                                            random_state=88888888))],
                                                           final_estimator=LogisticRegression(max_iter=5000,
                                                                                              random_state=88888888,
                                                                                              solver=&#x27;saga&#x27;)))]),
             n_jobs=-1,
             param_grid={&#x27;stacked_model__dt__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__final_estimator__class_weight&#x27;: [None],
                         &#x27;stacked_model__final_estimator__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;,
                                                                     None],
                         &#x27;stacked_model__rf__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__svm__C&#x27;: [0.5, 1.0]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;stacked_model&#x27;,
                 StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                                 DecisionTreeClassifier(criterion=&#x27;entropy&#x27;,
                                                                        max_depth=3,
                                                                        min_samples_leaf=3,
                                                                        random_state=88888888)),
                                                (&#x27;rf&#x27;,
                                                 RandomForestClassifier(criterion=&#x27;entropy&#x27;,
                                                                        max_depth=5,
                                                                        min_samples_leaf=3,
                                                                        random_state=88888888)),
                                                (&#x27;svm&#x27;,
                                                 SVC(kernel=&#x27;linear&#x27;,
                                                     probability=True,
                                                     random_state=88888888))],
                                    final_estimator=LogisticRegression(max_iter=5000,
                                                                       penalty=None,
                                                                       random_state=88888888,
                                                                       solver=&#x27;saga&#x27;)))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;stacked_model: StackingClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.StackingClassifier.html">?<span>Documentation for stacked_model: StackingClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                DecisionTreeClassifier(criterion=&#x27;entropy&#x27;,
                                                       max_depth=3,
                                                       min_samples_leaf=3,
                                                       random_state=88888888)),
                               (&#x27;rf&#x27;,
                                RandomForestClassifier(criterion=&#x27;entropy&#x27;,
                                                       max_depth=5,
                                                       min_samples_leaf=3,
                                                       random_state=88888888)),
                               (&#x27;svm&#x27;,
                                SVC(kernel=&#x27;linear&#x27;, probability=True,
                                    random_state=88888888))],
                   final_estimator=LogisticRegression(max_iter=5000,
                                                      penalty=None,
                                                      random_state=88888888,
                                                      solver=&#x27;saga&#x27;))</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>dt</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;DecisionTreeClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeClassifier.html">?<span>Documentation for DecisionTreeClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeClassifier(criterion=&#x27;entropy&#x27;, max_depth=3, min_samples_leaf=3,
                       random_state=88888888)</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>rf</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(criterion=&#x27;entropy&#x27;, max_depth=5, min_samples_leaf=3,
                       random_state=88888888)</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>svm</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SVC<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html">?<span>Documentation for SVC</span></a></label><div class="sk-toggleable__content fitted"><pre>SVC(kernel=&#x27;linear&#x27;, probability=True, random_state=88888888)</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>final_estimator</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(max_iter=5000, penalty=None, random_state=88888888,
                   solver=&#x27;saga&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Identifying the best model
##################################
stacked_balanced_class_best_model_upsampled = stacked_balanced_class_grid_search.best_estimator_
```


```python
##################################
# Evaluating the F1 scores
# on the training, cross-validation, and validation data
##################################
stacked_balanced_class_best_model_upsampled_f1_cv = stacked_balanced_class_grid_search.best_score_
stacked_balanced_class_best_model_upsampled_f1_train_smote = f1_score(y_train_smote, stacked_balanced_class_best_model_upsampled.predict(X_train_smote))
stacked_balanced_class_best_model_upsampled_f1_validation = f1_score(y_validation, stacked_balanced_class_best_model_upsampled.predict(X_validation))
```


```python
##################################
# Identifying the optimal model
##################################
print('Best Stacked Model using the SMOTE-Upsampled Train Data: ')
print(f"Best Stacked Model Parameters: {stacked_balanced_class_grid_search.best_params_}")
```

    Best Stacked Model using the SMOTE-Upsampled Train Data: 
    Best Stacked Model Parameters: {'stacked_model__dt__max_depth': 3, 'stacked_model__final_estimator__class_weight': None, 'stacked_model__final_estimator__penalty': None, 'stacked_model__rf__max_depth': 5, 'stacked_model__svm__C': 1.0}
    


```python
##################################
# Summarizing the F1 score results
# on the training and cross-validated data
# to assess overfitting optimism
##################################
print(f"F1 Score on Cross-Validated Data: {stacked_balanced_class_best_model_upsampled_f1_cv:.4f}")
print(f"F1 Score on Training Data: {stacked_balanced_class_best_model_upsampled_f1_train_smote:.4f}")
print("\nClassification Report on Training Data:\n", classification_report(y_train_smote, stacked_balanced_class_best_model_upsampled.predict(X_train_smote)))
```

    F1 Score on Cross-Validated Data: 0.9489
    F1 Score on Training Data: 0.9568
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
               0       0.95      0.96      0.96       151
               1       0.96      0.95      0.96       151
    
        accuracy                           0.96       302
       macro avg       0.96      0.96      0.96       302
    weighted avg       0.96      0.96      0.96       302
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the training data
##################################
cm_raw = confusion_matrix(y_train_smote, stacked_balanced_class_best_model_upsampled.predict(X_train_smote))
cm_normalized = confusion_matrix(y_train_smote, stacked_balanced_class_best_model_upsampled.predict(X_train_smote), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Stacked Model on Training Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Stacked Model on Training Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_190_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
##################################
print(f"F1 Score on Validation Data: {stacked_balanced_class_best_model_upsampled_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_validation, stacked_balanced_class_best_model_upsampled.predict(X_validation)))
```

    F1 Score on Validation Data: 0.9615
    
    Classification Report on Validation Data:
                   precision    recall  f1-score   support
    
               0       0.80      0.57      0.67         7
               1       0.94      0.98      0.96        51
    
        accuracy                           0.93        58
       macro avg       0.87      0.78      0.81        58
    weighted avg       0.93      0.93      0.93        58
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the validation data
##################################
cm_raw = confusion_matrix(y_validation, stacked_balanced_class_best_model_upsampled.predict(X_validation))
cm_normalized = confusion_matrix(y_validation, stacked_balanced_class_best_model_upsampled.predict(X_validation), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Stacked Model on Validation Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Stacked Model on Validation Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_192_0.png)
    



```python
##################################
# Obtaining the logit values (log-odds)
# from the decision function for training data
##################################
stacked_balanced_class_best_model_upsampled_logit_values = stacked_balanced_class_best_model_upsampled.decision_function(X_train_smote)
```


```python
##################################
# Obtaining the estimated probabilities 
# for the positive class (LUNG_CANCER=YES) for training data
##################################
stacked_balanced_class_best_model_upsampled_probabilities = stacked_balanced_class_best_model_upsampled.predict_proba(X_train_smote)[:, 1]
```


```python
##################################
# Sorting the values to generate
# a smoother curve
##################################
stacked_balanced_class_best_model_upsampled_sorted_indices = np.argsort(stacked_balanced_class_best_model_upsampled_logit_values)
stacked_balanced_class_best_model_upsampled_logit_values_sorted = stacked_balanced_class_best_model_upsampled_logit_values[stacked_balanced_class_best_model_upsampled_sorted_indices]
stacked_balanced_class_best_model_upsampled_probabilities_sorted = stacked_balanced_class_best_model_upsampled_probabilities[stacked_balanced_class_best_model_upsampled_sorted_indices]
```


```python
##################################
# Plotting the estimated logistic curve
# using the logit values
# and estimated probabilities
# obtained from the training data
##################################
plt.figure(figsize=(17, 8))
plt.plot(stacked_balanced_class_best_model_upsampled_logit_values_sorted, 
         stacked_balanced_class_best_model_upsampled_probabilities_sorted, label='Classification Model Logistic Curve', color='black')
plt.ylim(-0.05, 1.05)
plt.xlim(-8.00, 8.00)
target_0_indices = y_train_smote == 0
target_1_indices = y_train_smote == 1
plt.scatter(stacked_balanced_class_best_model_upsampled_logit_values[target_0_indices], 
            stacked_balanced_class_best_model_upsampled_probabilities[target_0_indices], 
            color='blue', alpha=0.40, s=100, marker= 'o', edgecolor='k', label='LUNG_CANCER=NO')
plt.scatter(stacked_balanced_class_best_model_upsampled_logit_values[target_1_indices], 
            stacked_balanced_class_best_model_upsampled_probabilities[target_1_indices], 
            color='red', alpha=0.40, s=100, marker='o', edgecolor='k', label='LUNG_CANCER=YES')
plt.axhline(0.5, color='green', linestyle='--', label='Classification Model Threshold')
plt.title('Logistic Curve (Upsampled Training Data): Stacked Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_196_0.png)
    



```python
##################################
# Saving the best stacked model
# developed from the upsampled training data
################################## 
joblib.dump(stacked_balanced_class_best_model_upsampled, 
            os.path.join("..", MODELS_PATH, "stacked_balanced_class_best_model_upsampled.pkl"))
```




    ['..\\models\\stacked_balanced_class_best_model_upsampled.pkl']



### 1.6.6 Model Fitting using Downsampled Training Data | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.6"></a>

#### 1.6.6.1 Individual Classifier <a class="anchor" id="1.6.6.1"></a>

[Condensed Nearest Neighbors](https://ieeexplore.ieee.org/document/1054155) is a prototype selection algorithm that aims to select a subset of instances from the original dataset, discarding redundant and less informative instances. The algorithm works by iteratively adding instances to the subset, starting with an empty set. At each iteration, an instance is added if it is not correctly classified by the current subset. The decision to add or discard an instance is based on its performance on a k-nearest neighbors classifier. If an instance is misclassified by the current subset's k-nearest neighbors, it is added to the subset. The process is repeated until no new instances are added to the subset. The resulting subset is a condensed representation of the dataset that retains the essential information needed for classification.

[Logistic Regression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=360300) models the relationship between the probability of an event (among two outcome levels) by having the log-odds of the event be a linear combination of a set of predictors weighted by their respective parameter estimates. The parameters are estimated via maximum likelihood estimation by testing different values through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimates. Given the optimal parameters, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability.

[Class Weights](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) are used to assign different levels of importance to different classes when the distribution of instances across different classes in a classification problem is not equal. By assigning higher weights to the minority class, the model is encouraged to give more attention to correctly predicting instances from the minority class. Class weights are incorporated into the loss function during training. The loss for each instance is multiplied by its corresponding class weight. This means that misclassifying an instance from the minority class will have a greater impact on the overall loss than misclassifying an instance from the majority class. The use of class weights helps balance the influence of each class during training, mitigating the impact of class imbalance. It provides a way to focus the learning process on the classes that are underrepresented in the training data.

[Hyperparameter Tuning](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) is an iterative process that involves experimenting with different hyperparameter combinations, evaluating the model's performance, and refining the hyperparameter values to achieve the best possible performance on new, unseen data - aimed at building effective and well-generalizing machine learning models. A model's performance depends not only on the learned parameters (weights) during training but also on hyperparameters, which are external configuration settings that cannot be learned from the data.

1. The optimal [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (**individual classifier**) from the 5-fold cross-validation of **train data (CNN-downsampled)** contained the following hyperparameters:
    * <span style="color: #FF0000">penalty</span> = L2
    * <span style="color: #FF0000">class_weight</span> = balanced
    * <span style="color: #FF0000">solver</span> = saga
    * <span style="color: #FF0000">max_iter</span> = 500
    * <span style="color: #FF0000">random_state</span> = 88888888
2. The **F1 scores** estimated for the different data subsets were as follows:
    * **train data (CNN-downsampled)** = 0.8533
    * **train data (cross-validated)** = 0.7537
    * **validation data** = 0.9709
3. High overfitting noted based on the large difference in the apparent and cross-validated **F1 scores**.


```python
##################################
# Fitting the model on the 
# downsampled training data
##################################
individual_unbalanced_class_grid_search.fit(X_train_cnn, y_train_cnn)
```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits
    




<style>#sk-container-id-5 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-5 {
  color: var(--sklearn-color-text);
}

#sk-container-id-5 pre {
  padding: 0;
}

#sk-container-id-5 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-5 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-5 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-5 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-5 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-5 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-5 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-5 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-5 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-5 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-5 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-5 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-5 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-5 div.sk-label label.sk-toggleable__label,
#sk-container-id-5 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-5 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-5 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-5 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-5 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-5 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-5 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-5 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-5 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-5 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;individual_model&#x27;,
                 LogisticRegression(class_weight=&#x27;balanced&#x27;, max_iter=5000,
                                    random_state=88888888, solver=&#x27;saga&#x27;))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-23" type="checkbox" ><label for="sk-estimator-id-23" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(class_weight=&#x27;balanced&#x27;, max_iter=5000,
                   random_state=88888888, solver=&#x27;saga&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Identifying the best model
##################################
individual_unbalanced_class_best_model_downsampled = individual_unbalanced_class_grid_search.best_estimator_
```


```python
##################################
# Evaluating the F1 scores
# on the training, cross-validation, and validation data
##################################
individual_unbalanced_class_best_model_downsampled_f1_cv = individual_unbalanced_class_grid_search.best_score_
individual_unbalanced_class_best_model_downsampled_f1_train_cnn = f1_score(y_train_cnn, individual_unbalanced_class_best_model_downsampled.predict(X_train_cnn))
individual_unbalanced_class_best_model_downsampled_f1_validation = f1_score(y_validation, individual_unbalanced_class_best_model_downsampled.predict(X_validation))
```


```python
##################################
# Identifying the optimal model
##################################
print('Best Individual Model using the CNN-Downsampled Train Data: ')
print(f"Best Individual Model Parameters: {individual_unbalanced_class_grid_search.best_params_}")
```

    Best Individual Model using the CNN-Downsampled Train Data: 
    Best Individual Model Parameters: {'individual_model__class_weight': 'balanced', 'individual_model__penalty': 'l2'}
    


```python
##################################
# Summarizing the F1 score results
# on the training and cross-validated data
# to assess overfitting optimism
##################################
print(f"F1 Score on Cross-Validated Data: {individual_unbalanced_class_best_model_downsampled_f1_cv:.4f}")
print(f"F1 Score on Training Data: {individual_unbalanced_class_best_model_downsampled_f1_train_cnn:.4f}")
print("\nClassification Report on Training Data:\n", classification_report(y_train_cnn, individual_unbalanced_class_best_model_downsampled.predict(X_train_cnn)))
```

    F1 Score on Cross-Validated Data: 0.7537
    F1 Score on Training Data: 0.8533
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
               0       0.72      0.82      0.77        22
               1       0.89      0.82      0.85        39
    
        accuracy                           0.82        61
       macro avg       0.80      0.82      0.81        61
    weighted avg       0.83      0.82      0.82        61
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the training data
##################################
cm_raw = confusion_matrix(y_train_cnn, individual_unbalanced_class_best_model_downsampled.predict(X_train_cnn))
cm_normalized = confusion_matrix(y_train_cnn, individual_unbalanced_class_best_model_downsampled.predict(X_train_cnn), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Individual Model on Training Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Individual Model on Training Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_205_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
##################################
print(f"F1 Score on Validation Data: {individual_unbalanced_class_best_model_downsampled_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_validation, individual_unbalanced_class_best_model_downsampled.predict(X_validation)))
```

    F1 Score on Validation Data: 0.9709
    
    Classification Report on Validation Data:
                   precision    recall  f1-score   support
    
               0       0.83      0.71      0.77         7
               1       0.96      0.98      0.97        51
    
        accuracy                           0.95        58
       macro avg       0.90      0.85      0.87        58
    weighted avg       0.95      0.95      0.95        58
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the validation data
##################################
cm_raw = confusion_matrix(y_validation, individual_unbalanced_class_best_model_downsampled.predict(X_validation))
cm_normalized = confusion_matrix(y_validation, individual_unbalanced_class_best_model_downsampled.predict(X_validation), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Individual Model on Validation Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Individual Model on Validation Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_207_0.png)
    



```python
##################################
# Obtaining the logit values (log-odds)
# from the decision function for training data
##################################
individual_unbalanced_class_best_model_downsampled_logit_values = individual_unbalanced_class_best_model_downsampled.decision_function(X_train_cnn)
```


```python
##################################
# Obtaining the estimated probabilities 
# for the positive class (LUNG_CANCER=YES) for training data
##################################
individual_unbalanced_class_best_model_downsampled_probabilities = individual_unbalanced_class_best_model_downsampled.predict_proba(X_train_cnn)[:, 1]
```


```python
##################################
# Sorting the values to generate
# a smoother curve
##################################
individual_unbalanced_class_best_model_downsampled_sorted_indices = np.argsort(individual_unbalanced_class_best_model_downsampled_logit_values)
individual_unbalanced_class_best_model_downsampled_logit_values_sorted = individual_unbalanced_class_best_model_downsampled_logit_values[individual_unbalanced_class_best_model_downsampled_sorted_indices]
individual_unbalanced_class_best_model_downsampled_probabilities_sorted = individual_unbalanced_class_best_model_downsampled_probabilities[individual_unbalanced_class_best_model_downsampled_sorted_indices]
```


```python
##################################
# Plotting the estimated logistic curve
# using the logit values
# and estimated probabilities
# obtained from the training data
##################################
plt.figure(figsize=(17, 8))
plt.plot(individual_unbalanced_class_best_model_downsampled_logit_values_sorted, 
         individual_unbalanced_class_best_model_downsampled_probabilities_sorted, label='Classification Model Logistic Curve', color='black')
plt.ylim(-0.05, 1.05)
plt.xlim(-8.00, 8.00)
target_0_indices = y_train_cnn == 0
target_1_indices = y_train_cnn == 1
plt.scatter(individual_unbalanced_class_best_model_downsampled_logit_values[target_0_indices], 
            individual_unbalanced_class_best_model_downsampled_probabilities[target_0_indices], 
            color='blue', alpha=0.40, s=100, marker= 'o', edgecolor='k', label='LUNG_CANCER=NO')
plt.scatter(individual_unbalanced_class_best_model_downsampled_logit_values[target_1_indices], 
            individual_unbalanced_class_best_model_downsampled_probabilities[target_1_indices], 
            color='red', alpha=0.40, s=100, marker='o', edgecolor='k', label='LUNG_CANCER=YES')
plt.axhline(0.5, color='green', linestyle='--', label='Classification Model Threshold')
plt.title('Logistic Curve (Downsampled Training Data): Individual Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_211_0.png)
    



```python
##################################
# Saving the best individual model
# developed from the downsampled training data
################################## 
joblib.dump(individual_unbalanced_class_best_model_downsampled, 
            os.path.join("..", MODELS_PATH, "individual_unbalanced_class_best_model_downsampled.pkl"))
```




    ['..\\models\\individual_unbalanced_class_best_model_downsampled.pkl']



#### 1.6.6.2 Stacked Classifier <a class="anchor" id="1.6.6.2"></a>

[Condensed Nearest Neighbors](https://ieeexplore.ieee.org/document/1054155) is a prototype selection algorithm that aims to select a subset of instances from the original dataset, discarding redundant and less informative instances. The algorithm works by iteratively adding instances to the subset, starting with an empty set. At each iteration, an instance is added if it is not correctly classified by the current subset. The decision to add or discard an instance is based on its performance on a k-nearest neighbors classifier. If an instance is misclassified by the current subset's k-nearest neighbors, it is added to the subset. The process is repeated until no new instances are added to the subset. The resulting subset is a condensed representation of the dataset that retains the essential information needed for classification.

[Logistic Regression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=360300) models the relationship between the probability of an event (among two outcome levels) by having the log-odds of the event be a linear combination of a set of predictors weighted by their respective parameter estimates. The parameters are estimated via maximum likelihood estimation by testing different values through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimates. Given the optimal parameters, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability.

[Decision Trees](https://www.semanticscholar.org/paper/Classification-and-Regression-Trees-Breiman-Friedman/8017699564136f93af21575810d557dba1ee6fc6) create a model that predicts the class label of a sample based on input features. A decision tree consists of nodes that represent decisions or choices, edges which connect nodes and represent the possible outcomes of a decision and leaf (or terminal) nodes which represent the final decision or the predicted class label. The decision-making process involves feature selection (at each internal node, the algorithm decides which feature to split on based on a certain criterion including gini impurity or entropy), splitting criteria (the splitting criteria aim to find the feature and its corresponding threshold that best separates the data into different classes. The goal is to increase homogeneity within each resulting subset), recursive splitting (the process of feature selection and splitting continues recursively, creating a tree structure. The dataset is partitioned at each internal node based on the chosen feature, and the process repeats for each subset) and stopping criteria (the recursion stops when a certain condition is met, known as a stopping criterion. Common stopping criteria include a maximum depth for the tree, a minimum number of samples required to split a node, or a minimum number of samples in a leaf node.)

[Random Forest](https://link.springer.com/article/10.1023/A:1010933404324) is an ensemble learning method made up of a large set of small decision trees called estimators, with each producing its own prediction. The random forest model aggregates the predictions of the estimators to produce a more accurate prediction. The algorithm involves bootstrap aggregating (where smaller subsets of the training data are repeatedly subsampled with replacement), random subspacing (where a subset of features are sampled and used to train each individual estimator), estimator training (where unpruned decision trees are formulated for each estimator) and inference by aggregating the predictions of all estimators.

[Support Vector Machine](https://dl.acm.org/doi/10.1145/130385.130401) plots each observation in an N-dimensional space corresponding to the number of features in the data set and finds a hyperplane that maximally separates the different classes by a maximally large margin (which is defined as the distance between the hyperplane and the closest data points from each class). The algorithm applies kernel transformation by mapping non-linearly separable data using the similarities between the points in a high-dimensional feature space for improved discrimination.

[Class Weights](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) are used to assign different levels of importance to different classes when the distribution of instances across different classes in a classification problem is not equal. By assigning higher weights to the minority class, the model is encouraged to give more attention to correctly predicting instances from the minority class. Class weights are incorporated into the loss function during training. The loss for each instance is multiplied by its corresponding class weight. This means that misclassifying an instance from the minority class will have a greater impact on the overall loss than misclassifying an instance from the majority class. The use of class weights helps balance the influence of each class during training, mitigating the impact of class imbalance. It provides a way to focus the learning process on the classes that are underrepresented in the training data.

[Hyperparameter Tuning](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) is an iterative process that involves experimenting with different hyperparameter combinations, evaluating the model's performance, and refining the hyperparameter values to achieve the best possible performance on new, unseen data - aimed at building effective and well-generalizing machine learning models. A model's performance depends not only on the learned parameters (weights) during training but also on hyperparameters, which are external configuration settings that cannot be learned from the data. 

[Model Stacking](https://www.manning.com/books/ensemble-methods-for-machine-learning) - also known as stacked generalization, is an ensemble approach which involves creating a variety of base learners and using them to create intermediate predictions, one for each learned model. A meta-model is incorporated that gains knowledge of the same target from intermediate predictions. Unlike bagging, in stacking, the models are typically different (e.g. not all decision trees) and fit on the same dataset (e.g. instead of samples of the training dataset). Unlike boosting, in stacking, a single model is used to learn how to best combine the predictions from the contributing models (e.g. instead of a sequence of models that correct the predictions of prior models). Stacking is appropriate when the predictions made by the base learners or the errors in predictions made by the models have minimal correlation. Achieving an improvement in performance is dependent upon the choice of base learners and whether they are sufficiently skillful in their predictions.

1. The optimal [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) (**base learner**) determined from the 5-fold cross-validation of **train data (CNN-downsampled)** contained the following hyperparameters:
    * <span style="color: #FF0000">max_depth</span> = 3
    * <span style="color: #FF0000">class_weight</span> = balanced
    * <span style="color: #FF0000">criterion</span> = entropy
    * <span style="color: #FF0000">min_samples_leaf</span> = 3
    * <span style="color: #FF0000">random_state</span> = 88888888
2. The optimal [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#) (**base learner**) determined from the 5-fold cross-validation of **train data (CNN-downsampled)** contained the following hyperparameters:
    * <span style="color: #FF0000">max_depth</span> = 3
    * <span style="color: #FF0000">class_weight</span> = balanced
    * <span style="color: #FF0000">criterion</span> = entropy
    * <span style="color: #FF0000">max_features</span> = sqrt
    * <span style="color: #FF0000">min_samples_leaf</span> = 3
    * <span style="color: #FF0000">random_state</span> = 88888888
3. The optimal [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) (**base learner**) determined from the 5-fold cross-validation of **train data (CNN-downsampled)** contained the following hyperparameters:
    * <span style="color: #FF0000">C</span> = 1.00
    * <span style="color: #FF0000">class_weight</span> = balanced
    * <span style="color: #FF0000">kernel</span> = linear
    * <span style="color: #FF0000">probability</span> = true
    * <span style="color: #FF0000">random_state</span> = 88888888  
4. The optimal [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (**meta-learner**) determined from the 5-fold cross-validation of **train data (CNN-downsampled)** contained the following hyperparameters:
    * <span style="color: #FF0000">penalty</span> = none
    * <span style="color: #FF0000">class_weight</span> = balanced
    * <span style="color: #FF0000">solver</span> = saga
    * <span style="color: #FF0000">max_iter</span> = 500
    * <span style="color: #FF0000">random_state</span> = 88888888
5. The **F1 scores** estimated for the different data subsets were as follows:
    * **train data (CNN-downsampled)** = 0.8219
    * **train data (cross-validated)** = 0.7531
    * **validation data** = 0.9524
6. High overfitting noted based on the large difference in the apparent and cross-validated **F1 scores**.


```python
##################################
# Fitting the model on the 
# downsampled training data
##################################
stacked_unbalanced_class_grid_search.fit(X_train_cnn, y_train_cnn)
```

    Fitting 5 folds for each of 24 candidates, totalling 120 fits
    




<style>#sk-container-id-6 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-6 {
  color: var(--sklearn-color-text);
}

#sk-container-id-6 pre {
  padding: 0;
}

#sk-container-id-6 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-6 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-6 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-6 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-6 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-6 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-6 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-6 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-6 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-6 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-6 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-6 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-6 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-6 div.sk-label label.sk-toggleable__label,
#sk-container-id-6 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-6 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-6 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-6 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-6 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-6 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-6 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-6 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-6 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-6 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;stacked_model&#x27;,
                                        StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                                                        DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;,
                                                                                               criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;rf&#x27;,
                                                                        RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                                                                               criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;svm&#x27;,
                                                                        SVC(class_weight=&#x27;b...
                                                           final_estimator=LogisticRegression(max_iter=5000,
                                                                                              random_state=88888888,
                                                                                              solver=&#x27;saga&#x27;)))]),
             n_jobs=-1,
             param_grid={&#x27;stacked_model__dt__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__final_estimator__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;stacked_model__final_estimator__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;,
                                                                     None],
                         &#x27;stacked_model__rf__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__svm__C&#x27;: [0.5, 1.0]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-24" type="checkbox" ><label for="sk-estimator-id-24" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;stacked_model&#x27;,
                                        StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                                                        DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;,
                                                                                               criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;rf&#x27;,
                                                                        RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                                                                               criterion=&#x27;entropy&#x27;,
                                                                                               min_samples_leaf=3,
                                                                                               random_state=88888888)),
                                                                       (&#x27;svm&#x27;,
                                                                        SVC(class_weight=&#x27;b...
                                                           final_estimator=LogisticRegression(max_iter=5000,
                                                                                              random_state=88888888,
                                                                                              solver=&#x27;saga&#x27;)))]),
             n_jobs=-1,
             param_grid={&#x27;stacked_model__dt__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__final_estimator__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;stacked_model__final_estimator__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;,
                                                                     None],
                         &#x27;stacked_model__rf__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__svm__C&#x27;: [0.5, 1.0]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-25" type="checkbox" ><label for="sk-estimator-id-25" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;stacked_model&#x27;,
                 StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                                 DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;,
                                                                        criterion=&#x27;entropy&#x27;,
                                                                        max_depth=3,
                                                                        min_samples_leaf=3,
                                                                        random_state=88888888)),
                                                (&#x27;rf&#x27;,
                                                 RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                                                        criterion=&#x27;entropy&#x27;,
                                                                        max_depth=3,
                                                                        min_samples_leaf=3,
                                                                        random_state=88888888)),
                                                (&#x27;svm&#x27;,
                                                 SVC(class_weight=&#x27;balanced&#x27;,
                                                     kernel=&#x27;linear&#x27;,
                                                     probability=True,
                                                     random_state=88888888))],
                                    final_estimator=LogisticRegression(class_weight=&#x27;balanced&#x27;,
                                                                       max_iter=5000,
                                                                       penalty=None,
                                                                       random_state=88888888,
                                                                       solver=&#x27;saga&#x27;)))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-26" type="checkbox" ><label for="sk-estimator-id-26" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;stacked_model: StackingClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.StackingClassifier.html">?<span>Documentation for stacked_model: StackingClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;,
                                                       criterion=&#x27;entropy&#x27;,
                                                       max_depth=3,
                                                       min_samples_leaf=3,
                                                       random_state=88888888)),
                               (&#x27;rf&#x27;,
                                RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                                       criterion=&#x27;entropy&#x27;,
                                                       max_depth=3,
                                                       min_samples_leaf=3,
                                                       random_state=88888888)),
                               (&#x27;svm&#x27;,
                                SVC(class_weight=&#x27;balanced&#x27;, kernel=&#x27;linear&#x27;,
                                    probability=True, random_state=88888888))],
                   final_estimator=LogisticRegression(class_weight=&#x27;balanced&#x27;,
                                                      max_iter=5000,
                                                      penalty=None,
                                                      random_state=88888888,
                                                      solver=&#x27;saga&#x27;))</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>dt</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-27" type="checkbox" ><label for="sk-estimator-id-27" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;DecisionTreeClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeClassifier.html">?<span>Documentation for DecisionTreeClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;, criterion=&#x27;entropy&#x27;,
                       max_depth=3, min_samples_leaf=3, random_state=88888888)</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>rf</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-28" type="checkbox" ><label for="sk-estimator-id-28" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(class_weight=&#x27;balanced&#x27;, criterion=&#x27;entropy&#x27;,
                       max_depth=3, min_samples_leaf=3, random_state=88888888)</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>svm</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" ><label for="sk-estimator-id-29" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SVC<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html">?<span>Documentation for SVC</span></a></label><div class="sk-toggleable__content fitted"><pre>SVC(class_weight=&#x27;balanced&#x27;, kernel=&#x27;linear&#x27;, probability=True,
    random_state=88888888)</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><label>final_estimator</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-30" type="checkbox" ><label for="sk-estimator-id-30" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(class_weight=&#x27;balanced&#x27;, max_iter=5000, penalty=None,
                   random_state=88888888, solver=&#x27;saga&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Identifying the best model
##################################
stacked_unbalanced_class_best_model_downsampled = stacked_unbalanced_class_grid_search.best_estimator_
```


```python
##################################
# Evaluating the F1 scores
# on the training, cross-validation, and validation data
##################################
stacked_unbalanced_class_best_model_downsampled_f1_cv = stacked_unbalanced_class_grid_search.best_score_
stacked_unbalanced_class_best_model_downsampled_f1_train_cnn = f1_score(y_train_cnn, stacked_unbalanced_class_best_model_downsampled.predict(X_train_cnn))
stacked_unbalanced_class_best_model_downsampled_f1_validation = f1_score(y_validation, stacked_unbalanced_class_best_model_downsampled.predict(X_validation))
```


```python
##################################
# Identifying the optimal model
##################################
print('Best Stacked Model using the CNN-Downsampled Train Data: ')
print(f"Best Stacked Model Parameters: {stacked_unbalanced_class_grid_search.best_params_}")
```

    Best Stacked Model using the CNN-Downsampled Train Data: 
    Best Stacked Model Parameters: {'stacked_model__dt__max_depth': 3, 'stacked_model__final_estimator__class_weight': 'balanced', 'stacked_model__final_estimator__penalty': None, 'stacked_model__rf__max_depth': 3, 'stacked_model__svm__C': 1.0}
    


```python
##################################
# Summarizing the F1 score results
# on the training and cross-validated data
# to assess overfitting optimism
##################################
print(f"F1 Score on Cross-Validated Data: {stacked_unbalanced_class_best_model_downsampled_f1_cv:.4f}")
print(f"F1 Score on Training Data: {stacked_unbalanced_class_best_model_downsampled_f1_train_cnn:.4f}")
print("\nClassification Report on Training Data:\n", classification_report(y_train_cnn, stacked_unbalanced_class_best_model_downsampled.predict(X_train_cnn)))
```

    F1 Score on Cross-Validated Data: 0.7531
    F1 Score on Training Data: 0.8219
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
               0       0.67      0.82      0.73        22
               1       0.88      0.77      0.82        39
    
        accuracy                           0.79        61
       macro avg       0.77      0.79      0.78        61
    weighted avg       0.80      0.79      0.79        61
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the training data
##################################
cm_raw = confusion_matrix(y_train_cnn, stacked_unbalanced_class_best_model_downsampled.predict(X_train_cnn))
cm_normalized = confusion_matrix(y_train_cnn, stacked_unbalanced_class_best_model_downsampled.predict(X_train_cnn), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Stacked Model on Training Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Stacked Model on Training Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_219_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
##################################
print(f"F1 Score on Validation Data: {stacked_unbalanced_class_best_model_downsampled_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_validation, stacked_unbalanced_class_best_model_downsampled.predict(X_validation)))
```

    F1 Score on Validation Data: 0.9524
    
    Classification Report on Validation Data:
                   precision    recall  f1-score   support
    
               0       0.75      0.43      0.55         7
               1       0.93      0.98      0.95        51
    
        accuracy                           0.91        58
       macro avg       0.84      0.70      0.75        58
    weighted avg       0.90      0.91      0.90        58
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the validation data
##################################
cm_raw = confusion_matrix(y_validation, stacked_unbalanced_class_best_model_downsampled.predict(X_validation))
cm_normalized = confusion_matrix(y_validation, stacked_unbalanced_class_best_model_downsampled.predict(X_validation), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Best Stacked Model on Validation Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Best Stacked Model on Validation Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()
```


    
![png](output_221_0.png)
    



```python
##################################
# Obtaining the logit values (log-odds)
# from the decision function for training data
##################################
stacked_unbalanced_class_best_model_downsampled_logit_values = stacked_unbalanced_class_best_model_downsampled.decision_function(X_train_cnn)
```


```python
##################################
# Obtaining the estimated probabilities 
# for the positive class (LUNG_CANCER=YES) for training data
##################################
stacked_unbalanced_class_best_model_downsampled_probabilities = stacked_unbalanced_class_best_model_downsampled.predict_proba(X_train_cnn)[:, 1]
```


```python
##################################
# Sorting the values to generate
# a smoother curve
##################################
stacked_unbalanced_class_best_model_downsampled_sorted_indices = np.argsort(stacked_unbalanced_class_best_model_downsampled_logit_values)
stacked_unbalanced_class_best_model_downsampled_logit_values_sorted = stacked_unbalanced_class_best_model_downsampled_logit_values[stacked_unbalanced_class_best_model_downsampled_sorted_indices]
stacked_unbalanced_class_best_model_downsampled_probabilities_sorted = stacked_unbalanced_class_best_model_downsampled_probabilities[stacked_unbalanced_class_best_model_downsampled_sorted_indices]
```


```python
##################################
# Plotting the estimated logistic curve
# using the logit values
# and estimated probabilities
# obtained from the training data
##################################
plt.figure(figsize=(17, 8))
plt.plot(stacked_unbalanced_class_best_model_downsampled_logit_values_sorted, 
         stacked_unbalanced_class_best_model_downsampled_probabilities_sorted, label='Classification Model Logistic Curve', color='black')
plt.ylim(-0.05, 1.05)
plt.xlim(-8.00, 8.00)
target_0_indices = y_train_cnn == 0
target_1_indices = y_train_cnn == 1
plt.scatter(stacked_unbalanced_class_best_model_downsampled_logit_values[target_0_indices], 
            stacked_unbalanced_class_best_model_downsampled_probabilities[target_0_indices], 
            color='blue', alpha=0.40, s=100, marker= 'o', edgecolor='k', label='LUNG_CANCER=NO')
plt.scatter(stacked_unbalanced_class_best_model_downsampled_logit_values[target_1_indices], 
            stacked_unbalanced_class_best_model_downsampled_probabilities[target_1_indices], 
            color='red', alpha=0.40, s=100, marker='o', edgecolor='k', label='LUNG_CANCER=YES')
plt.axhline(0.5, color='green', linestyle='--', label='Classification Model Threshold')
plt.title('Logistic Curve (Downsampled Training Data): Stacked Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_225_0.png)
    



```python
##################################
# Saving the best stacked model
# developed from the downsampled training data
################################## 
joblib.dump(stacked_unbalanced_class_best_model_downsampled, 
            os.path.join("..", MODELS_PATH, "stacked_unbalanced_class_best_model_downsampled.pkl"))
```




    ['..\\models\\stacked_unbalanced_class_best_model_downsampled.pkl']



### 1.6.7 Model Selection <a class="anchor" id="1.6.7"></a>

1. The stacked classifier developed from the **train data (SMOTE-upsampled)** was selected as the final model by demonstrating the best validation **F1 score** with minimal overfitting :
    * **train data (SMOTE-upsampled)** = 0.9568
    * **train data (cross-validated)** = 0.9488
    * **validation data** = 0.9615
2. The final model configuration are described as follows:
    * **Base learner**: [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) with optimal hyperparameters:
        * <span style="color: #FF0000">max_depth</span> = 3
        * <span style="color: #FF0000">class_weight</span> = none
        * <span style="color: #FF0000">criterion</span> = entropy
        * <span style="color: #FF0000">min_samples_leaf</span> = 3
        * <span style="color: #FF0000">random_state</span> = 88888888
    * **Base learner**: [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#) with optimal hyperparameters:
        * <span style="color: #FF0000">max_depth</span> = 5
        * <span style="color: #FF0000">class_weight</span> = none
        * <span style="color: #FF0000">criterion</span> = entropy
        * <span style="color: #FF0000">max_features</span> = sqrt
        * <span style="color: #FF0000">min_samples_leaf</span> = 3
        * <span style="color: #FF0000">random_state</span> = 88888888
    * **Base learner**: [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) with optimal hyperparameters:
        * <span style="color: #FF0000">C</span> = 1.00
        * <span style="color: #FF0000">class_weight</span> = none
        * <span style="color: #FF0000">kernel</span> = linear
        * <span style="color: #FF0000">probability</span> = true
        * <span style="color: #FF0000">random_state</span> = 88888888  
    * **Meta-learner**: [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) with optimal hyperparameters:
        * <span style="color: #FF0000">penalty</span> = none
        * <span style="color: #FF0000">class_weight</span> = none
        * <span style="color: #FF0000">solver</span> = saga
        * <span style="color: #FF0000">max_iter</span> = 500
        * <span style="color: #FF0000">random_state</span> = 88888888



```python
##################################
# Gathering the F1 scores from 
# training, cross-validation and validation
##################################
set_labels = ['Train','Cross-Validation','Validation']
f1_plot = pd.DataFrame({'INDIVIDUAL_ORIGINAL_TRAIN': list([individual_unbalanced_class_best_model_original_f1_train,
                                                           individual_unbalanced_class_best_model_original_f1_cv,
                                                           individual_unbalanced_class_best_model_original_f1_validation]),
                        'STACKED_ORIGINAL_TRAIN': list([stacked_unbalanced_class_best_model_original_f1_train,
                                                        stacked_unbalanced_class_best_model_original_f1_cv,
                                                        stacked_unbalanced_class_best_model_original_f1_validation]),
                        'INDIVIDUAL_UPSAMPLED_TRAIN': list([individual_balanced_class_best_model_upsampled_f1_train_smote,
                                                           individual_balanced_class_best_model_upsampled_f1_cv,
                                                           individual_balanced_class_best_model_upsampled_f1_validation]),
                        'STACKED_UPSAMPLED_TRAIN': list([stacked_balanced_class_best_model_upsampled_f1_train_smote,
                                                        stacked_balanced_class_best_model_upsampled_f1_cv,
                                                        stacked_balanced_class_best_model_upsampled_f1_validation]),
                        'INDIVIDUAL_DOWNSAMPLED_TRAIN': list([individual_unbalanced_class_best_model_downsampled_f1_train_cnn,
                                                              individual_unbalanced_class_best_model_downsampled_f1_cv,
                                                              individual_unbalanced_class_best_model_downsampled_f1_validation]),
                        'STACKED_DOWNSAMPLED_TRAIN': list([stacked_unbalanced_class_best_model_downsampled_f1_train_cnn,
                                                           stacked_unbalanced_class_best_model_downsampled_f1_cv,
                                                           stacked_unbalanced_class_best_model_downsampled_f1_validation])},
                       index = set_labels)
display(f1_plot)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>INDIVIDUAL_ORIGINAL_TRAIN</th>
      <th>STACKED_ORIGINAL_TRAIN</th>
      <th>INDIVIDUAL_UPSAMPLED_TRAIN</th>
      <th>STACKED_UPSAMPLED_TRAIN</th>
      <th>INDIVIDUAL_DOWNSAMPLED_TRAIN</th>
      <th>STACKED_DOWNSAMPLED_TRAIN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train</th>
      <td>0.930556</td>
      <td>0.940351</td>
      <td>0.912162</td>
      <td>0.956811</td>
      <td>0.853333</td>
      <td>0.821918</td>
    </tr>
    <tr>
      <th>Cross-Validation</th>
      <td>0.911574</td>
      <td>0.912498</td>
      <td>0.910870</td>
      <td>0.948878</td>
      <td>0.753711</td>
      <td>0.753114</td>
    </tr>
    <tr>
      <th>Validation</th>
      <td>0.949495</td>
      <td>0.914894</td>
      <td>0.927835</td>
      <td>0.961538</td>
      <td>0.970874</td>
      <td>0.952381</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting all the F1 scores
# for all models
##################################
f1_plot = f1_plot.plot.barh(figsize=(10, 6), width=0.90)
f1_plot.set_xlim(0.00,1.00)
f1_plot.set_title("Classification Model Comparison by F1 Score")
f1_plot.set_xlabel("F1 Score")
f1_plot.set_ylabel("Data Set")
f1_plot.grid(False)
f1_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in f1_plot.containers:
    f1_plot.bar_label(container, fmt='%.5f', padding=-50, color='white', fontweight='bold')
```


    
![png](output_229_0.png)
    


### 1.6.8 Model Testing <a class="anchor" id="1.6.8"></a>

1. The selected stacked classifier developed from the **train data (SMOTE-upsampled)** also demonstrated a high **F1 score** on the independent test dataset:
    * **train data (SMOTE-upsampled)** = 0.9568
    * **train data (cross-validated)** = 0.9488
    * **validation data** = 0.9615
    * **test data** = 0.9489
    


```python
##################################
# Evaluating the F1 scores
# on the test data
##################################
individual_unbalanced_class_best_model_original_f1_test = f1_score(y_test, individual_unbalanced_class_best_model_original.predict(X_test))
stacked_unbalanced_class_best_model_original_f1_test = f1_score(y_test, stacked_unbalanced_class_best_model_original.predict(X_test))
individual_balanced_class_best_model_upsampled_f1_test = f1_score(y_test, individual_balanced_class_best_model_upsampled.predict(X_test))
stacked_balanced_class_best_model_upsampled_f1_test = f1_score(y_test, stacked_balanced_class_best_model_upsampled.predict(X_test))
individual_unbalanced_class_best_model_downsampled_f1_test = f1_score(y_test, individual_unbalanced_class_best_model_downsampled.predict(X_test))
stacked_unbalanced_class_best_model_downsampled_f1_test = f1_score(y_test, stacked_unbalanced_class_best_model_downsampled.predict(X_test))
```


```python
##################################
# Adding the the F1 score estimated
# from the test data
##################################
set_labels = ['Train','Cross-Validation','Validation','Test']
updated_f1_plot = pd.DataFrame({'INDIVIDUAL_ORIGINAL_TRAIN': list([individual_unbalanced_class_best_model_original_f1_train,
                                                                   individual_unbalanced_class_best_model_original_f1_cv,
                                                                   individual_unbalanced_class_best_model_original_f1_validation,
                                                                   individual_unbalanced_class_best_model_original_f1_test]),
                                'STACKED_ORIGINAL_TRAIN': list([stacked_unbalanced_class_best_model_original_f1_train,
                                                                stacked_unbalanced_class_best_model_original_f1_cv,
                                                                stacked_unbalanced_class_best_model_original_f1_validation,
                                                               stacked_unbalanced_class_best_model_original_f1_test]),
                                'INDIVIDUAL_UPSAMPLED_TRAIN': list([individual_balanced_class_best_model_upsampled_f1_train_smote,
                                                                    individual_balanced_class_best_model_upsampled_f1_cv,
                                                                    individual_balanced_class_best_model_upsampled_f1_validation,
                                                                   individual_balanced_class_best_model_upsampled_f1_test]),
                                'STACKED_UPSAMPLED_TRAIN': list([stacked_balanced_class_best_model_upsampled_f1_train_smote,
                                                                 stacked_balanced_class_best_model_upsampled_f1_cv,
                                                                 stacked_balanced_class_best_model_upsampled_f1_validation,
                                                                stacked_balanced_class_best_model_upsampled_f1_test]),
                                'INDIVIDUAL_DOWNSAMPLED_TRAIN': list([individual_unbalanced_class_best_model_downsampled_f1_train_cnn,
                                                                      individual_unbalanced_class_best_model_downsampled_f1_cv,
                                                                      individual_unbalanced_class_best_model_downsampled_f1_validation,
                                                                      individual_unbalanced_class_best_model_downsampled_f1_test]),
                                'STACKED_DOWNSAMPLED_TRAIN': list([stacked_unbalanced_class_best_model_downsampled_f1_train_cnn,
                                                                   stacked_unbalanced_class_best_model_downsampled_f1_cv,
                                                                   stacked_unbalanced_class_best_model_downsampled_f1_validation,
                                                                  stacked_unbalanced_class_best_model_downsampled_f1_test])},
                               index = set_labels)
display(updated_f1_plot)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>INDIVIDUAL_ORIGINAL_TRAIN</th>
      <th>STACKED_ORIGINAL_TRAIN</th>
      <th>INDIVIDUAL_UPSAMPLED_TRAIN</th>
      <th>STACKED_UPSAMPLED_TRAIN</th>
      <th>INDIVIDUAL_DOWNSAMPLED_TRAIN</th>
      <th>STACKED_DOWNSAMPLED_TRAIN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train</th>
      <td>0.930556</td>
      <td>0.940351</td>
      <td>0.912162</td>
      <td>0.956811</td>
      <td>0.853333</td>
      <td>0.821918</td>
    </tr>
    <tr>
      <th>Cross-Validation</th>
      <td>0.911574</td>
      <td>0.912498</td>
      <td>0.910870</td>
      <td>0.948878</td>
      <td>0.753711</td>
      <td>0.753114</td>
    </tr>
    <tr>
      <th>Validation</th>
      <td>0.949495</td>
      <td>0.914894</td>
      <td>0.927835</td>
      <td>0.961538</td>
      <td>0.970874</td>
      <td>0.952381</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>0.904762</td>
      <td>0.878049</td>
      <td>0.890625</td>
      <td>0.948905</td>
      <td>0.939394</td>
      <td>0.916031</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting all the F1 scores
# for all models
##################################
updated_f1_plot = updated_f1_plot.plot.barh(figsize=(10, 8), width=0.90)
updated_f1_plot.set_xlim(0.00,1.00)
updated_f1_plot.set_title("Classification Model Comparison by F1 Score")
updated_f1_plot.set_xlabel("F1 Score")
updated_f1_plot.set_ylabel("Data Set")
updated_f1_plot.grid(False)
updated_f1_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in updated_f1_plot.containers:
    updated_f1_plot.bar_label(container, fmt='%.5f', padding=-50, color='white', fontweight='bold')
```


    
![png](output_233_0.png)
    


### 1.6.9 Model Inference <a class="anchor" id="1.6.9"></a>

1. For the final selected stacked classifier developed from the **train data (SMOTE-upsampled)**, the contributions of the base learners, ranked by importance, are given as follows:
    * **Base learner**: [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#)
    * **Base learner**: [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    * **Base learner**: [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
3. For each base learner of the final selected stacked classifier developed from the **train data (SMOTE-upsampled)**, the contributions of the predictors, ranked by importance, are given as follows:
    * **Base learner**: [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#)
        * <span style="color: #FF0000">ALLERGY</span>
        * <span style="color: #FF0000">ALCOHOL_CONSUMING </span>
        * <span style="color: #FF0000">PEER_PRESSURE</span>
        * <span style="color: #FF0000">ANXIETY</span>
        * <span style="color: #FF0000">FATIGUE</span>
        * <span style="color: #FF0000">WHEEZING</span>
        * <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span>
        * <span style="color: #FF0000">COUGHING</span>
        * <span style="color: #FF0000">CHEST_PAIN</span>
        * <span style="color: #FF0000">YELLOW_FINGERS</span>
    * **Base learner**: [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
        * <span style="color: #FF0000">ALLERGY</span>
        * <span style="color: #FF0000">PEER_PRESSURE</span>
        * <span style="color: #FF0000">ALCOHOL_CONSUMING </span>
        * <span style="color: #FF0000">YELLOW_FINGERS</span>
    * **Base learner**: [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
        * <span style="color: #FF0000">ALLERGY</span>
        * <span style="color: #FF0000">PEER_PRESSURE</span>
        * <span style="color: #FF0000">ANXIETY</span>
        * <span style="color: #FF0000">FATIGUE</span>
        * <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span>
        * <span style="color: #FF0000">WHEEZING</span>
        * <span style="color: #FF0000">ALCOHOL_CONSUMING </span>
        * <span style="color: #FF0000">COUGHING</span>
        * <span style="color: #FF0000">CHEST_PAIN</span>
        * <span style="color: #FF0000">YELLOW_FINGERS</span>
4. Model inference involved indicating the characteristics and predicting the probability of the new case against the model training observations.
    * Characteristics based on all predictors used for generating the final selected stacked classifier
    * Predicted lung cancer probability based on the final selected stacked classifier logistic curve


```python
##################################
# Assigning as the final model
# the candidate model which 
# demonstrated the best performance
# on the test set
##################################
final_model = stacked_balanced_class_best_model_upsampled.named_steps['stacked_model']
final_model_base_learner = ['Stacked Model Base Learner: Decision Trees',
                            'Stacked Model Base Learner: Random Forest',
                            'Stacked Model Base Learner: Support Vector Machine']
```


```python
##################################
# Defining a function to compute and plot 
# the feature importance for a defined model
##################################
def plot_feature_importance(importance, feature_names, model_name):
    indices = np.argsort(importance)
    plt.figure(figsize=(17, 8))
    plt.title(f"Feature Importance - {model_name}")
    plt.barh(range(len(importance)), importance[indices], align="center")
    plt.yticks(range(len(importance)), [feature_names[i] for i in indices])
    plt.tight_layout()
    plt.show()
```


```python
##################################
# Defining the predictor names
##################################
feature_names = X_test.columns
```


```python
##################################
# Ranking the predictors based on model importance
# for each base learner using feature importance
# for tree-based models like DecisionTree and Random Forest
# and coefficients for linear models like SVC with linear kernel
##################################
for index, (name, model) in enumerate(final_model.named_estimators_.items()):
    if hasattr(model, 'feature_importances_'):  # For tree-based models like DecisionTree and RandomForest
        plot_feature_importance(model.feature_importances_, feature_names, model_name=final_model_base_learner[index])
    elif hasattr(model, 'coef_'):  # For linear models like SVC with linear kernel
        importance = np.abs(model.coef_).flatten()
        plot_feature_importance(importance, feature_names, model_name=final_model_base_learner[index])
```


    
![png](output_238_0.png)
    



    
![png](output_238_1.png)
    



    
![png](output_238_2.png)
    



```python
##################################
# Generating predictions from the 
# base learners to be used as input
# to the logistic regression meta-learner
##################################
base_learners_predictions = []
for name, model in final_model.named_estimators_.items():
    base_learners_predictions.append(model.predict_proba(X_test)[:, 1])
```


```python
##################################
# Stacking the base learners' predictions
##################################
meta_input = np.column_stack(base_learners_predictions)

##################################
# Defining the base learner model names
##################################
meta_feature_names = [f'Model Prediction - {x}' for x in final_model_base_learner]

##################################
# Ranking the predictors based on model importance
# for each meta-learner using coefficients
# for linear models like logistic regression
##################################
if hasattr(final_model.final_estimator_, 'coef_'):
    importance = np.abs(final_model.final_estimator_.coef_).flatten()
    plot_feature_importance(importance, meta_feature_names, model_name='Stacked Model Meta-Learner: Logistic Regression')
```


    
![png](output_240_0.png)
    



```python
##################################
# Rebuilding the upsampled training data
# for plotting categorical distributions
##################################
lung_cancer_train_smote = pd.concat([X_train_smote, y_train_smote], axis=1)
lung_cancer_train_smote.iloc[:,0:10] = lung_cancer_train_smote.iloc[:,0:10].replace({0: 'Absent', 1: 'Present'})
lung_cancer_train_smote['LUNG_CANCER'] = lung_cancer_train_smote['LUNG_CANCER'].astype('category')
lung_cancer_train_smote['LUNG_CANCER'] = lung_cancer_train_smote['LUNG_CANCER'].cat.rename_categories({0: 'No', 1: 'Yes'})
lung_cancer_train_smote[lung_cancer_train_smote.columns[0:11]] = lung_cancer_train_smote[lung_cancer_train_smote.columns[0:11]].astype('category')
lung_cancer_train_smote.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
      <th>LUNG_CANCER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the categorical distributions
# for a low-risk test case
##################################
fig, axs = plt.subplots(2, 5, figsize=(17, 8))

colors = ['blue','red']
level_order = ['Absent','Present']

sns.countplot(x='YELLOW_FINGERS', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 0], order=level_order, palette=colors)
axs[0, 0].set_title('YELLOW_FINGERS')
axs[0, 0].set_ylabel('Classification Model Training Case Count')
axs[0, 0].set_xlabel(None)
axs[0, 0].set_ylim(0, 200)
axs[0, 0].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 0].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='ANXIETY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 1], order=level_order, palette=colors)
axs[0, 1].set_title('ANXIETY')
axs[0, 1].set_ylabel('Classification Model Training Case Count')
axs[0, 1].set_xlabel(None)
axs[0, 1].set_ylim(0, 200)
axs[0, 1].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 1].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='PEER_PRESSURE', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 2], order=level_order, palette=colors)
axs[0, 2].set_title('PEER_PRESSURE')
axs[0, 2].set_ylabel('Classification Model Training Case Count')
axs[0, 2].set_xlabel(None)
axs[0, 2].set_ylim(0, 200)
axs[0, 2].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 2].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='FATIGUE', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 3], order=level_order, palette=colors)
axs[0, 3].set_title('FATIGUE')
axs[0, 3].set_ylabel('Classification Model Training Case Count')
axs[0, 3].set_xlabel(None)
axs[0, 3].set_ylim(0, 200)
axs[0, 3].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 3].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='ALLERGY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 4], order=level_order, palette=colors)
axs[0, 4].set_title('ALLERGY')
axs[0, 4].set_ylabel('Classification Model Training Case Count')
axs[0, 4].set_xlabel(None)
axs[0, 4].set_ylim(0, 200)
axs[0, 4].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 4].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='WHEEZING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 0], order=level_order, palette=colors)
axs[1, 0].set_title('WHEEZING')
axs[1, 0].set_ylabel('Classification Model Training Case Count')
axs[1, 0].set_xlabel(None)
axs[1, 0].set_ylim(0, 200)
axs[1, 0].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 0].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='ALCOHOL_CONSUMING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 1], order=level_order, palette=colors)
axs[1, 1].set_title('ALCOHOL_CONSUMING')
axs[1, 1].set_ylabel('Classification Model Training Case Count')
axs[1, 1].set_xlabel(None)
axs[1, 1].set_ylim(0, 200)
axs[1, 1].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 1].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='COUGHING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 2], order=level_order, palette=colors)
axs[1, 2].set_title('COUGHING')
axs[1, 2].set_ylabel('Classification Model Training Case Count')
axs[1, 2].set_xlabel(None)
axs[1, 2].set_ylim(0, 200)
axs[1, 2].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 2].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='SWALLOWING_DIFFICULTY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 3], order=level_order, palette=colors)
axs[1, 3].set_title('SWALLOWING_DIFFICULTY')
axs[1, 3].set_ylabel('Classification Model Training Case Count')
axs[1, 3].set_xlabel(None)
axs[1, 3].set_ylim(0, 200)
axs[1, 3].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 3].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='CHEST_PAIN', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 4], order=level_order, palette=colors)
axs[1, 4].set_title('CHEST_PAIN')
axs[1, 4].set_ylabel('Classification Model Training Case Count')
axs[1, 4].set_xlabel(None)
axs[1, 4].set_ylim(0, 200)
axs[1, 4].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 4].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

plt.tight_layout()
plt.show()
```


    
![png](output_242_0.png)
    



```python
##################################
# Plotting the estimated logistic curve
# of the final classification model
# involving a stacked model with
# a logistic regression meta-learner
# and random forest, SVC and decision tree
# base learners
##################################
plt.figure(figsize=(17, 8))
plt.plot(stacked_balanced_class_best_model_upsampled_logit_values_sorted, 
         stacked_balanced_class_best_model_upsampled_probabilities_sorted, label='Classification Model Logistic Curve', color='black')
plt.ylim(-0.05, 1.05)
plt.xlim(-6.00, 6.00)
target_0_indices = y_train_smote == 0
target_1_indices = y_train_smote == 1
plt.scatter(stacked_balanced_class_best_model_upsampled_logit_values[target_0_indices], 
            stacked_balanced_class_best_model_upsampled_probabilities[target_0_indices], 
            color='blue', alpha=0.40, s=100, marker= 'o', edgecolor='k', label='LUNG_CANCER=NO')
plt.scatter(stacked_balanced_class_best_model_upsampled_logit_values[target_1_indices], 
            stacked_balanced_class_best_model_upsampled_probabilities[target_1_indices], 
            color='red', alpha=0.40, s=100, marker='o', edgecolor='k', label='LUNG_CANCER=YES')
plt.axhline(0.5, color='green', linestyle='--', label='Classification Model Threshold')
plt.title('Final Classification Model: Stacked Model (Meta-Learner = Logistic Regression, Base Learners: Random Forest, Support Vector Classifier, Decision Tree)')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_243_0.png)
    



```python
##################################
# Describing the details of a 
# low-risk test case
##################################
X_sample = {"YELLOW_FINGERS":1,
            "ANXIETY":0,
            "PEER_PRESSURE":0,
            "FATIGUE":0,
            "ALLERGY":0,
            "WHEEZING":1,
            "ALCOHOL_CONSUMING":0,
            "COUGHING":0,
            "SWALLOWING_DIFFICULTY":1,
            "CHEST_PAIN":1}
X_test_sample = pd.DataFrame([X_sample])
X_test_sample.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Rebuilding the low-risk test case data
# for plotting categorical distributions
##################################
X_test_sample_category = X_test_sample.copy()
int_test_columns = X_test_sample_category.columns
X_test_sample_category[int_test_columns] = X_test_sample_category[int_test_columns].astype(object)
X_test_sample_category[int_test_columns] = X_test_sample_category[int_test_columns].replace({0: 'Absent', 1: 'Present'})
X_test_sample_category.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the categorical distributions
# for a low-risk test case
##################################
fig, axs = plt.subplots(2, 5, figsize=(17, 8))

colors = ['blue','red']
level_order = ['Absent','Present']

sns.countplot(x='YELLOW_FINGERS', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 0], order=level_order, palette=colors)
axs[0, 0].axvline(level_order.index(X_test_sample_category['YELLOW_FINGERS'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[0, 0].set_title('YELLOW_FINGERS')
axs[0, 0].set_ylabel('Classification Model Training Case Count')
axs[0, 0].set_xlabel(None)
axs[0, 0].set_ylim(0, 200)
axs[0, 0].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 0].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='ANXIETY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 1], order=level_order, palette=colors)
axs[0, 1].axvline(level_order.index(X_test_sample_category['ANXIETY'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[0, 1].set_title('ANXIETY')
axs[0, 1].set_ylabel('Classification Model Training Case Count')
axs[0, 1].set_xlabel(None)
axs[0, 1].set_ylim(0, 200)
axs[0, 1].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 1].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='PEER_PRESSURE', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 2], order=level_order, palette=colors)
axs[0, 2].axvline(level_order.index(X_test_sample_category['PEER_PRESSURE'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[0, 2].set_title('PEER_PRESSURE')
axs[0, 2].set_ylabel('Classification Model Training Case Count')
axs[0, 2].set_xlabel(None)
axs[0, 2].set_ylim(0, 200)
axs[0, 2].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 2].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='FATIGUE', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 3], order=level_order, palette=colors)
axs[0, 3].axvline(level_order.index(X_test_sample_category['FATIGUE'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[0, 3].set_title('FATIGUE')
axs[0, 3].set_ylabel('Classification Model Training Case Count')
axs[0, 3].set_xlabel(None)
axs[0, 3].set_ylim(0, 200)
axs[0, 3].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 3].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='ALLERGY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 4], order=level_order, palette=colors)
axs[0, 4].axvline(level_order.index(X_test_sample_category['ALLERGY'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[0, 4].set_title('ALLERGY')
axs[0, 4].set_ylabel('Classification Model Training Case Count')
axs[0, 4].set_xlabel(None)
axs[0, 4].set_ylim(0, 200)
axs[0, 4].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 4].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='WHEEZING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 0], order=level_order, palette=colors)
axs[1, 0].axvline(level_order.index(X_test_sample_category['WHEEZING'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[1, 0].set_title('WHEEZING')
axs[1, 0].set_ylabel('Classification Model Training Case Count')
axs[1, 0].set_xlabel(None)
axs[1, 0].set_ylim(0, 200)
axs[1, 0].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 0].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='ALCOHOL_CONSUMING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 1], order=level_order, palette=colors)
axs[1, 1].axvline(level_order.index(X_test_sample_category['ALCOHOL_CONSUMING'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[1, 1].set_title('ALCOHOL_CONSUMING')
axs[1, 1].set_ylabel('Classification Model Training Case Count')
axs[1, 1].set_xlabel(None)
axs[1, 1].set_ylim(0, 200)
axs[1, 1].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 1].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='COUGHING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 2], order=level_order, palette=colors)
axs[1, 2].axvline(level_order.index(X_test_sample_category['COUGHING'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[1, 2].set_title('COUGHING')
axs[1, 2].set_ylabel('Classification Model Training Case Count')
axs[1, 2].set_xlabel(None)
axs[1, 2].set_ylim(0, 200)
axs[1, 2].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 2].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='SWALLOWING_DIFFICULTY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 3], order=level_order, palette=colors)
axs[1, 3].axvline(level_order.index(X_test_sample_category['SWALLOWING_DIFFICULTY'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[1, 3].set_title('SWALLOWING_DIFFICULTY')
axs[1, 3].set_ylabel('Classification Model Training Case Count')
axs[1, 3].set_xlabel(None)
axs[1, 3].set_ylim(0, 200)
axs[1, 3].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 3].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='CHEST_PAIN', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 4], order=level_order, palette=colors)
axs[1, 4].axvline(level_order.index(X_test_sample_category['CHEST_PAIN'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[1, 4].set_title('CHEST_PAIN')
axs[1, 4].set_ylabel('Classification Model Training Case Count')
axs[1, 4].set_xlabel(None)
axs[1, 4].set_ylim(0, 200)
axs[1, 4].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 4].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

plt.tight_layout()
plt.show()
```


    
![png](output_246_0.png)
    



```python
##################################
# Computing the logit and estimated probability
# for the test case
##################################
X_sample_logit = stacked_balanced_class_best_model_upsampled.decision_function(X_test_sample)[0]
X_sample_probability = stacked_balanced_class_best_model_upsampled.predict_proba(X_test_sample)[0, 1]
X_sample_class = "Low-Risk" if X_sample_probability < 0.50 else "High-Risk"
print(f"Test Case Risk Index: {X_sample_logit}")
print(f"Test Case Probability: {X_sample_probability}")
print(f"Test Case Risk Category: {X_sample_class}")
```

    Test Case Risk Index: -1.2117837409390746
    Test Case Probability: 0.22938559072691203
    Test Case Risk Category: Low-Risk
    


```python
##################################
# Plotting the logit and estimated probability
# for the low-risk test case 
# in the estimated logistic curve
# of the final classification model
##################################
plt.figure(figsize=(17, 8))
plt.plot(stacked_balanced_class_best_model_upsampled_logit_values_sorted, 
         stacked_balanced_class_best_model_upsampled_probabilities_sorted, label='Classification Model Logistic Curve', color='black')
plt.ylim(-0.05, 1.05)
plt.xlim(-6.00, 6.00)
target_0_indices = y_train_smote == 0
target_1_indices = y_train_smote == 1
plt.axhline(0.5, color='green', linestyle='--', label='Classification Model Threshold')
plt.scatter(stacked_balanced_class_best_model_upsampled_logit_values[target_0_indices], 
            stacked_balanced_class_best_model_upsampled_probabilities[target_0_indices], 
            color='blue', alpha=0.20, s=100, marker= 'o', edgecolor='k', label='Classification Model Training Cases: LUNG_CANCER = No')
plt.scatter(stacked_balanced_class_best_model_upsampled_logit_values[target_1_indices], 
            stacked_balanced_class_best_model_upsampled_probabilities[target_1_indices], 
            color='red', alpha=0.20, s=100, marker='o', edgecolor='k', label='Classification Model Training Cases: LUNG_CANCER = Yes')
if X_sample_class == "Low-Risk":
    plt.scatter(X_sample_logit, X_sample_probability, color='blue', s=125, edgecolor='k', label='Test Case (Low-Risk)', marker= 's', zorder=5)
    plt.axvline(X_sample_logit, color='black', linestyle='--', linewidth=3)
    plt.axhline(X_sample_probability, color='black', linestyle='--', linewidth=3)
if X_sample_class == "High-Risk":
    plt.scatter(X_sample_logit, X_sample_probability, color='red', s=125, edgecolor='k', label='Test Case (High-Risk)', marker= 's', zorder=5)
    plt.axvline(X_sample_logit, color='black', linestyle='--', linewidth=3)
    plt.axhline(X_sample_probability, color='black', linestyle='--', linewidth=3)
plt.title('Final Classification Model: Stacked Model (Meta-Learner = Logistic Regression, Base Learners = Random Forest, Support Vector Classifier, Decision Tree)')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(False)
plt.legend(facecolor='white', framealpha=1, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)
plt.tight_layout(rect=[0, 0, 1.00, 0.95])
plt.show()
```


    
![png](output_248_0.png)
    



```python
##################################
# Describing the details of a 
# high-risk test case
##################################
X_sample = {"YELLOW_FINGERS":1,
            "ANXIETY":0,
            "PEER_PRESSURE":1,
            "FATIGUE":0,
            "ALLERGY":1,
            "WHEEZING":1,
            "ALCOHOL_CONSUMING":0,
            "COUGHING":1,
            "SWALLOWING_DIFFICULTY":1,
            "CHEST_PAIN":1}
X_test_sample = pd.DataFrame([X_sample])
X_test_sample.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Rebuilding the high-risk test case data
# for plotting categorical distributions
##################################
X_test_sample_category = X_test_sample.copy()
int_test_columns = X_test_sample_category.columns
X_test_sample_category[int_test_columns] = X_test_sample_category[int_test_columns].astype(object)
X_test_sample_category[int_test_columns] = X_test_sample_category[int_test_columns].replace({0: 'Absent', 1: 'Present'})
X_test_sample_category.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YELLOW_FINGERS</th>
      <th>ANXIETY</th>
      <th>PEER_PRESSURE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL_CONSUMING</th>
      <th>COUGHING</th>
      <th>SWALLOWING_DIFFICULTY</th>
      <th>CHEST_PAIN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Absent</td>
      <td>Present</td>
      <td>Present</td>
      <td>Present</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the categorical distributions
# for a low-risk test case
##################################
fig, axs = plt.subplots(2, 5, figsize=(17, 8))

colors = ['blue','red']
level_order = ['Absent','Present']

sns.countplot(x='YELLOW_FINGERS', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 0], order=level_order, palette=colors)
axs[0, 0].axvline(level_order.index(X_test_sample_category['YELLOW_FINGERS'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[0, 0].set_title('YELLOW_FINGERS')
axs[0, 0].set_ylabel('Classification Model Training Case Count')
axs[0, 0].set_xlabel(None)
axs[0, 0].set_ylim(0, 200)
axs[0, 0].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 0].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='ANXIETY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 1], order=level_order, palette=colors)
axs[0, 1].axvline(level_order.index(X_test_sample_category['ANXIETY'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[0, 1].set_title('ANXIETY')
axs[0, 1].set_ylabel('Classification Model Training Case Count')
axs[0, 1].set_xlabel(None)
axs[0, 1].set_ylim(0, 200)
axs[0, 1].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 1].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='PEER_PRESSURE', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 2], order=level_order, palette=colors)
axs[0, 2].axvline(level_order.index(X_test_sample_category['PEER_PRESSURE'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[0, 2].set_title('PEER_PRESSURE')
axs[0, 2].set_ylabel('Classification Model Training Case Count')
axs[0, 2].set_xlabel(None)
axs[0, 2].set_ylim(0, 200)
axs[0, 2].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 2].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='FATIGUE', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 3], order=level_order, palette=colors)
axs[0, 3].axvline(level_order.index(X_test_sample_category['FATIGUE'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[0, 3].set_title('FATIGUE')
axs[0, 3].set_ylabel('Classification Model Training Case Count')
axs[0, 3].set_xlabel(None)
axs[0, 3].set_ylim(0, 200)
axs[0, 3].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 3].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='ALLERGY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 4], order=level_order, palette=colors)
axs[0, 4].axvline(level_order.index(X_test_sample_category['ALLERGY'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[0, 4].set_title('ALLERGY')
axs[0, 4].set_ylabel('Classification Model Training Case Count')
axs[0, 4].set_xlabel(None)
axs[0, 4].set_ylim(0, 200)
axs[0, 4].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[0, 4].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='WHEEZING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 0], order=level_order, palette=colors)
axs[1, 0].axvline(level_order.index(X_test_sample_category['WHEEZING'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[1, 0].set_title('WHEEZING')
axs[1, 0].set_ylabel('Classification Model Training Case Count')
axs[1, 0].set_xlabel(None)
axs[1, 0].set_ylim(0, 200)
axs[1, 0].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 0].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='ALCOHOL_CONSUMING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 1], order=level_order, palette=colors)
axs[1, 1].axvline(level_order.index(X_test_sample_category['ALCOHOL_CONSUMING'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[1, 1].set_title('ALCOHOL_CONSUMING')
axs[1, 1].set_ylabel('Classification Model Training Case Count')
axs[1, 1].set_xlabel(None)
axs[1, 1].set_ylim(0, 200)
axs[1, 1].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 1].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='COUGHING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 2], order=level_order, palette=colors)
axs[1, 2].axvline(level_order.index(X_test_sample_category['COUGHING'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[1, 2].set_title('COUGHING')
axs[1, 2].set_ylabel('Classification Model Training Case Count')
axs[1, 2].set_xlabel(None)
axs[1, 2].set_ylim(0, 200)
axs[1, 2].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 2].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='SWALLOWING_DIFFICULTY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 3], order=level_order, palette=colors)
axs[1, 3].axvline(level_order.index(X_test_sample_category['SWALLOWING_DIFFICULTY'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[1, 3].set_title('SWALLOWING_DIFFICULTY')
axs[1, 3].set_ylabel('Classification Model Training Case Count')
axs[1, 3].set_xlabel(None)
axs[1, 3].set_ylim(0, 200)
axs[1, 3].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 3].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

sns.countplot(x='CHEST_PAIN', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 4], order=level_order, palette=colors)
axs[1, 4].axvline(level_order.index(X_test_sample_category['CHEST_PAIN'].iloc[0]), color='black', linestyle='--', linewidth=3)
axs[1, 4].set_title('CHEST_PAIN')
axs[1, 4].set_ylabel('Classification Model Training Case Count')
axs[1, 4].set_xlabel(None)
axs[1, 4].set_ylim(0, 200)
axs[1, 4].legend(title='LUNG_CANCER', loc='upper center')
for patch, color in zip(axs[1, 4].patches, ['blue','blue','red','red'] ):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)

plt.tight_layout()
plt.show()
```


    
![png](output_251_0.png)
    



```python
##################################
# Computing the logit and estimated probability
# for a high-risk test case
##################################
X_sample_logit = stacked_balanced_class_best_model_upsampled.decision_function(X_test_sample)[0]
X_sample_probability = stacked_balanced_class_best_model_upsampled.predict_proba(X_test_sample)[0, 1]
X_sample_class = "Low-Risk" if X_sample_probability < 0.50 else "High-Risk"
print(f"Test Case Risk Index: {X_sample_logit}")
print(f"Test Case Probability: {X_sample_probability}")
print(f"Test Case Risk Category: {X_sample_class}")
```

    Test Case Risk Index: 3.4784950973590973
    Test Case Probability: 0.9700696569701589
    Test Case Risk Category: High-Risk
    


```python
##################################
# Plotting the logit and estimated probability
# for the high-risk test case 
# in the estimated logistic curve
# of the final classification model
##################################
plt.figure(figsize=(17, 8))
plt.plot(stacked_balanced_class_best_model_upsampled_logit_values_sorted, 
         stacked_balanced_class_best_model_upsampled_probabilities_sorted, label='Classification Model Logistic Curve', color='black')
plt.ylim(-0.05, 1.05)
plt.xlim(-6.00, 6.00)
target_0_indices = y_train_smote == 0
target_1_indices = y_train_smote == 1
plt.axhline(0.5, color='green', linestyle='--', label='Classification Model Threshold')
plt.scatter(stacked_balanced_class_best_model_upsampled_logit_values[target_0_indices], 
            stacked_balanced_class_best_model_upsampled_probabilities[target_0_indices], 
            color='blue', alpha=0.20, s=100, marker= 'o', edgecolor='k', label='Classification Model Training Cases: LUNG_CANCER = No')
plt.scatter(stacked_balanced_class_best_model_upsampled_logit_values[target_1_indices], 
            stacked_balanced_class_best_model_upsampled_probabilities[target_1_indices], 
            color='red', alpha=0.20, s=100, marker='o', edgecolor='k', label='Classification Model Training Cases: LUNG_CANCER = Yes')
if X_sample_class == "Low-Risk":
    plt.scatter(X_sample_logit, X_sample_probability, color='blue', s=125, edgecolor='k', label='Test Case (Low-Risk)', marker= 's', zorder=5)
    plt.axvline(X_sample_logit, color='black', linestyle='--', linewidth=3)
    plt.axhline(X_sample_probability, color='black', linestyle='--', linewidth=3)
if X_sample_class == "High-Risk":
    plt.scatter(X_sample_logit, X_sample_probability, color='red', s=125, edgecolor='k', label='Test Case (High-Risk)', marker= 's', zorder=5)
    plt.axvline(X_sample_logit, color='black', linestyle='--', linewidth=3)
    plt.axhline(X_sample_probability, color='black', linestyle='--', linewidth=3)
plt.title('Final Classification Model: Stacked Model (Meta-Learner = Logistic Regression, Base Learners = Random Forest, Support Vector Classifier, Decision Tree)')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(False)
plt.legend(facecolor='white', framealpha=1, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)
plt.tight_layout(rect=[0, 0, 1.00, 0.95])
plt.show()
```


    
![png](output_253_0.png)
    


## 1.7. Predictive Model Deployment Using Streamlit and Streamlit Community Cloud <a class="anchor" id="1.7"></a>

[Streamlit](https://streamlit.io/) is an open-source Python library that simplifies the creation and deployment of web applications for machine learning and data science projects. It allows developers and data scientists to turn Python scripts into interactive web apps quickly without requiring extensive web development knowledge. Streamlit seamlessly integrates with popular Python libraries such as Pandas, Matplotlib, Plotly, and TensorFlow, allowing one to leverage existing data processing and visualization tools within the application. Streamlit apps can be easily deployed on various platforms, including Streamlit Community Cloud, Heroku, or any cloud service that supports Python web applications.

[Streamlit Community Cloud](https://streamlit.io/cloud), formerly known as Streamlit Sharing, is a free cloud-based platform provided by Streamlit that allows users to easily deploy and share Streamlit apps online. It is particularly popular among data scientists, machine learning engineers, and developers for quickly showcasing projects, creating interactive demos, and sharing data-driven applications with a wider audience without needing to manage server infrastructure. Significant features include free hosting (Streamlit Community Cloud provides free hosting for Streamlit apps, making it accessible for users who want to share their work without incurring hosting costs), easy deployment (users can connect their GitHub repository to Streamlit Community Cloud, and the app is automatically deployed from the repository), continuous deployment (if the code in the connected GitHub repository is updated, the app is automatically redeployed with the latest changes), 
sharing capabilities (once deployed, apps can be shared with others via a simple URL, making it easy for collaborators, stakeholders, or the general public to access and interact with the app), built-in authentication (users can restrict access to their apps using GitHub-based authentication, allowing control over who can view and interact with the app), and community support (the platform is supported by a community of users and developers who share knowledge, templates, and best practices for building and deploying Streamlit apps).


### 1.7.1 Model Prediction Application Code Development <a class="anchor" id="1.7.1"></a>

1. A model prediction application code in Python was developed to:
    * compute risk indices for the test case and the study population data as baseline
    * estimate lung cancer probabilities for the test case and the study population data as baseline
    * predict risk categories for the test case
2. The model prediction application code was saved in a repository that was eventually cloned for uploading to Streamlit Community Cloud.

![ModelDeployment1_ModelPredictionApplicationCode.png](793869ab-c222-42c3-a287-fc2997fd6172.png)

### 1.7.2 User Interface Application Code Development <a class="anchor" id="1.7.2"></a>

1. A user interface application code in Python was developed to:
    * enable binary category selection (Present|Absent) to identify the status of the test case for each of the ten clinical symptoms and behavioral indicators
    * process study population data as baseline
    * process user input as test case
    * render all entries into visualization charts
    * execute all computations, estimations and predictions
    * render test case prediction into logistic probability plot
2. The user interface application code was saved in a repository that was eventually cloned for uploading to Streamlit Community Cloud.

![ModelDeployment1_UserInterfaceApplicationCode.png](04867edc-15c3-4a27-a755-bb150a384388.png)

### 1.7.3 Web Application <a class="anchor" id="1.7.3"></a>

1. The prediction model was deployed using a web application hosted at [<mark style="background-color: #CCECFF"><b>Streamlit</b></mark>](https://lung-cancer-diagnosis-probability-estimation.streamlit.app).
2. The user interface input consists of the following:
    * radio buttons to:
        * enable binary category selection (Present | Absent) to identify the status of the test case for each of the ten clinical symptoms and behavioral indicators
    * action button to:
        * process study population data as baseline
        * process user input as test case
        * render all entries into visualization charts
        * execute all computations, estimations and predictions
        * render test case prediction into logistic probability plot
3. The user interface ouput consists of the following:
    * count plots to:
        * provide a visualization of the proportion of lung cancer categories (Yes | No) by status (Present | Absent) as baseline
        * indicate the entries made from the user input to visually assess the test case characteristics against the study population 
    * logistic curve plot to:
        * provide a visualization of the baseline logistic regression probability curve using the study population with lung cancer categories (Yes | No)
        * indicate the estimated risk index and lung cancer probability of the test case into the baseline logistic regression probability curvee
    * summary table to:
        * present the computed risk index, estimated lung cancer probability and predicted risk category for the test case 


![ModelDeployment1_WebApplication.png](2f861ce9-e253-481e-b991-c2d9426bfb46.png)

# 2. Summary <a class="anchor" id="Summary"></a>

![ModelDeployment1_Summary_0.png](39f37d50-664f-4792-be20-915cec32bef2.png)

![ModelDeployment1_Summary_1.png](0d6f3518-3953-4b5c-845b-a50bd1ca1016.png)

![ModelDeployment1_Summary_2.png](c5c5dab7-ec6c-42f4-91b4-50b562453e6f.png)

![ModelDeployment1_Summary_3.png](79e581f8-5e78-487a-b207-ad9991d27255.png)

# 3. References <a class="anchor" id="References"></a>

* **[Book]** [Data Preparation for Machine Learning: Data Cleaning, Feature Selection, and Data Transforms in Python](https://machinelearningmastery.com/data-preparation-for-machine-learning/) by Jason Brownlee
* **[Book]** [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/) by Max Kuhn and Kjell Johnson
* **[Book]** [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) by Alice Zheng and Amanda Casari
* **[Book]** [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) by Max Kuhn and Kjell Johnson
* **[Book]** [Data Mining: Practical Machine Learning Tools and Techniques](https://www.sciencedirect.com/book/9780123748560/data-mining-practical-machine-learning-tools-and-techniques?via=ihub=) by Ian Witten, Eibe Frank, Mark Hall and Christopher Pal 
* **[Book]** [Data Cleaning](https://dl.acm.org/doi/book/10.1145/3310205) by Ihab Ilyas and Xu Chu
* **[Book]** [Data Wrangling with Python](https://www.oreilly.com/library/view/data-wrangling-with/9781491948804/) by Jacqueline Kazil and Katharine Jarmul
* **[Book]** [Regression Modeling Strategies](https://link.springer.com/book/10.1007/978-1-4757-3462-1) by Frank Harrell
* **[Book]** [Ensemble Methods for Machine Learning](https://www.manning.com/books/ensemble-methods-for-machine-learning) by Gautam Kunapuli
* **[Book]** [Imbalanced Classification with Python: Better Metrics, Balance Skewed Classes, Cost-Sensitive Learning](https://machinelearningmastery.com/imbalanced-classification-with-python/) by Jason Brownlee
* **[Python Library API]** [NumPy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [itertools](https://docs.python.org/3/library/itertools.html) by Python Team
* **[Python Library API]** [operator](https://docs.python.org/3/library/operator.html) by Python Team
* **[Python Library API]** [sklearn.experimental](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental) by Scikit-Learn Team
* **[Python Library API]** [sklearn.impute](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) by Scikit-Learn Team
* **[Python Library API]** [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) by Scikit-Learn Team
* **[Python Library API]** [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) by Scikit-Learn Team
* **[Python Library API]** [scipy](https://docs.scipy.org/doc/scipy/) by SciPy Team
* **[Python Library API]** [sklearn.tree](https://scikit-learn.org/stable/modules/tree.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.ensemble](https://scikit-learn.org/stable/modules/ensemble.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.svm](https://scikit-learn.org/stable/modules/svm.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.model_selection](https://scikit-learn.org/stable/model_selection.html) by Scikit-Learn Team
* **[Python Library API]** [imblearn.over_sampling](https://imbalanced-learn.org/stable/over_sampling.html) by Imbalanced-Learn Team
* **[Python Library API]** [imblearn.under_sampling](https://imbalanced-learn.org/stable/under_sampling.html) by Imbalanced-Learn Team
* **[Python Library API]** [Streamlit](https://pypi.org/project/streamlit/) by Streamlit Team
* **[Python Library API]** [Streamlit Community Cloud](https://streamlit.io/cloud) by Streamlit Team
* **[Article]** [Step-by-Step Exploratory Data Analysis (EDA) using Python](https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/) by Malamahadevan Mahadevan (Analytics Vidhya)
* **[Article]** [Exploratory Data Analysis in Python — A Step-by-Step Process](https://towardsdatascience.com/exploratory-data-analysis-in-python-a-step-by-step-process-d0dfa6bf94ee) by Andrea D'Agostino (Towards Data Science)
* **[Article]** [Exploratory Data Analysis with Python](https://medium.com/@douglas.rochedo/exploratory-data-analysis-with-python-78b6c1d479cc) by Douglas Rocha (Medium)
* **[Article]** [4 Ways to Automate Exploratory Data Analysis (EDA) in Python](https://builtin.com/data-science/EDA-python) by Abdishakur Hassan (BuiltIn)
* **[Article]** [10 Things To Do When Conducting Your Exploratory Data Analysis (EDA)](https://www.analyticsvidhya.com) by Alifia Harmadi (Medium)
* **[Article]** [How to Handle Missing Data with Python](https://machinelearningmastery.com/handle-missing-data-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Statistical Imputation for Missing Values in Machine Learning](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Imputing Missing Data with Simple and Advanced Techniques](https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87) by Idil Ismiguzel (Towards Data Science)
* **[Article]** [Missing Data Imputation Approaches | How to handle missing values in Python](https://www.machinelearningplus.com/machine-learning/missing-data-imputation-how-to-handle-missing-values-in-python/) by Selva Prabhakaran (Machine Learning +)
* **[Article]** [Master The Skills Of Missing Data Imputation Techniques In Python(2022) And Be Successful](https://medium.com/analytics-vidhya/a-quick-guide-on-missing-data-imputation-techniques-in-python-2020-5410f3df1c1e) by Mrinal Walia (Analytics Vidhya)
* **[Article]** [How to Preprocess Data in Python](https://builtin.com/machine-learning/how-to-preprocess-data-python) by Afroz Chakure (BuiltIn)
* **[Article]** [Easy Guide To Data Preprocessing In Python](https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html) by Ahmad Anis (KDNuggets)
* **[Article]** [Data Preprocessing in Python](https://towardsdatascience.com/data-preprocessing-in-python-b52b652e37d5) by Tarun Gupta (Towards Data Science)
* **[Article]** [Data Preprocessing using Python](https://medium.com/@suneet.bhopal/data-preprocessing-using-python-1bfee9268fb3) by Suneet Jain (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.com/@abonia/data-preprocessing-in-python-1f90d95d44f4) by Abonia Sojasingarayar (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.datadriveninvestor.com/data-preprocessing-3cd01eefd438) by Afroz Chakure (Medium)
* **[Article]** [Detecting and Treating Outliers | Treating the Odd One Out!](https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/) by Harika Bonthu (Analytics Vidhya)
* **[Article]** [Outlier Treatment with Python](https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02) by Sangita Yemulwar (Analytics Vidhya)
* **[Article]** [A Guide to Outlier Detection in Python](https://builtin.com/data-science/outlier-detection-python) by Sadrach Pierre (BuiltIn)
* **[Article]** [How To Find Outliers in Data Using Python (and How To Handle Them)](https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/) by Eric Kleppen (Career Foundry)
* **[Article]** [Statistics in Python — Collinearity and Multicollinearity](https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f) by Wei-Meng Lee (Towards Data Science)
* **[Article]** [Understanding Multicollinearity and How to Detect it in Python](https://towardsdatascience.com/everything-you-need-to-know-about-multicollinearity-2f21f082d6dc) by Terence Shin (Towards Data Science)
* **[Article]** [A Python Library to Remove Collinearity](https://www.yourdatateacher.com/2021/06/28/a-python-library-to-remove-collinearity/) by Gianluca Malato (Your Data Teacher)
* **[Article]** [How to Normalize Data Using scikit-learn in Python](https://www.digitalocean.com/community/tutorials/normalize-data-in-python) by Jayant Verma (Digital Ocean)
* **[Article]** [What are Categorical Data Encoding Methods | Binary Encoding](https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/) by Shipra Saxena  (Analytics Vidhya)
* **[Article]** [Guide to Encoding Categorical Values in Python](https://pbpython.com/categorical-encoding.html) by Chris Moffitt (Practical Business Python)
* **[Article]** [Categorical Data Encoding Techniques in Python: A Complete Guide](https://soumenatta.medium.com/categorical-data-encoding-techniques-in-python-a-complete-guide-a913aae19a22) by Soumen Atta (Medium)
* **[Article]** [Categorical Feature Encoding Techniques](https://towardsdatascience.com/categorical-encoding-techniques-93ebd18e1f24) by Tara Boyle (Medium)
* **[Article]** [Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Hypothesis Testing with Python: Step by Step Hands-On Tutorial with Practical Examples](https://towardsdatascience.com/hypothesis-testing-with-python-step-by-step-hands-on-tutorial-with-practical-examples-e805975ea96e) by Ece Işık Polat (Towards Data Science)
* **[Article]** [17 Statistical Hypothesis Tests in Python (Cheat Sheet)](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [A Step-by-Step Guide to Hypothesis Testing in Python using Scipy](https://medium.com/@gabriel_renno/a-step-by-step-guide-to-hypothesis-testing-in-python-using-scipy-8eb5b696ab07) by Gabriel Rennó (Medium)
* **[Article]** [How to Evaluate Classification Models in Python: A Beginner's Guide](https://builtin.com/data-science/evaluating-classification-models) by Sadrach Pierre (BuiltIn)
* **[Article]** [Machine Learning Classifiers Comparison with Python](https://towardsdatascience.com/machine-learning-classifiers-comparison-with-python-33149aecdbca) by Roberto Salazar (Towards Data Science)
* **[Article]** [Top 6 Machine Learning Algorithms for Classification](https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501) by Destin Gong (Towards Data Science)
* **[Article]** [Metrics For Evaluating Machine Learning Classification Models](https://towardsdatascience.com/metrics-for-evaluating-machine-learning-classification-models-python-example-59b905e079a5) by Cory Maklin (Towards Data Science)
* **[Article]** [Evaluation Metrics for Classification Problems with Implementation in Python](https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-problems-with-implementation-in-python-a20193b4f2c3) by Venu Gopal Kadamba (Medium)
* **[Article]** [Tour of Evaluation Metrics for Imbalanced Classification](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Metrics To Evaluate Machine Learning Algorithms in Python](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [How To Compare Machine Learning Algorithms in Python with scikit-learn](https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [How to Deal With Imbalanced Classification and Regression Data](https://neptune.ai/blog/how-to-deal-with-imbalanced-classification-and-regression-data) by Prince Canuma (Neptune.AI)
* **[Article]** [Random Oversampling and Undersampling for Imbalanced Classification](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [How to Handle Imbalance Data and Small Training Sets in ML](https://towardsdatascience.com/how-to-handle-imbalance-data-and-small-training-sets-in-ml-989f8053531d) by Ege Hosgungor (Towards Data Science)
* **[Article]** [Class Imbalance Strategies — A Visual Guide with Code](https://towardsdatascience.com/class-imbalance-strategies-a-visual-guide-with-code-8bc8fae71e1a) by Travis Tang (Towards Data Science)
* **[Article]** [Machine Learning: How to Handle Class Imbalance](https://medium.com/analytics-vidhya/machine-learning-how-to-handle-class-imbalance-920e48c3e970) by Ken Hoffman (Medium)
* **[Article]** [Handling Class Imbalance in Machine Learning](https://medium.com/mlearning-ai/handling-class-imbalance-in-machine-learning-cb1473e825ce) by Okan Yenigün (Medium)
* **[Article]** [Undersampling Algorithms for Imbalanced Classification](https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Condensed Nearest Neighbor Rule Undersampling (CNN) & TomekLinks](https://bobrupakroy.medium.com/condensed-nearest-neighbor-rule-undersampling-cnn-380c0d84ca88) by Rupak Roy (Medium)
* **[Article]** [CNN (Condensed Nearest Neighbors)](https://abhic159.medium.com/cnn-condensed-nearest-neighbors-3261bd0c39fb) by Abhishek (Medium)
* **[Article]** [Synthetic Minority Over-sampling TEchnique (SMOTE)](https://medium.com/@corymaklin/synthetic-minority-over-sampling-technique-smote-7d419696b88c) by Cory Maklin (Medium)
* **[Article]** [SMOTE for Imbalanced Classification with Python](https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/) by Swastik Satpathy (Analytics Vidhya)
* **[Article]** [An Introduction to SMOTE](https://www.kdnuggets.com/2022/11/introduction-smote.html) by Abid Ali Awan (KD Nuggets)
* **[Article]** [7 SMOTE Variations for Oversampling](https://www.kdnuggets.com/2023/01/7-smote-variations-oversampling.html) by Cornellius Yudha Wijaya (KD Nuggets)
* **[Article]** [A Comprehensive Guide to Ensemble Learning (with Python codes)](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/) by Aishwarya Singh (Analytics Vidhya)
* **[Article]** [Stacked Ensembles — Improving Model Performance on a Higher Level](https://towardsdatascience.com/stacked-ensembles-improving-model-performance-on-a-higher-level-99ffc4ea5523) by Yenwee Lim (Towards Data Science)
* **[Article]** [Stacking to Improve Model Performance: A Comprehensive Guide on Ensemble Learning in Python](https://medium.com/@brijesh_soni/stacking-to-improve-model-performance-a-comprehensive-guide-on-ensemble-learning-in-python-9ed53c93ce28) by Brijesh Soni (Medium)
* **[Article]** [Stacking Ensemble Machine Learning With Python](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Machine Learning Model Deployment with FastAPI, Streamlit and Docker](https://medium.com/latinxinai/fastapi-and-streamlit-app-with-docker-compose-e4d18d78d61d) by Felipe Fernandez (Medium)
* **[Article]** [End-To-End Machine Learning using FastAPI, Streamlit, Docker, Google Cloud Platform](https://medium.com/@marcozaninitaly/end-to-end-machine-learning-using-fastapi-streamlit-docker-google-cloud-platform-fcdf9f9216e0) by Marco Zanin (Medium)
* **[Article]** [FastAPI and Streamlit: The Python Duo You Must Know About](https://towardsdatascience.com/fastapi-and-streamlit-the-python-duo-you-must-know-about-72825def1243) by Paul Lusztin (Medium)
* **[Article]** [How to Build an Instant Machine Learning Web Application with Streamlit and FastAPI](https://developer.nvidia.com/blog/how-to-build-an-instant-machine-learning-web-application-with-streamlit-and-fastapi/) by Kurtis Pykes (Developer.Nvidia.Com)
* **[Article]** [ML - Deploy Machine Learning Models Using FastAPI](https://dorian599.medium.com/ml-deploy-machine-learning-models-using-fastapi-6ab6aef7e777) by Dorian Machado (Medium)
* **[Article]** [FastAPI: The Modern Toolkit for Machine Learning Deployment](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/) by Reza Shokrzad (Medium)
* **[Article]** [Deploying and Hosting a Machine Learning Model with FastAPI and Heroku](https://testdriven.io/blog/fastapi-machine-learning/) by Michael Herman (TestDriven.IO)
* **[Article]** [Using FastAPI to deploy Machine Learning models](https://engineering.rappi.com/using-fastapi-to-deploy-machine-learning-models-cd5ed7219ea) by Carl Handlin (Medium)
* **[Video Tutorial]** [Machine Learning Model with FastAPI, Streamlit and Docker](https://www.youtube.com/watch?v=cCsnmxXxWaM) by codetricks (YouTube)
* **[Video Tutorial]** [Machine learning model serving with streamlit and FastAPI - PyConES 2020](https://www.youtube.com/watch?v=IvHCxycjeR0) by Python Espana (YouTube)
* **[Video Tutorial]** [Deploying a Public Machine Learning Web App using Streamlit in Python | ML Deployment](https://www.youtube.com/watch?v=qzo5B8PuNoY) by Siddhardhan (YouTube)
* **[Video Tutorial]** [Deploy Machine Learning Model using Streamlit in Python | ML model Deployment](https://www.youtube.com/watch?v=WLwjvWq0GWA) by Siddhardhan (YouTube)
* **[Video Tutorial]** [How to Deploy Machine Learning Model as an API in Python - FastAPI](https://www.youtube.com/watch?v=ZTz26f6XXrQ&list=PLfFghEzKVmjscIdESnS8ebs50mvXTnyVH&index=3) by Siddhardhan (YouTube)
* **[Video Tutorial]** [Deploying Machine Learning model as API on Heroku | FastAPI | Heroku | Python | ML](https://www.youtube.com/watch?v=VEjgGI3vU-k&list=PLfFghEzKVmjscIdESnS8ebs50mvXTnyVH&index=5) by Siddhardhan (YouTube)
* **[Video Tutorial]** [Deploying a Machine Learning web app using Streamlit on Heroku](https://www.youtube.com/watch?v=10k_tC3Nzp0&list=PLfFghEzKVmjscIdESnS8ebs50mvXTnyVH&index=6) by Siddhardhan (YouTube)
* **[Video Tutorial]** [Deploy a Machine Learning Streamlit App Using Docker Containers | 2024 Tutorial | Step-by-Step Guide](https://www.youtube.com/watch?v=5pPTNzUcIxg&list=PLfFghEzKVmjscIdESnS8ebs50mvXTnyVH&index=8) by Siddhardhan (YouTube)
* **[Video Tutorial]** [Deploying a Machine Learning model as Dockerized API | ML model Deployment | MLOPS](https://www.youtube.com/watch?v=HlMPjUFjeQQ&list=PLfFghEzKVmjscIdESnS8ebs50mvXTnyVH&index=9) by Siddhardhan (YouTube)
* **[Video Tutorial]** [Machine Learning Model Deployment with Python (Streamlit + MLflow) | Part 1/2](https://www.youtube.com/watch?v=zhgRFBWa6bk&list=PLV8yxwGOxvvrGSOqaPdxIpQKdPXJnZb17) by DeepFindr (YouTube)
* **[Video Tutorial]** [Machine Learning Model Deployment with Python (Streamlit + MLflow) | Part 2/2](https://www.youtube.com/watch?v=RVMIibDbzaE&list=PLV8yxwGOxvvrGSOqaPdxIpQKdPXJnZb17&index=3) by DeepFindr (YouTube)
* **[Publication]** [Data Quality for Machine Learning Tasks](https://journals.sagepub.com/doi/10.1177/0962280206074463) by Nitin Gupta, Shashank Mujumdar, Hima Patel, Satoshi Masuda, Naveen Panwar, Sambaran Bandyopadhyay, Sameep Mehta, Shanmukha Guttula, Shazia Afzal, Ruhi Sharma Mittal and Vitobha Munigala (KDD ’21: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining)
* **[Publication]** [Overview and Importance of Data Quality for Machine Learning Tasks](https://dl.acm.org/doi/10.1145/3394486.3406477) by Abhinav Jain, Hima Patel, Lokesh Nagalapatti, Nitin Gupta, Sameep Mehta, Shanmukha Guttula, Shashank Mujumdar, Shazia Afzal, Ruhi Sharma Mittal and Vitobha Munigala (KDD ’20: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining)
* **[Publication]** [Mathematical Contributions to the Theory of Evolution: Regression, Heredity and Panmixia](https://royalsocietypublishing.org/doi/10.1098/rsta.1896.0007) by Karl Pearson (Royal Society)
* **[Publication]** [The Probable Error of the Mean](http://seismo.berkeley.edu/~kirchner/eps_120/Odds_n_ends/Students_original_paper.pdf) by Student (Biometrika)
* **[Publication]** [On the Criterion That a Given System of Deviations from the Probable in the Case of a Correlated System of Variables is Such That It can Be Reasonably Supposed to Have Arisen From Random Sampling](https://www.tandfonline.com/doi/abs/10.1080/14786440009463897) by Karl Pearson (Philosophical Magazine)
* **[Publication]** [The Origins of Logistic Regression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=360300) by JS Cramer (Econometrics eJournal)
* **[Publication]** [Classification and Regression Trees](https://www.semanticscholar.org/paper/Classification-and-Regression-Trees-Breiman-Friedman/8017699564136f93af21575810d557dba1ee6fc6) by Leo Breiman, Jerome Friedman, Richard Olshen and Charles Stone (Computer Science)
* **[Publication]** [Random Forest](https://link.springer.com/article/10.1023/A:1010933404324) by Leo Breiman (Machine Learning)
* **[Publication]** [A Training Algorithm for Optimal Margin Classifiers](https://dl.acm.org/doi/10.1145/130385.130401) by Bernhard Boser, Isabelle Guyon and Vladimir Vapnik (Proceedings of the Fifth Annual Workshop on Computational Learning Theory)
* **[Publication]** [SMOTE: Synthetic Minority Over-Sampling Technique](https://dl.acm.org/doi/10.5555/1622407.1622416) by Nitesh Chawla, Kevin Bowyer, Lawrence Hall and Philip Kegelmeyer (Journal of Artificial Intelligence Research)
* **[Publication]** [The Condensed Nearest Neighbor Rule](https://ieeexplore.ieee.org/document/1054155) by Peter Hart (IEEE Transactions on Information Theory)
* **[Course]** [DataCamp Python Data Analyst Certificate](https://app.datacamp.com/learn/career-tracks/data-analyst-with-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Python Associate Data Scientist Certificate](https://app.datacamp.com/learn/career-tracks/associate-data-scientist-in-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Python Data Scientist Certificate](https://app.datacamp.com/learn/career-tracks/data-scientist-in-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Machine Learning Engineer Certificate](https://app.datacamp.com/learn/career-tracks/machine-learning-engineer) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Machine Learning Scientist Certificate](https://app.datacamp.com/learn/career-tracks/machine-learning-scientist-with-python) by DataCamp Team (DataCamp)
* **[Course]** [IBM Data Analyst Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-analyst) by IBM Team (Coursera)
* **[Course]** [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science) by IBM Team (Coursera)
* **[Course]** [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) by IBM Team (Coursera)


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

