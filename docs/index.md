***
# Model Deployment : Estimating Lung Cancer Probabilities From Demographic Factors And Behavioral Indicators

***
### John Pauline Pineda <br> <br> *August 17, 2024*
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
        * [1.6.4 Model Fitting using Upsampled Training Data | Hyperparameter Tuning | Validation](#1.6.4)
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
        * [1.6.9 Model Inference | Interpretation](#1.6.9)
    * [1.7 Consolidated Findings](#1.7)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

## 1.1. Data Background <a class="anchor" id="1.1"></a>

## 1.2. Data Description <a class="anchor" id="1.2"></a>


```python
##################################
# Setting up compatibility issues
# between the scikit-learn and imblearn packages
##################################
# !pip uninstall scikit-learn --yes
# !pip uninstall imblearn --yes
# !pip install scikit-learn==1.2.2
# !pip install imblearn
```


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
import shap

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
    CHRONIC DISEASE           int64
    FATIGUE                   int64
    ALLERGY                   int64
    WHEEZING                  int64
    ALCOHOL CONSUMING         int64
    COUGHING                  int64
    SHORTNESS OF BREATH       int64
    SWALLOWING DIFFICULTY     int64
    CHEST PAIN                int64
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
      <th>CHRONIC DISEASE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL CONSUMING</th>
      <th>COUGHING</th>
      <th>SHORTNESS OF BREATH</th>
      <th>SWALLOWING DIFFICULTY</th>
      <th>CHEST PAIN</th>
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
lung_cancer.iloc[:,2:15] = lung_cancer.iloc[:,2:15].replace({1: 'Absent', 2: 'Present'})
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
    CHRONIC DISEASE            object
    FATIGUE                    object
    ALLERGY                    object
    WHEEZING                   object
    ALCOHOL CONSUMING          object
    COUGHING                   object
    SHORTNESS OF BREATH        object
    SWALLOWING DIFFICULTY      object
    CHEST PAIN                 object
    LUNG_CANCER              category
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
      <th>CHRONIC DISEASE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL CONSUMING</th>
      <th>COUGHING</th>
      <th>SHORTNESS OF BREATH</th>
      <th>SWALLOWING DIFFICULTY</th>
      <th>CHEST PAIN</th>
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
      <th>CHRONIC DISEASE</th>
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
      <th>ALCOHOL CONSUMING</th>
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
      <th>SHORTNESS OF BREATH</th>
      <td>309</td>
      <td>2</td>
      <td>Present</td>
      <td>198</td>
    </tr>
    <tr>
      <th>SWALLOWING DIFFICULTY</th>
      <td>309</td>
      <td>2</td>
      <td>Absent</td>
      <td>164</td>
    </tr>
    <tr>
      <th>CHEST PAIN</th>
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


```python
##################################
# Counting the number of duplicated rows
##################################
lung_cancer.duplicated().sum()
```




    33




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
      <td>CHRONIC DISEASE</td>
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
      <td>ALCOHOL CONSUMING</td>
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
      <td>SHORTNESS OF BREATH</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SWALLOWING DIFFICULTY</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CHEST PAIN</td>
      <td>object</td>
      <td>309</td>
      <td>309</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LUNG_CANCER</td>
      <td>category</td>
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
      <td>63</td>
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
      <td>CHRONIC DISEASE</td>
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
      <td>ALCOHOL CONSUMING</td>
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
      <td>SHORTNESS OF BREATH</td>
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
      <td>SWALLOWING DIFFICULTY</td>
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
      <td>CHEST PAIN</td>
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


    
![png](output_78_0.png)
    



```python
##################################
# Creating a dataset copy and
# converting all values to numeric
# for correlation analysis
##################################
lung_cancer_correlation = lung_cancer.copy()
lung_cancer_correlation_object = lung_cancer_correlation.iloc[:,2:15].columns
lung_cancer_correlation[lung_cancer_correlation_object] = lung_cancer_correlation[lung_cancer_correlation_object].replace({'Absent': 0, 'Present': 1})
lung_cancer_correlation = lung_cancer_correlation.drop(['GENDER','LUNG_CANCER'], axis=1)
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
      <th>CHRONIC DISEASE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL CONSUMING</th>
      <th>COUGHING</th>
      <th>SHORTNESS OF BREATH</th>
      <th>SWALLOWING DIFFICULTY</th>
      <th>CHEST PAIN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
    </tr>
    <tr>
      <th>1</th>
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
    </tr>
    <tr>
      <th>2</th>
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
    </tr>
    <tr>
      <th>3</th>
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
    </tr>
    <tr>
      <th>4</th>
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
    </tr>
    <tr>
      <th>305</th>
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
    </tr>
    <tr>
      <th>306</th>
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
    </tr>
    <tr>
      <th>307</th>
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
    </tr>
    <tr>
      <th>308</th>
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
            if lung_cancer_correlation.dtypes[i] == 'int64' and lung_cancer_correlation.dtypes[j] == 'int64':
                # Pearson correlation for two continuous variables
                corr = lung_cancer_correlation.iloc[:, i].corr(lung_cancer_correlation.iloc[:, j])
            elif lung_cancer_correlation.dtypes[i] == 'int64' or lung_cancer_correlation.dtypes[j] == 'int64':
                # Point-biserial correlation for one continuous and one binary variable
                continuous_var = lung_cancer_correlation.iloc[:, i] if lung_cancer_correlation.dtypes[i] == 'int64' else lung_cancer_correlation.iloc[:, j]
                binary_var = lung_cancer_correlation.iloc[:, j] if lung_cancer_correlation.dtypes[j] == 'int64' else lung_cancer_correlation.iloc[:, i]
                corr, _ = pointbiserialr(continuous_var, binary_var)
            else:
                # Phi coefficient for two binary variables
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


    
![png](output_82_0.png)
    


## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>


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
plt.boxplot([group[boxplot_x_variable] for name, group in lung_cancer.groupby(boxplot_y_variable)])
plt.title(f'{boxplot_y_variable} Versus {boxplot_x_variable}')
plt.xlabel(boxplot_y_variable)
plt.ylabel(boxplot_x_variable)
plt.xticks(range(1, len(lung_cancer[boxplot_y_variable].unique()) + 1), ['No', 'Yes'])
plt.show()
```


    
![png](output_87_0.png)
    



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
    category_counts = lung_cancer.groupby([proportion_x_variable, y_variable]).size().unstack(fill_value=0)
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


    
![png](output_89_0.png)
    


### 1.5.2 Exploratory Data Analysis <a class="anchor" id="1.5.2"></a>


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
      <th>LUNG_CANCER_ALCOHOL CONSUMING</th>
      <td>24.005406</td>
      <td>9.606559e-07</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_SWALLOWING DIFFICULTY</th>
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
      <th>LUNG_CANCER_CHEST PAIN</th>
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
      <th>LUNG_CANCER_CHRONIC DISEASE</th>
      <td>3.161200</td>
      <td>7.540772e-02</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_GENDER</th>
      <td>1.021545</td>
      <td>3.121527e-01</td>
    </tr>
    <tr>
      <th>LUNG_CANCER_SHORTNESS OF BREATH</th>
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


```python
##################################
# Creating a dataset copy and
# transforming all values to numeric
# prior to data splitting and modelling
##################################
lung_cancer_transformed = lung_cancer.copy()
lung_cancer_transformed_object = lung_cancer_transformed.iloc[:,2:15].columns
lung_cancer_transformed['GENDER'] = lung_cancer_transformed['GENDER'].replace({'F': 0, 'M': 1})
lung_cancer_transformed['LUNG_CANCER'] = lung_cancer_transformed['LUNG_CANCER'].replace({'NO': 0, 'YES': 1})
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
      <th>CHRONIC DISEASE</th>
      <th>FATIGUE</th>
      <th>ALLERGY</th>
      <th>WHEEZING</th>
      <th>ALCOHOL CONSUMING</th>
      <th>COUGHING</th>
      <th>SHORTNESS OF BREATH</th>
      <th>SWALLOWING DIFFICULTY</th>
      <th>CHEST PAIN</th>
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
lung_cancer_filtered = lung_cancer_transformed.drop(['GENDER','CHRONIC DISEASE', 'SHORTNESS OF BREATH', 'SMOKING', 'AGE'], axis=1)
lung_cancer_filtered.to_csv(os.path.join("..", DATASETS_FINAL_PATH, "lung_cancer_final.csv"), index=False)
```

### 1.6.2 Data Splitting <a class="anchor" id="1.6.2"></a>


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
lung_cancer_breakdown = lung_cancer_final.groupby('LUNG_CANCER').size().reset_index(name='Count')
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
display(y_train_initial.value_counts(normalize = True))
```

    Initial Training Dataset Dimensions: 
    


    (231, 10)



    (231,)


    Initial Training Target Variable Breakdown: 
    


    1    0.874459
    0    0.125541
    Name: LUNG_CANCER, dtype: float64



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
display(y_test.value_counts(normalize = True))
```

    Test Dataset Dimensions: 
    


    (78, 10)



    (78,)


    Test Target Variable Breakdown: 
    


    1    0.871795
    0    0.128205
    Name: LUNG_CANCER, dtype: float64



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
print('Original Training Dataset Dimensions: ')
display(X_train.shape)
display(y_train.shape)
print('Original Training Target Variable Breakdown: ')
display(y_train.value_counts())
print('Original Training Target Variable Proportion: ')
display(y_train.value_counts(normalize = True))
```

    Original Training Dataset Dimensions: 
    


    (173, 10)



    (173,)


    Original Training Target Variable Breakdown: 
    


    1    151
    0     22
    Name: LUNG_CANCER, dtype: int64


    Original Training Target Variable Proportion: 
    


    1    0.872832
    0    0.127168
    Name: LUNG_CANCER, dtype: float64



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
display(y_validation.value_counts(normalize = True))
```

    Validation Dataset Dimensions: 
    


    (58, 10)



    (58,)


    Validation Target Variable Breakdown: 
    


    1    0.87931
    0    0.12069
    Name: LUNG_CANCER, dtype: float64



```python
##################################
# Initiating an oversampling instance
# on the training data using
# Synthetic Minority Oversampling Technique
##################################
smote = SMOTE(random_state = 88888888)
X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
print('Upsampled Training Dataset Dimensions: ')
display(X_train_smote.shape)
display(y_train_smote.shape)
print('Upsampled Training Target Variable Breakdown: ')
display(y_train_smote.value_counts())
print('Upsampled Training Target Variable Proportion: ')
display(y_train_smote.value_counts(normalize = True))
```

    Upsampled Training Dataset Dimensions: 
    


    (302, 10)



    (302,)


    Upsampled Training Target Variable Breakdown: 
    


    0    151
    1    151
    Name: LUNG_CANCER, dtype: int64


    Upsampled Training Target Variable Proportion: 
    


    0    0.5
    1    0.5
    Name: LUNG_CANCER, dtype: float64



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
    


    1    39
    0    22
    Name: LUNG_CANCER, dtype: int64


    Downsampled Training Target Variable Proportion: 
    


    1    0.639344
    0    0.360656
    Name: LUNG_CANCER, dtype: float64



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
                                                       C=1.0,
                                                       random_state=88888888))]
```


```python
##################################
# Defining the meta learner
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
                                                'stacked_model__svm__kernel': ['linear', 'poly', 'rbf'],
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
                                                       C=1.0,
                                                       random_state=88888888))]
```


```python
##################################
# Defining the meta learner
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
                                                'stacked_model__svm__kernel': ['linear', 'poly', 'rbf'],
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


```python
##################################
# Fitting the model on the 
# original training data
##################################
individual_unbalanced_class_grid_search.fit(X_train, y_train)
```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits
    




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;individual_model&#x27;,
                 LogisticRegression(max_iter=5000, random_state=88888888,
                                    solver=&#x27;saga&#x27;))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=5000, random_state=88888888, solver=&#x27;saga&#x27;)</pre></div></div></div></div></div></div></div></div></div></div></div></div>




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


    
![png](output_142_0.png)
    



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


    
![png](output_144_0.png)
    



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
         individual_unbalanced_class_best_model_original_probabilities_sorted, label='Logistic Curve', color='black')
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
plt.axhline(0.5, color='green', linestyle='--', label='Classification Threshold (50%)')
plt.title('Logistic Curve (Original Training Data): Individual Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_148_0.png)
    



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


```python
##################################
# Fitting the model on the 
# original training data
##################################
stacked_unbalanced_class_grid_search.fit(X_train, y_train)
```

    Fitting 5 folds for each of 36 candidates, totalling 180 fits
    




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
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
                         &#x27;stacked_model__svm__kernel&#x27;: [&#x27;linear&#x27;, &#x27;poly&#x27;,
                                                        &#x27;rbf&#x27;]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
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
                         &#x27;stacked_model__svm__kernel&#x27;: [&#x27;linear&#x27;, &#x27;poly&#x27;,
                                                        &#x27;rbf&#x27;]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;stacked_model&#x27;,
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
                                                 SVC(class_weight=&#x27;balanced&#x27;,
                                                     random_state=88888888))],
                                    final_estimator=LogisticRegression(max_iter=5000,
                                                                       random_state=88888888,
                                                                       solver=&#x27;saga&#x27;)))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">stacked_model: StackingClassifier</label><div class="sk-toggleable__content"><pre>StackingClassifier(estimators=[(&#x27;dt&#x27;,
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
                                SVC(class_weight=&#x27;balanced&#x27;,
                                    random_state=88888888))],
                   final_estimator=LogisticRegression(max_iter=5000,
                                                      random_state=88888888,
                                                      solver=&#x27;saga&#x27;))</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>dt</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;, criterion=&#x27;entropy&#x27;,
                       min_samples_leaf=3, random_state=88888888)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>rf</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(class_weight=&#x27;balanced&#x27;, criterion=&#x27;entropy&#x27;,
                       min_samples_leaf=3, random_state=88888888)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>svm</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC(class_weight=&#x27;balanced&#x27;, random_state=88888888)</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>final_estimator</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=5000, random_state=88888888, solver=&#x27;saga&#x27;)</pre></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




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
    Best Stacked Model Parameters: {'stacked_model__dt__max_depth': 5, 'stacked_model__final_estimator__class_weight': 'balanced', 'stacked_model__final_estimator__penalty': None, 'stacked_model__rf__max_depth': 5, 'stacked_model__svm__kernel': 'rbf'}
    


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

    F1 Score on Cross-Validated Data: 0.9085
    F1 Score on Training Data: 0.9343
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
               0       0.54      0.86      0.67        22
               1       0.98      0.89      0.93       151
    
        accuracy                           0.89       173
       macro avg       0.76      0.88      0.80       173
    weighted avg       0.92      0.89      0.90       173
    
    


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


    
![png](output_156_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
##################################
print(f"F1 Score on Validation Data: {stacked_unbalanced_class_best_model_original_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_validation, stacked_unbalanced_class_best_model_original.predict(X_validation)))
```

    F1 Score on Validation Data: 0.9703
    
    Classification Report on Validation Data:
                   precision    recall  f1-score   support
    
               0       0.75      0.86      0.80         7
               1       0.98      0.96      0.97        51
    
        accuracy                           0.95        58
       macro avg       0.86      0.91      0.89        58
    weighted avg       0.95      0.95      0.95        58
    
    


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


    
![png](output_158_0.png)
    



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
         stacked_unbalanced_class_best_model_original_probabilities_sorted, label='Logistic Curve', color='black')
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
plt.axhline(0.5, color='green', linestyle='--', label='Classification Threshold (50%)')
plt.title('Logistic Curve (Original Training Data): Stacked Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_162_0.png)
    



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


```python
##################################
# Fitting the model on the 
# upsampled training data
##################################
individual_balanced_class_grid_search.fit(X_train_smote, y_train_smote)
```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits
    




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [None],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [None],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;individual_model&#x27;,
                 LogisticRegression(max_iter=5000, random_state=88888888,
                                    solver=&#x27;saga&#x27;))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=5000, random_state=88888888, solver=&#x27;saga&#x27;)</pre></div></div></div></div></div></div></div></div></div></div></div></div>




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
print('Best Individual Model using the Original Train Data: ')
print(f"Best Individual Model Parameters: {individual_balanced_class_grid_search.best_params_}")
```

    Best Individual Model using the Original Train Data: 
    Best Individual Model Parameters: {'individual_model__class_weight': None, 'individual_model__penalty': 'l2'}
    


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

    F1 Score on Cross-Validated Data: 0.9474
    F1 Score on Training Data: 0.9495
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
               0       0.94      0.97      0.95       151
               1       0.97      0.93      0.95       151
    
        accuracy                           0.95       302
       macro avg       0.95      0.95      0.95       302
    weighted avg       0.95      0.95      0.95       302
    
    


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


    
![png](output_171_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
##################################
print(f"F1 Score on Validation Data: {individual_balanced_class_best_model_upsampled_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_validation, individual_balanced_class_best_model_upsampled.predict(X_validation)))
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


    
![png](output_173_0.png)
    



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
         individual_balanced_class_best_model_upsampled_probabilities_sorted, label='Logistic Curve', color='black')
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
plt.axhline(0.5, color='green', linestyle='--', label='Classification Threshold (50%)')
plt.title('Logistic Curve (Upsampled Training Data): Individual Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_177_0.png)
    



```python
##################################
# Saving the best individual model
# developed from the upsampled training data
################################## 
joblib.dump(individual_balanced_class_best_model_upsampled, 
            os.path.join("..", MODELS_PATH, "individual_balanced_class_best_model_upsampled.pkl"))
```




    ['..\\models\\individual_balanced_class_best_model_upsampled.pkl']



#### 1.6.5.2 Stacked Classifier <a class="anchor" id="1.6.5.2"></a>


```python
##################################
# Fitting the model on the 
# upsampled training data
##################################
stacked_balanced_class_grid_search.fit(X_train_smote, y_train_smote)
```

    Fitting 5 folds for each of 36 candidates, totalling 180 fits
    




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
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
                                                                        SVC(random_state=88888888))],
                                                           final_estimator=LogisticRegression(max_iter=5000,
                                                                                              random_state=88888888,
                                                                                              solver=&#x27;saga&#x27;)))]),
             n_jobs=-1,
             param_grid={&#x27;stacked_model__dt__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__final_estimator__class_weight&#x27;: [None],
                         &#x27;stacked_model__final_estimator__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;,
                                                                     None],
                         &#x27;stacked_model__rf__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__svm__kernel&#x27;: [&#x27;linear&#x27;, &#x27;poly&#x27;,
                                                        &#x27;rbf&#x27;]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
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
                                                                        SVC(random_state=88888888))],
                                                           final_estimator=LogisticRegression(max_iter=5000,
                                                                                              random_state=88888888,
                                                                                              solver=&#x27;saga&#x27;)))]),
             n_jobs=-1,
             param_grid={&#x27;stacked_model__dt__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__final_estimator__class_weight&#x27;: [None],
                         &#x27;stacked_model__final_estimator__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;,
                                                                     None],
                         &#x27;stacked_model__rf__max_depth&#x27;: [3, 5],
                         &#x27;stacked_model__svm__kernel&#x27;: [&#x27;linear&#x27;, &#x27;poly&#x27;,
                                                        &#x27;rbf&#x27;]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;stacked_model&#x27;,
                 StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                                 DecisionTreeClassifier(criterion=&#x27;entropy&#x27;,
                                                                        min_samples_leaf=3,
                                                                        random_state=88888888)),
                                                (&#x27;rf&#x27;,
                                                 RandomForestClassifier(criterion=&#x27;entropy&#x27;,
                                                                        min_samples_leaf=3,
                                                                        random_state=88888888)),
                                                (&#x27;svm&#x27;,
                                                 SVC(random_state=88888888))],
                                    final_estimator=LogisticRegression(max_iter=5000,
                                                                       random_state=88888888,
                                                                       solver=&#x27;saga&#x27;)))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label sk-toggleable__label-arrow">stacked_model: StackingClassifier</label><div class="sk-toggleable__content"><pre>StackingClassifier(estimators=[(&#x27;dt&#x27;,
                                DecisionTreeClassifier(criterion=&#x27;entropy&#x27;,
                                                       min_samples_leaf=3,
                                                       random_state=88888888)),
                               (&#x27;rf&#x27;,
                                RandomForestClassifier(criterion=&#x27;entropy&#x27;,
                                                       min_samples_leaf=3,
                                                       random_state=88888888)),
                               (&#x27;svm&#x27;, SVC(random_state=88888888))],
                   final_estimator=LogisticRegression(max_iter=5000,
                                                      random_state=88888888,
                                                      solver=&#x27;saga&#x27;))</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>dt</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(criterion=&#x27;entropy&#x27;, min_samples_leaf=3,
                       random_state=88888888)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>rf</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(criterion=&#x27;entropy&#x27;, min_samples_leaf=3,
                       random_state=88888888)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>svm</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC(random_state=88888888)</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>final_estimator</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=5000, random_state=88888888, solver=&#x27;saga&#x27;)</pre></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




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
print('Best Stacked Model using the Upsampled Train Data: ')
print(f"Best Stacked Model Parameters: {stacked_balanced_class_grid_search.best_params_}")
```

    Best Stacked Model using the Upsampled Train Data: 
    Best Stacked Model Parameters: {'stacked_model__dt__max_depth': 3, 'stacked_model__final_estimator__class_weight': None, 'stacked_model__final_estimator__penalty': None, 'stacked_model__rf__max_depth': 3, 'stacked_model__svm__kernel': 'rbf'}
    


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

    F1 Score on Cross-Validated Data: 0.9522
    F1 Score on Training Data: 0.9603
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
               0       0.96      0.96      0.96       151
               1       0.96      0.96      0.96       151
    
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


    
![png](output_185_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
##################################
print(f"F1 Score on Validation Data: {stacked_balanced_class_best_model_upsampled_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_validation, stacked_balanced_class_best_model_upsampled.predict(X_validation)))
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


    
![png](output_187_0.png)
    



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
         stacked_balanced_class_best_model_upsampled_probabilities_sorted, label='Logistic Curve', color='black')
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
plt.axhline(0.5, color='green', linestyle='--', label='Classification Threshold (50%)')
plt.title('Logistic Curve (Upsampled Training Data): Stacked Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_191_0.png)
    



```python
##################################
# Saving the best stacked model
# developed from the upsampled training data
################################## 
joblib.dump(stacked_balanced_class_best_model_upsampled, 
            os.path.join("..", MODELS_PATH, "stacked_balanced_class_best_model_upsampled.pkl"))
```




    ['..\\models\\stacked_balanced_class_best_model_upsampled.pkl']



### 1.6.6 Model Fitting using Downsampled Training Data | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.5"></a>

#### 1.6.6.1 Individual Classifier <a class="anchor" id="1.6.6.1"></a>


```python
##################################
# Fitting the model on the 
# downsampled training data
##################################
individual_unbalanced_class_grid_search.fit(X_train_cnn, y_train_cnn)
```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits
    




<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;individual_model&#x27;,
                                        LogisticRegression(max_iter=5000,
                                                           random_state=88888888,
                                                           solver=&#x27;saga&#x27;))]),
             n_jobs=-1,
             param_grid={&#x27;individual_model__class_weight&#x27;: [&#x27;balanced&#x27;],
                         &#x27;individual_model__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, None]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;individual_model&#x27;,
                 LogisticRegression(max_iter=5000, random_state=88888888,
                                    solver=&#x27;saga&#x27;))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-23" type="checkbox" ><label for="sk-estimator-id-23" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=5000, random_state=88888888, solver=&#x27;saga&#x27;)</pre></div></div></div></div></div></div></div></div></div></div></div></div>




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


    
![png](output_200_0.png)
    



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


    
![png](output_202_0.png)
    



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
         individual_unbalanced_class_best_model_downsampled_probabilities_sorted, label='Logistic Curve', color='black')
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
plt.axhline(0.5, color='green', linestyle='--', label='Classification Threshold (50%)')
plt.title('Logistic Curve (Downsampled Training Data): Individual Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_206_0.png)
    



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


```python
##################################
# Fitting the model on the 
# downsampled training data
##################################
stacked_unbalanced_class_grid_search.fit(X_train_cnn, y_train_cnn)
```

    Fitting 5 folds for each of 36 candidates, totalling 180 fits
    




<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
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
                         &#x27;stacked_model__svm__kernel&#x27;: [&#x27;linear&#x27;, &#x27;poly&#x27;,
                                                        &#x27;rbf&#x27;]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-24" type="checkbox" ><label for="sk-estimator-id-24" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
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
                         &#x27;stacked_model__svm__kernel&#x27;: [&#x27;linear&#x27;, &#x27;poly&#x27;,
                                                        &#x27;rbf&#x27;]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-25" type="checkbox" ><label for="sk-estimator-id-25" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;stacked_model&#x27;,
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
                                                 SVC(class_weight=&#x27;balanced&#x27;,
                                                     random_state=88888888))],
                                    final_estimator=LogisticRegression(max_iter=5000,
                                                                       random_state=88888888,
                                                                       solver=&#x27;saga&#x27;)))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-26" type="checkbox" ><label for="sk-estimator-id-26" class="sk-toggleable__label sk-toggleable__label-arrow">stacked_model: StackingClassifier</label><div class="sk-toggleable__content"><pre>StackingClassifier(estimators=[(&#x27;dt&#x27;,
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
                                SVC(class_weight=&#x27;balanced&#x27;,
                                    random_state=88888888))],
                   final_estimator=LogisticRegression(max_iter=5000,
                                                      random_state=88888888,
                                                      solver=&#x27;saga&#x27;))</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>dt</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-27" type="checkbox" ><label for="sk-estimator-id-27" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(class_weight=&#x27;balanced&#x27;, criterion=&#x27;entropy&#x27;,
                       min_samples_leaf=3, random_state=88888888)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>rf</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-28" type="checkbox" ><label for="sk-estimator-id-28" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(class_weight=&#x27;balanced&#x27;, criterion=&#x27;entropy&#x27;,
                       min_samples_leaf=3, random_state=88888888)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>svm</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" ><label for="sk-estimator-id-29" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC(class_weight=&#x27;balanced&#x27;, random_state=88888888)</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>final_estimator</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-30" type="checkbox" ><label for="sk-estimator-id-30" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=5000, random_state=88888888, solver=&#x27;saga&#x27;)</pre></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




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
print('Best Stacked Model using the Downsampled Train Data: ')
print(f"Best Stacked Model Parameters: {stacked_unbalanced_class_grid_search.best_params_}")
```

    Best Stacked Model using the Downsampled Train Data: 
    Best Stacked Model Parameters: {'stacked_model__dt__max_depth': 3, 'stacked_model__final_estimator__class_weight': 'balanced', 'stacked_model__final_estimator__penalty': 'l1', 'stacked_model__rf__max_depth': 3, 'stacked_model__svm__kernel': 'rbf'}
    


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


    
![png](output_214_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
##################################
print(f"F1 Score on Validation Data: {stacked_unbalanced_class_best_model_downsampled_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_validation, stacked_unbalanced_class_best_model_downsampled.predict(X_validation)))
```

    F1 Score on Validation Data: 0.9505
    
    Classification Report on Validation Data:
                   precision    recall  f1-score   support
    
               0       0.62      0.71      0.67         7
               1       0.96      0.94      0.95        51
    
        accuracy                           0.91        58
       macro avg       0.79      0.83      0.81        58
    weighted avg       0.92      0.91      0.92        58
    
    


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


    
![png](output_216_0.png)
    



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
         stacked_unbalanced_class_best_model_downsampled_probabilities_sorted, label='Logistic Curve', color='black')
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
plt.axhline(0.5, color='green', linestyle='--', label='Classification Threshold (50%)')
plt.title('Logistic Curve (Downsampled Training Data): Stacked Model')
plt.xlabel('Logit (Log-Odds)')
plt.ylabel('Estimated Lung Cancer Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
```


    
![png](output_220_0.png)
    



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
      <td>0.934256</td>
      <td>0.949495</td>
      <td>0.960265</td>
      <td>0.853333</td>
      <td>0.891892</td>
    </tr>
    <tr>
      <th>Cross-Validation</th>
      <td>0.911574</td>
      <td>0.908472</td>
      <td>0.947396</td>
      <td>0.952151</td>
      <td>0.753711</td>
      <td>0.778272</td>
    </tr>
    <tr>
      <th>Validation</th>
      <td>0.949495</td>
      <td>0.970297</td>
      <td>0.961538</td>
      <td>0.970874</td>
      <td>0.970874</td>
      <td>0.950495</td>
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


    
![png](output_224_0.png)
    


### 1.6.8 Model Testing <a class="anchor" id="1.6.8"></a>


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
      <td>0.934256</td>
      <td>0.949495</td>
      <td>0.960265</td>
      <td>0.853333</td>
      <td>0.891892</td>
    </tr>
    <tr>
      <th>Cross-Validation</th>
      <td>0.911574</td>
      <td>0.908472</td>
      <td>0.947396</td>
      <td>0.952151</td>
      <td>0.753711</td>
      <td>0.778272</td>
    </tr>
    <tr>
      <th>Validation</th>
      <td>0.949495</td>
      <td>0.970297</td>
      <td>0.961538</td>
      <td>0.970874</td>
      <td>0.970874</td>
      <td>0.950495</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>0.904762</td>
      <td>0.923077</td>
      <td>0.932331</td>
      <td>0.942029</td>
      <td>0.939394</td>
      <td>0.909091</td>
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


    
![png](output_228_0.png)
    


### 1.6.9 Model Inference | Interpretation <a class="anchor" id="1.6.7"></a>

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
* **[Article]** [Step-by-Step Exploratory Data Analysis (EDA) using Python](https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/#:~:text=Exploratory%20Data%20Analysis%20(EDA)%20with,distributions%20using%20Python%20programming%20language.) by Malamahadevan Mahadevan (Analytics Vidhya)
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
* **[Article]** [An Introduction to SMOTE](https://www.kdnuggets.com/2022/11/introduction-smote.html#:~:text=SMOTE%20(Synthetic%20Minority%20Oversampling%20Technique)%20is%20an%20oversampling%20method%20of,a%20point%20along%20that%20line.) by Abid Ali Awan (KD Nuggets)
* **[Article]** [A Comprehensive Guide to Ensemble Learning (with Python codes)](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/) by Aishwarya Singh (Analytics Vidhya)
* **[Article]** [Stacked Ensembles — Improving Model Performance on a Higher Level](https://towardsdatascience.com/stacked-ensembles-improving-model-performance-on-a-higher-level-99ffc4ea5523) by Yenwee Lim (Towards Data Science)
* **[Article]** [Stacking to Improve Model Performance: A Comprehensive Guide on Ensemble Learning in Python](https://medium.com/@brijesh_soni/stacking-to-improve-model-performance-a-comprehensive-guide-on-ensemble-learning-in-python-9ed53c93ce28) by Brijesh Soni (Medium)
* **[Article]** [Stacking Ensemble Machine Learning With Python](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/) by Jason Brownlee (Machine Learning Mastery)
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
* **[Course]** [DataCamp Machine Learning Scientist Certificate](https://app.datacamp.com/learn/career-tracks/machine-learning-scientist-with-python) by DataCamp Team (DataCamp)
* **[Course]** [IBM Data Analyst Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-analyst) by IBM Team (Coursera)
* **[Course]** [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science) by IBM Team (Coursera)
* **[Course]** [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) by IBM Team (Coursera)


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

