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
        * [1.4.1 Data Cleaning](#1.4.1)
        * [1.4.2 Missing Data Imputation](#1.4.2)
        * [1.4.3 Outlier Treatment](#1.4.3)
        * [1.4.4 Collinearity](#1.4.4)
        * [1.4.5 Shape Transformation](#1.4.5)
        * [1.4.6 Centering and Scaling](#1.4.6)
        * [1.4.7 Data Encoding](#1.4.7)
        * [1.4.8 Preprocessed Data Description](#1.4.8)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Predictive Model Development](#1.6)
        * [1.6.1 Data Preprocessing Pipeline](#1.6.1)
        * [1.6.2 Model Testing](#1.6.2)
        * [1.6.3 Model Validation](#1.6.2)
    * [1.7 Consolidated Findings](#1.7)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

## 1.1. Data Background <a class="anchor" id="1.1"></a>

## 1.2. Data Description <a class="anchor" id="1.2"></a>


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
%matplotlib inline
import shap

from operator import add,mul,truediv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier
```


```python
##################################
# Defining file paths
##################################
DATASETS_ORIGINAL_PATH = r"datasets\original"
DATASET_PREPROCESSED_PATH = r"datasets\preprocessed"
DATASET_FINAL_PATH = r"datasets\final"
```


```python
##################################
# Loading the dataset
##################################
lung_cancer = pd.read_csv(os.path.join("..", DATASETS_ORIGINAL_PATH, "LungCancer.csv"))
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




```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

