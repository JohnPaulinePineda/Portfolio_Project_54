##################################
# Loading Python Libraries
##################################
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from model_prediction_community_cloud_app import compute_individual_logit_probability_class, compute_list_logit_probability_class
from pathlib import Path

##################################
# Defining file paths
##################################
DATASETS_FINAL_TRAIN_FEATURES_PATH = Path("datasets/final/train/features")
DATASETS_FINAL_TRAIN_TARGET_PATH = Path("datasets/final/train/target")

##################################
# Loading the dataset
# from the DATASETS_FINAL_TRAIN_PATH
##################################
X_train_smote = pd.read_csv(DATASETS_FINAL_TRAIN_FEATURES_PATH / "X_train_smote.csv")
y_train_smote = pd.read_csv(DATASETS_FINAL_TRAIN_TARGET_PATH / "y_train_smote.csv")

##################################
# Rebuilding the upsampled training data
# for plotting categorical distributions
##################################
lung_cancer_train_smote = pd.concat([X_train_smote, y_train_smote], axis=1)
cat_columns_all = lung_cancer_train_smote.columns
cat_columns_predictors = cat_columns_all[:-1]
lung_cancer_train_smote = lung_cancer_train_smote.astype('object')
lung_cancer_train_smote[cat_columns_predictors] = lung_cancer_train_smote[cat_columns_predictors].replace({0: 'Absent', 1: 'Present'})
lung_cancer_train_smote['LUNG_CANCER'] = lung_cancer_train_smote['LUNG_CANCER'].replace({0: 'No', 1: 'Yes'})


##################################
# Setting the page layout to wide
##################################
st.set_page_config(layout="wide")

##################################
# Listing the variables
##################################
variables = ["YELLOW_FINGERS",
             "ANXIETY", 
             "PEER_PRESSURE", 
             "FATIGUE",
             "ALLERGY", 
             "WHEEZING", 
             "ALCOHOL_CONSUMING", 
             "COUGHING",
             "SWALLOWING_DIFFICULTY", 
             "CHEST_PAIN"]

##################################
# Initializing lists to store user responses
##################################
categorical_responses = {}
numeric_responses = {}

##################################
# Creating a title for the application
##################################
st.markdown("""---""")
st.markdown("<h1 style='text-align: center;'>Lung Cancer Probability Estimator</h1>", unsafe_allow_html=True)

##################################
# Providing a description for the application
##################################
st.markdown("""---""")
st.markdown("<h5 style='font-size: 20px;'>This model evaluates the lung cancer risk of a test case based on certain clinical symptoms and behavioral indicators. Pass the appropriate details below to visually assess your characteristics against the study population, compute your risk index, estimate your lung cancer probability, and determine your risk category. For more information on the complete model development process, you may refer to this <a href='https://johnpaulinepineda.github.io/Portfolio_Project_54/' style='font-weight: bold;'>Jupyter Notebook</a>. Additionally, all associated datasets and code files can be accessed from this <a href='https://github.com/JohnPaulinePineda/Portfolio_Project_54' style='font-weight: bold;'>GitHub Project Repository</a>.</h5>", unsafe_allow_html=True)

##################################
# Creating a section for 
# selecting the options
# for the test case characteristics
##################################
st.markdown("""---""")
st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Clinical Symptoms and Behavioral Indicators</h4>", unsafe_allow_html=True)
st.markdown("""---""")

##################################
# Looping to create radio buttons for each variable
# and storing the user inputs
##################################
for i, var in enumerate(variables):
    col1 = st.columns(1)[0]
    with col1:
        response = st.radio(f"**{var}**:", ["Present", "Absent"], key=var, horizontal=True)
        categorical_responses[var] = response
        numeric_responses[var] = 1 if response == "Present" else 0
       
st.markdown("""---""")
               
##################################
# Converting the user inputs
# to different data types
##################################       
X_test_sample_numeric = pd.DataFrame([numeric_responses])
X_test_sample_category = pd.DataFrame([categorical_responses])

st.markdown("""
    <style>
    .stButton > button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

entered = st.button("Assess Characteristics Against Study Population + Compute Risk Index + Estimate Lung Cancer Probability + Predict Risk Category")

##################################
# Defining the code logic
# for the button action
##################################    
if entered:
    ##################################
    # Defining a section title
    # for the test case characteristics
    ##################################    
    st.markdown("""---""")      
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Case Characteristics</h4>", unsafe_allow_html=True)    
    st.markdown("""---""") 
    
    ##################################
    # Creating a 2x5 grid of plots
    # for comparing the test case characteristics
    # against the training data distribution
    ##################################
    fig, axs = plt.subplots(2, 5, figsize=(17, 8), dpi=1000)
    
    ##################################
    # Defining fixed colors to
    # represent different category levels
    ##################################
    colors = ['red','blue']
    level_order = ['Absent','Present']
    
    ##################################
    # Plotting the user inputs
    # against the countplots
    # of the training data
    # for each variable
    ##################################
    sns.countplot(x='YELLOW_FINGERS', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 0], order=level_order, palette=colors)
    axs[0, 0].axvline(level_order.index(X_test_sample_category['YELLOW_FINGERS'].iloc[0]), color='black', linestyle='--', linewidth=3)
    axs[0, 0].set_title('YELLOW_FINGERS')
    axs[0, 0].set_ylabel('Classification Model Training Case Count')
    axs[0, 0].set_xlabel(None)
    axs[0, 0].set_ylim(0, 200)
    axs[0, 0].legend(title='LUNG_CANCER', loc='upper center')
    for patch, color in zip(axs[0, 0].patches, ['red','red','blue','blue'] ):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    sns.countplot(x='ANXIETY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 1], order=level_order, palette=colors)
    axs[0, 1].axvline(level_order.index(X_test_sample_category['ANXIETY'].iloc[0]), color='black', linestyle='--', linewidth=3)
    axs[0, 1].set_title('ANXIETY')
    axs[0, 1].set_ylabel('Classification Model Training Case Count')
    axs[0, 1].set_xlabel(None)
    axs[0, 1].set_ylim(0, 200)
    axs[0, 1].legend(title='LUNG_CANCER', loc='upper center')
    for patch, color in zip(axs[0, 1].patches, ['red','red','blue','blue'] ):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    sns.countplot(x='PEER_PRESSURE', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 2], order=level_order, palette=colors)
    axs[0, 2].axvline(level_order.index(X_test_sample_category['PEER_PRESSURE'].iloc[0]), color='black', linestyle='--', linewidth=3)
    axs[0, 2].set_title('PEER_PRESSURE')
    axs[0, 2].set_ylabel('Classification Model Training Case Count')
    axs[0, 2].set_xlabel(None)
    axs[0, 2].set_ylim(0, 200)
    axs[0, 2].legend(title='LUNG_CANCER', loc='upper center')
    for patch, color in zip(axs[0, 2].patches, ['red','red','blue','blue'] ):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    sns.countplot(x='FATIGUE', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 3], order=level_order, palette=colors)
    axs[0, 3].axvline(level_order.index(X_test_sample_category['FATIGUE'].iloc[0]), color='black', linestyle='--', linewidth=3)
    axs[0, 3].set_title('FATIGUE')
    axs[0, 3].set_ylabel('Classification Model Training Case Count')
    axs[0, 3].set_xlabel(None)
    axs[0, 3].set_ylim(0, 200)
    axs[0, 3].legend(title='LUNG_CANCER', loc='upper center')
    for patch, color in zip(axs[0, 3].patches, ['red','red','blue','blue'] ):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    sns.countplot(x='ALLERGY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[0, 4], order=level_order, palette=colors)
    axs[0, 4].axvline(level_order.index(X_test_sample_category['ALLERGY'].iloc[0]), color='black', linestyle='--', linewidth=3)
    axs[0, 4].set_title('ALLERGY')
    axs[0, 4].set_ylabel('Classification Model Training Case Count')
    axs[0, 4].set_xlabel(None)
    axs[0, 4].set_ylim(0, 200)
    axs[0, 4].legend(title='LUNG_CANCER', loc='upper center')
    for patch, color in zip(axs[0, 4].patches, ['red','red','blue','blue'] ):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    sns.countplot(x='WHEEZING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 0], order=level_order, palette=colors)
    axs[1, 0].axvline(level_order.index(X_test_sample_category['WHEEZING'].iloc[0]), color='black', linestyle='--', linewidth=3)
    axs[1, 0].set_title('WHEEZING')
    axs[1, 0].set_ylabel('Classification Model Training Case Count')
    axs[1, 0].set_xlabel(None)
    axs[1, 0].set_ylim(0, 200)
    axs[1, 0].legend(title='LUNG_CANCER', loc='upper center')
    for patch, color in zip(axs[1, 0].patches, ['red','red','blue','blue'] ):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    sns.countplot(x='ALCOHOL_CONSUMING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 1], order=level_order, palette=colors)
    axs[1, 1].axvline(level_order.index(X_test_sample_category['ALCOHOL_CONSUMING'].iloc[0]), color='black', linestyle='--', linewidth=3)
    axs[1, 1].set_title('ALCOHOL_CONSUMING')
    axs[1, 1].set_ylabel('Classification Model Training Case Count')
    axs[1, 1].set_xlabel(None)
    axs[1, 1].set_ylim(0, 200)
    axs[1, 1].legend(title='LUNG_CANCER', loc='upper center')
    for patch, color in zip(axs[1, 1].patches, ['red','red','blue','blue'] ):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    sns.countplot(x='COUGHING', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 2], order=level_order, palette=colors)
    axs[1, 2].axvline(level_order.index(X_test_sample_category['COUGHING'].iloc[0]), color='black', linestyle='--', linewidth=3)
    axs[1, 2].set_title('COUGHING')
    axs[1, 2].set_ylabel('Classification Model Training Case Count')
    axs[1, 2].set_xlabel(None)
    axs[1, 2].set_ylim(0, 200)
    axs[1, 2].legend(title='LUNG_CANCER', loc='upper center')
    for patch, color in zip(axs[1, 2].patches, ['red','red','blue','blue'] ):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    sns.countplot(x='SWALLOWING_DIFFICULTY', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 3], order=level_order, palette=colors)
    axs[1, 3].axvline(level_order.index(X_test_sample_category['SWALLOWING_DIFFICULTY'].iloc[0]), color='black', linestyle='--', linewidth=3)
    axs[1, 3].set_title('SWALLOWING_DIFFICULTY')
    axs[1, 3].set_ylabel('Classification Model Training Case Count')
    axs[1, 3].set_xlabel(None)
    axs[1, 3].set_ylim(0, 200)
    axs[1, 3].legend(title='LUNG_CANCER', loc='upper center')
    for patch, color in zip(axs[1, 3].patches, ['red','red','blue','blue'] ):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    sns.countplot(x='CHEST_PAIN', hue='LUNG_CANCER', data=lung_cancer_train_smote, ax=axs[1, 4], order=level_order, palette=colors)
    axs[1, 4].axvline(level_order.index(X_test_sample_category['CHEST_PAIN'].iloc[0]), color='black', linestyle='--', linewidth=3)
    axs[1, 4].set_title('CHEST_PAIN')
    axs[1, 4].set_ylabel('Classification Model Training Case Count')
    axs[1, 4].set_xlabel(None)
    axs[1, 4].set_ylim(0, 200)
    axs[1, 4].legend(title='LUNG_CANCER', loc='upper center')
    for patch, color in zip(axs[1, 4].patches, ['red','red','blue','blue'] ):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    plt.tight_layout()
    
    ##################################
    # Displaying the plot
    ##################################
    st.pyplot(fig)
    
    ##################################
    # Generating the test case prediction
    # to be superimposed on the baseline logistic curve
    ##################################   
    test_case_model_prediction = compute_individual_logit_probability_class(X_test_sample_numeric)
    X_sample_logit = test_case_model_prediction[0]
    X_sample_probability = test_case_model_prediction[1]
    X_sample_class = test_case_model_prediction[2]
    
    ##################################
    # Generating the train case predictions
    # for formulating the baseline logistic curve
    ##################################
    X_list_logit, X_list_probability, X_list_logit_sorted, X_list_probability_sorted = compute_list_logit_probability_class(X_train_smote)
    
    ##################################
    # Defining a section title
    # for the test case lung cancer probability estimation
    ################################## 
    st.markdown("""---""")    
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Case Lung Cancer Probability Estimation</h4>", unsafe_allow_html=True)    
    st.markdown("""---""") 
    
    ##################################
    # Creating a 1x1 plot
    # for plotting the estimated logistic curve
    # of the final classification model
    ##################################
    fig, ax = plt.subplots(figsize=(17, 8), dpi=1000)
    
    ##################################
    # Plotting the computed logit value
    # estimated probability
    # and predicted risk category
    # for the test case 
    # in the estimated logistic curve
    # of the final classification model
    ##################################    
    ax.plot(X_list_logit_sorted, 
            X_list_probability_sorted, 
            label='Classification Model Logistic Curve', 
            color='black')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-6.00, 6.00)
    target_0_indices = y_train_smote == 0
    target_1_indices = y_train_smote == 1
    ax.axhline(0.5, color='green', linestyle='--', label='Classification Model Threshold')
    ax.scatter(X_list_logit[target_0_indices['LUNG_CANCER'].tolist()], 
               X_list_probability[target_0_indices['LUNG_CANCER'].tolist()], 
               color='blue', alpha=0.20, s=100, marker= 'o', edgecolor='k', label='Classification Model Training Cases: LUNG_CANCER = No')
    ax.scatter(X_list_logit[target_1_indices['LUNG_CANCER'].tolist()],
               X_list_probability[target_1_indices['LUNG_CANCER'].tolist()], 
               color='red', alpha=0.20, s=100, marker='o', edgecolor='k', label='Classification Model Training Cases: LUNG_CANCER = Yes')
    if X_sample_class == "Low-Risk":
        ax.scatter(X_sample_logit, X_sample_probability, color='blue', s=125, edgecolor='k', label='Test Case (Low-Risk)', marker= 's', zorder=5)
        ax.axvline(X_sample_logit, color='black', linestyle='--', linewidth=3)
        ax.axhline(X_sample_probability, color='black', linestyle='--', linewidth=3)
    if X_sample_class == "High-Risk":
        ax.scatter(X_sample_logit, X_sample_probability, color='red', s=125, edgecolor='k', label='Test Case (High-Risk)', marker= 's', zorder=5)
        ax.axvline(X_sample_logit, color='black', linestyle='--', linewidth=3)
        ax.axhline(X_sample_probability, color='black', linestyle='--', linewidth=3)
    ax.set_title('Final Classification Model: Stacked Model (Meta-Learner = Logistic Regression, Base Learners = Random Forest, Support Vector Classifier, Decision Tree)')
    ax.set_xlabel('Risk Index (Log-Odds)')
    ax.set_ylabel('Estimated Lung Cancer Probability')
    ax.grid(False)
    ax.legend(facecolor='white', framealpha=1, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)
    plt.tight_layout(rect=[0, 0, 1.00, 0.95])
    
    ##################################
    # Displaying the plot
    ##################################    
    st.pyplot(fig)
    
    ##################################
    # Defining a section title
    # for the test case model prediction summary
    ##################################      
    st.markdown("""---""")   
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Case Model Prediction Summary</h4>", unsafe_allow_html=True)    
    st.markdown("""---""")
    
    ##################################
    # Summarizing the test case model prediction results
    ##################################     
    if X_sample_class == "Low-Risk":
        st.markdown(f"<h4 style='font-size: 20px;'>Computed Risk Index: <span style='color:blue;'>{X_sample_logit:.5f}</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Lung Cancer Probability: <span style='color:blue;'>{X_sample_probability*100:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Predicted Risk Category: <span style='color:blue;'>{X_sample_class}</span></h4>", unsafe_allow_html=True)
    
    if X_sample_class == "High-Risk":
        st.markdown(f"<h4 style='font-size: 20px;'>Computed Risk Index: <span style='color:red;'>{X_sample_logit:.5f}</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Estimated Lung Cancer Probability: <span style='color:red;'>{X_sample_probability*100:.5f}%</span></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-size: 20px;'>Predicted Risk Category: <span style='color:red;'>{X_sample_class}</span></h4>", unsafe_allow_html=True)
    
    st.markdown("""---""")
