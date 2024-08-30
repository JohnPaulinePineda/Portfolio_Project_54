##################################
# Loading Python Libraries
##################################
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

##################################
# Defining file paths
##################################
MODELS_PATH = r"models"

##################################
# Loading the final classification model
# from the MODELS_PATH
##################################
final_classification_model = joblib.load(os.path.join(MODELS_PATH, "stacked_balanced_class_best_model_upsampled.pkl"))

##################################
# Formulating a function to
# compute the risk index,
# estimate the lung cancer probability,
# and predict the risk category
# of an individual test case
##################################
def compute_individual_logit_probability_class(X_test_sample):
    X_sample_logit = final_classification_model.decision_function(X_test_sample)[0]
    X_sample_probability = final_classification_model.predict_proba(X_test_sample)[0, 1]
    X_sample_class = "Low-Risk" if X_sample_probability < 0.50 else "High-Risk"
    return X_sample_logit, X_sample_probability, X_sample_class

##################################
# Formulating a function to
# compute the risk index,
# estimate the lung cancer probability,
# and predict the risk category
# of a list of train cases
##################################
def compute_list_logit_probability_class(X_train_list):
    X_list_logit = final_classification_model.decision_function(X_train_list)
    X_list_probability = final_classification_model.predict_proba(X_train_list)[:, 1]
    X_list_logit_index_sorted = np.argsort(X_list_logit)
    X_list_logit_sorted = X_list_logit[X_list_logit_index_sorted]
    X_list_probability_sorted = X_list_probability[X_list_logit_index_sorted]
    return X_list_logit, X_list_probability, X_list_logit_sorted, X_list_probability_sorted
