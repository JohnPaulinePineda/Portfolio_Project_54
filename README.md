# [Model Deployment : Estimating Lung Cancer Probabilities From Demographic Factors, Clinical Symptoms And Behavioral Indicators](https://johnpaulinepineda.github.io/Portfolio_Project_54/)

[![](https://img.shields.io/badge/Python-black?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-black?logo=Jupyter)](#) [![](https://img.shields.io/badge/Github-black?logo=Github)](#) [![](https://img.shields.io/badge/Streamlit-black?logo=Streamlit)](#)

This [project](https://johnpaulinepineda.github.io/Portfolio_Project_54/) aims to develop a web application to enable the accessible and efficient use of a classification model for computing the risk index, estimating the lung cancer probability and predicting the risk category of a test case, given various clinical symptoms and behavioral indicators. To enable plotting of the logistic probability curve, the model development process implemented the Logistic Regression model, either as an Independent Learner, or as a Meta-Learner of a Stacking Ensemble with Decision Trees, Random Forest, and Support Vector Machine classifier algorithms as the Base Learners, while evaluating for optimal hyperparameter combinations (using K-Fold Cross Validation), addressing class imbalance (using Upsampling with Synthetic Minority Oversampling Technique (SMOTE) and Downsampling with Condensed Nearest Neighbors (CNN)), and delivering accurate predictions when applied to new unseen data (using model performance evaluation on Independent Validation and Test Sets). Creating the prototype required cloning the repository containing two application codes and uploading to Streamlit Community Cloud - a Model Prediction Code to compute risk indices, estimate lung cancer probabilities, and predict risk categories; and a User Interface Code to process the study population data as baseline, gather the user input as test case, render all entries into the visualization charts, execute all computations, estimations and predictions, and indicate the test case prediction into the logistic curve plot. The final lung cancer prediction model was deployed as a [Streamlit Web Application](https://lung-cancer-diagnosis-probability-estimation.streamlit.app).

<img src="images/ModelDeployment1_Summary_0.png?raw=true"/>

<img src="images/ModelDeployment1_Summary_1.png?raw=true"/>

<img src="images/ModelDeployment1_Summary_2.png?raw=true"/>

<img src="images/ModelDeployment1_Summary_3.png?raw=true"/>
