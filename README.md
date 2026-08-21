# DS605 Lab 3 – Machine Learning Preprocessing and Model Comparison

# 

# Name: Divyesh Padaliya

# ID: 202518028

# 

# 

# DATASET

# 

# Dataset: Hotel Booking Demand

# Dataset Link: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

# 

# 

# PREPROCESSING CHOICES

# 

# The dataset was split into training and testing sets using an 80:20 ratio, with stratification and random\_state=42.

# 

# Numerical Features:

# \- Missing values were handled using KNNImputer(n\_neighbors=5).

# \- Pipeline A: KNN Imputer + StandardScaler.

# \- Pipeline B: KNN Imputer + MinMaxScaler.

# 

# Categorical Features:

# \- Missing values were replaced using SimpleImputer(strategy="most\_frequent").

# \- Categorical variables were encoded using OneHotEncoder(handle\_unknown="ignore").

# 

# All preprocessing was performed using Pipeline and ColumnTransformer to avoid data leakage.

# 

# 

# MODELS USED

# 

# 1\. Logistic Regression

# 2\. Decision Tree Classifier

# 

# Each model was tested with both preprocessing pipelines, resulting in four experiments.

# 

# 

# RESULTS

# 

# Model                    Pipeline       Test Accuracy   Precision   Recall   F1-Score

# Logistic Regression      Pipeline A        0.8010        0.7738     0.6163    0.6862

# Logistic Regression      Pipeline B        0.8012        0.7789     0.6099    0.6841

# Decision Tree             Pipeline A        0.8513        0.7830     0.8006    0.7917

# Decision Tree             Pipeline B        0.8510        0.7822     0.8008    0.7914

# 

# 

# FINAL OBSERVATIONS

# 

# 1\. Decision Tree performed better than Logistic Regression, achieving around 85.1% testing accuracy compared with around 80.1%.

# 

# 2\. Decision Tree with Pipeline A gave the best overall result, with the highest F1-score of 0.7917.

# 

# 3\. StandardScaler and MinMaxScaler had very little effect on the performance of both models.

# 

# 4\. Logistic Regression showed very little overfitting, while the Decision Tree had a larger train-test performance difference, indicating possible overfitting.

# 

# 5\. The confusion matrices also support the better performance of the Decision Tree, as it correctly classified more instances overall and achieved higher recall and F1-score.

# 

# 

# CONCLUSION

# 

# Overall, Decision Tree with Pipeline A (KNN Imputer + StandardScaler) was the best-performing preprocessing-model combination for this dataset. Although it achieved better test performance, its larger train-test performance gap suggests that the model may be somewhat overfitted.

