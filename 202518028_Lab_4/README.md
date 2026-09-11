\# 202518028\_Lab\_4 – Airbnb Price Prediction



\## Student Details



\* \*\*Name:\*\* Divyesh Padaliya

\* \*\*Student ID:\*\* 202518028

\* \*\*Course:\*\* MSc Data Science

\* \*\*Course Code:\*\* DS605

\* \*\*Lab:\*\* Lab 4

\* \*\*Project:\*\* Airbnb Price Prediction



\---



\## 1. Project Overview



This project develops a machine learning system to predict the nightly price of an Airbnb listing using the \*\*Airbnb NYC 2019\*\* dataset.



The project covers a complete machine learning workflow, including:



\* Data analysis

\* Data cleaning

\* Exploratory Data Analysis (EDA)

\* Feature engineering

\* Feature preprocessing

\* Model training

\* Model comparison

\* Hyperparameter tuning

\* Model evaluation

\* Model saving

\* Streamlit web application



The final system allows a user to enter Airbnb listing information and receive an estimated nightly price.



\---



\## 2. Dataset



The project uses the \*\*New York City Airbnb Open Data (2019)\*\* dataset.



The dataset contains information about Airbnb listings, including:



\* Neighbourhood

\* Neighbourhood group

\* Latitude

\* Longitude

\* Room type

\* Minimum nights

\* Number of reviews

\* Reviews per month

\* Host listing count

\* Availability

\* Price



\### Dataset Source



Kaggle:



https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data



\---



\## 3. Data Preprocessing



The following preprocessing steps were performed:



1\. Duplicate records were removed.

2\. Unnecessary identification columns were removed:



&#x20;  \* `id`

&#x20;  \* `name`

&#x20;  \* `host\_id`

&#x20;  \* `host\_name`

3\. The `last\_review` column was removed.

4\. Missing values in `reviews\_per\_month` were replaced using median imputation.

5\. Listings with a price of zero were removed.

6\. Extreme price values above the 99th percentile were removed.

7\. Extreme `minimum\_nights` values above the 99th percentile were removed.



These steps were performed to improve data quality and reduce the influence of extreme observations.



\---



\## 4. Exploratory Data Analysis



Exploratory analysis was performed to understand the factors associated with Airbnb prices.



The following relationships were investigated:



\* Price distribution

\* Price and room type

\* Price and neighbourhood group

\* Price and latitude

\* Price and longitude

\* Price and number of reviews

\* Price and availability

\* Correlation between numerical variables



The analysis showed that \*\*room type, location, neighbourhood, availability, and host/listing characteristics\*\* can have an important relationship with Airbnb prices.



\---



\## 5. Feature Engineering



Two additional features were created.



\### Reviews per Availability



This feature represents the number of reviews relative to listing availability.



```text

reviews\_per\_availability =

number\_of\_reviews / (availability\_365 + 1)

```



\### Host Listing Density



This feature represents the host's listing count relative to availability.



```text

host\_listing\_density =

calculated\_host\_listings\_count / (availability\_365 + 1)

```



These engineered features were included in the final model.



\---



\## 6. Features Used



The final model uses the following features:



```text

neighbourhood\_group

neighbourhood

latitude

longitude

room\_type

minimum\_nights

number\_of\_reviews

reviews\_per\_month

calculated\_host\_listings\_count

availability\_365

reviews\_per\_availability

host\_listing\_density

```



\### Target Variable



```text

price

```



The target variable represents the nightly Airbnb price.



\---



\## 7. Train-Test Split



The dataset was divided into training and testing sets using an 80:20 split.



```text

Training data: 80%

Testing data: 20%

Random state: 42

```



The same split was used for model comparison to ensure a fair evaluation.



\---



\## 8. Data Preprocessing Pipeline



\### Numerical Features



Numerical features were processed using:



\* Median imputation

\* StandardScaler



\### Categorical Features



Categorical features were processed using:



\* Most-frequent-value imputation

\* OneHotEncoder

\* `handle\_unknown="ignore"`



A `ColumnTransformer` was used to apply the appropriate preprocessing to numerical and categorical features.



\---



\## 9. Machine Learning Models



Four regression algorithms were compared:



1\. \*\*Linear Regression\*\*

2\. \*\*Ridge Regression\*\*

3\. \*\*Random Forest Regression\*\*

4\. \*\*Gradient Boosting Regression\*\*



The models were evaluated using:



\* \*\*R² Score\*\*

\* \*\*Mean Absolute Error (MAE)\*\*

\* \*\*Root Mean Squared Error (RMSE)\*\*



\### Evaluation Metrics



\*\*R² Score:\*\* Measures how well the model explains the variation in Airbnb prices.



\*\*MAE:\*\* Measures the average absolute difference between actual and predicted prices.



\*\*RMSE:\*\* Measures prediction error while giving more importance to larger errors.



\---



\## 10. Hyperparameter Tuning



Random Forest Regression was further optimized using `RandomizedSearchCV`.



The following hyperparameters were considered:



```text

n\_estimators

max\_depth

min\_samples\_split

min\_samples\_leaf

max\_features

```



Cross-validation was used to find a suitable combination of hyperparameters.



\---



\## 11. Final Model



The final model is a \*\*tuned Random Forest Regressor\*\* combined with the preprocessing pipeline.



The trained model was saved as:



```text

airbnb\_price\_model.pkl

```



The saved model contains the preprocessing and machine learning workflow required to make predictions on new Airbnb listing information.



\---



\## 12. Final Model Performance



The final model was evaluated on the test dataset.



| Metric  	 |         Test Result |

| --------	 | ------------------: |

| Train R² Score | \*\*0.719553\*\*        |

| Test R² Score  | \*\*0.513836\*\*        |

| MAE     	 | \*\*43.714373\*\*       |

| RMSE     	 | \*\*71.947890\*\*       |



The difference between training and testing performance was also examined to identify possible overfitting.



\---



\## 13. Streamlit Application



A Streamlit web application was developed to make the model easy to use.



The application accepts the following inputs:



\* Neighbourhood Group

\* Neighbourhood

\* Latitude

\* Longitude

\* Room Type

\* Minimum Nights

\* Number of Reviews

\* Reviews Per Month

\* Host Listings Count

\* Availability



The application automatically calculates:



```text

reviews\_per\_availability

host\_listing\_density

```



The trained model then predicts the estimated nightly Airbnb price.



\### Running the Application



Install the required libraries:



```bash

pip install -r requirements.txt

```



Run the Streamlit application:



```bash

python -m streamlit run app.py

```



The application will open in the browser and allow the user to enter listing information and obtain a predicted price.



\---



\## 14. Project Structure



```text

202518028\_Lab\_4/

│

├── app.py

├── retrain\_airbnb\_model.py

├── airbnb\_price\_model.pkl

├── requirements.txt

└── README.md

```



The dataset can be downloaded from the Kaggle link provided above.



\---



\## 15. Application Result



The developed application successfully connects the trained machine learning model with a simple user interface.



A user can provide realistic Airbnb listing details, click \*\*Predict Price\*\*, and receive an estimated nightly price.



Example:



```text

Estimated Nightly Price: $XXX.XX

```



The prediction is generated using the saved trained model.



\---



\## 16. Limitations



The developed system has several limitations:



1\. \*\*Historical data:\*\* The model uses Airbnb NYC 2019 data and may not represent current Airbnb prices.



2\. \*\*Location limitation:\*\* The model was trained using New York City listings and may not perform well for other cities.



3\. \*\*Limited features:\*\* Important factors such as amenities, property quality, ratings, photographs, seasonal demand, and special events are not included.



4\. \*\*Outlier removal:\*\* Extreme price values were removed during preprocessing, so the model may not accurately predict very expensive luxury listings.



5\. \*\*Prediction uncertainty:\*\* The predicted value is an estimate and should not be considered the exact market price.



6\. \*\*Data quality:\*\* Prediction accuracy depends on the quality and correctness of the information entered by the user.



\---



\## 17. Conclusion



This project demonstrates a complete machine learning workflow for Airbnb price prediction.



The project started with data cleaning and exploratory analysis, followed by feature engineering and preprocessing. Multiple regression algorithms were trained and compared using appropriate evaluation metrics. Random Forest Regression was further optimized using hyperparameter tuning.



The final trained model was saved and integrated into a Streamlit application. The application provides a simple interface where users can enter Airbnb listing information and obtain an estimated nightly price.



Overall, the project demonstrates how machine learning can be applied to real-world Airbnb data to build a practical price prediction system.



\---



\## 18. Technologies Used



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Matplotlib

\* Seaborn

\* Joblib

\* Streamlit



\---



\## 19. Author



\*\*Divyesh Padaliya\*\*

\*\*Student ID:\*\* 202518028

\*\*MSc Data Science\*\*



