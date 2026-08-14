\# Lab 2 – NumPy, Pandas and Data Analysis



\## Overview



This project is a Python-based data analysis laboratory that demonstrates the use of \*\*NumPy, Pandas, Matplotlib, and Seaborn\*\*. The notebook is divided into two major parts.



\*\*Part A\*\* focuses on NumPy arrays, mathematical operations, array manipulation, and basic statistical analysis.



\*\*Part B\*\* focuses on analyzing the \*\*Titanic dataset\*\* using Pandas. It includes data exploration, filtering, grouping, handling missing values, outlier detection, feature engineering, correlation analysis, and visualization.



\---



\## Objectives



The main objectives of this lab are:



\* Understand and create NumPy arrays.

\* Perform statistical calculations using NumPy.

\* Understand array dimensions, indexing, slicing, reshaping, and flattening.

\* Perform matrix operations.

\* Generate and analyze normally distributed data.

\* Load and explore datasets using Pandas.

\* Filter and select data using conditions.

\* Perform grouping and aggregation operations.

\* Identify and handle missing values.

\* Detect outliers using the IQR method.

\* Create new features from existing columns.

\* Calculate correlations between numerical variables.

\* Visualize data using Matplotlib and Seaborn.



\---



\## Technologies Used



\* \*\*Python\*\*

\* \*\*NumPy\*\* – Numerical and array operations

\* \*\*Pandas\*\* – Data manipulation and analysis

\* \*\*Matplotlib\*\* – Data visualization

\* \*\*Seaborn\*\* – Statistical visualization

\* \*\*Jupyter Notebook / Google Colab\*\*



\---



\# Part A – NumPy



\## Task 1: NumPy Arrays



A random array of 100 integers between 1 and 100 is generated using NumPy.



The following statistical values are calculated:



\* Minimum

\* Maximum

\* Mean

\* Median

\* Standard Deviation



Different types of arrays are also created, including:



\* Zero arrays

\* One arrays

\* Linearly spaced arrays

\* 2-dimensional arrays

\* 3-dimensional arrays



Array properties such as `shape`, `ndim`, and `dtype` are examined.



The notebook also demonstrates:



\* Array indexing

\* Row and column selection

\* Slicing

\* Reshaping

\* Flattening



\---



\## Task 2: Matrix and Arithmetic Operations



Two matrices are created and different matrix operations are performed.



Operations include:



\* Matrix addition

\* Element-wise multiplication

\* Matrix multiplication

\* Transpose

\* Determinant

\* Inverse

\* Verification of a matrix multiplied by its inverse



Basic arithmetic operations are also demonstrated using NumPy arrays:



\* Addition

\* Subtraction

\* Multiplication

\* Division



\---



\## Task 3: Normal Distribution



A dataset containing 1,000 randomly generated values is created using a normal distribution with:



\* Mean = 36

\* Standard deviation = 20



The sample mean and sample standard deviation are calculated.



A histogram is also created to visualize the distribution of the generated data.



\---



\# Part B – Pandas and Titanic Dataset



\## Task 4: Dataset Exploration



The Titanic dataset is loaded using Pandas.



```python

train = pd.read\_csv('train.csv')

```



The dataset is explored using:



\* `head()`

\* `tail()`

\* `shape`

\* `columns`

\* `info()`

\* `describe()`

\* `loc`

\* `iloc`



The difference between `loc` and `iloc` is also demonstrated.



\---



\## Task 5: Data Filtering



Different conditions are applied to the Titanic dataset to answer specific questions.



Examples include:



\* Number of male passengers above 50 years old.

\* Number of female passengers traveling in first class.

\* Survival rate of female first-class passengers.

\* Passengers between 20 and 40 years old with fares above the median who survived.

\* Passengers below 30 years old who traveled alone and did not survive.

\* Passengers from Southampton traveling in second or third class with fares above the Southampton median fare.



These operations demonstrate Pandas conditional filtering.



\---



\## Task 6: Grouping and Aggregation



The dataset is grouped using different categorical variables.



The analysis includes:



\* Survival rate by gender.

\* Survival rate by passenger class.

\* Average age and fare by passenger class.

\* Passenger count and survival rate by gender and class.

\* Passenger count, average fare, and survival rate by embarkation location.



The `groupby()` and `agg()` functions are used for this analysis.



\---



\## Task 7: Missing Values and Data Cleaning



Missing values are identified using:



```python

train.isnull().sum()

```



The percentage of missing values is also calculated.



Missing values are handled as follows:



\* Missing `Age` values are replaced with the mean age.

\* Missing `Cabin` values are replaced with `"Unknown"`.

\* Missing `Embarked` values are replaced with the mode.



The dataset is then checked again to verify the remaining missing values.



\### Outlier Detection



Outliers in the `Fare` column are detected using the \*\*Interquartile Range (IQR)\*\* method.



The following are calculated:



\* Q1

\* Q3

\* IQR

\* Lower Bound

\* Upper Bound



Values outside the calculated boundaries are considered outliers.



\---



\## Feature Engineering



Two new features are created:



\### FamilySize



```python

FamilySize = SibSp + Parch + 1

```



This represents the total family size of a passenger.



\### IsAlone



Passengers with a family size of 1 are considered to be traveling alone.



```python

IsAlone = (FamilySize == 1).astype(int)

```



\---



\## Pivot Table Analysis



A pivot table is created to calculate survival rates based on:



\* Gender

\* Passenger class



The analysis identifies the combination with the highest and lowest survival rates.



\---



\# Task 9 – Correlation and Visualization



A correlation matrix is created for selected numerical variables:



\* Survived

\* Pclass

\* Age

\* SibSp

\* Parch

\* Fare



A heatmap is generated using Seaborn to visualize the correlations.



The notebook also identifies:



\* Strongest positive correlation

\* Strongest negative correlation



Additional visualizations include:



\### Survival Rate by Sex



A bar chart compares the survival rates of male and female passengers.



The analysis shows:



\* Female survival rate: \*\*74.2%\*\*

\* Male survival rate: \*\*18.9%\*\*



\### Age vs Fare



A scatter plot is created to examine the relationship between passenger age, fare, and survival.



\---



\# Key Observations



The analysis of the Titanic dataset produced several observations:



1\. Female passengers had a significantly higher survival rate than male passengers.



2\. The survival rate of female passengers was approximately \*\*74.2%\*\*, while the survival rate of male passengers was approximately \*\*18.9%\*\*.



3\. Passengers traveling in \*\*1st class\*\* had a higher survival rate compared with passengers in lower classes.



4\. `Pclass` and `Fare` showed the strongest negative correlation in the analyzed numerical variables, approximately \*\*-0.55\*\*.



5\. `SibSp` and `Parch` showed the strongest positive correlation, approximately \*\*0.41\*\*.



6\. Most passengers traveled with relatively lower fares.



\---



\# Project Structure



```text

Lab\_2/

│

├── 202518028\_Lab\_2.ipynb

├── train.csv

└── README.md

```



\---



\# How to Run



\### 1. Install the required libraries



```bash

pip install numpy pandas matplotlib seaborn

```



\### 2. Open the notebook



The notebook can be opened using:



\* Jupyter Notebook

\* JupyterLab

\* Google Colab

\* VS Code with Jupyter support



\### 3. Add the Dataset



Make sure `train.csv` is available in the same working directory as the notebook.



\### 4. Run the Cells



Execute the notebook cells sequentially to reproduce the analysis and visualizations.



\---



\# Conclusion



This laboratory demonstrates the fundamental concepts required for Python-based data analysis. NumPy is used for numerical and matrix operations, while Pandas is used to explore, filter, clean, group, and transform the Titanic dataset. Matplotlib and Seaborn are used to visualize important patterns in the data.



The analysis shows clear differences in Titanic survival rates based on gender and passenger class. The project also demonstrates important data preprocessing techniques such as handling missing values, detecting outliers, creating new features, and performing correlation analysis.



Overall, this lab provides practical experience with the basic tools and techniques used in \*\*Data Analysis and Machine Learning preprocessing\*\*.



