\# DS605 Lab 8: Decision Trees and Support Vector Machines



\## 📌 Project Overview



This project implements Decision Tree and Linear Support Vector Machine (SVM) classifiers from scratch using NumPy. The models are compared with Scikit-learn implementations. Nonlinear SVM kernels are also explored using a synthetic dataset.



The project uses the \*\*UCI Banknote Authentication Dataset\*\* to classify banknotes into two classes based on their statistical features.



\## 🎯 Objectives



\- Implement Gini impurity and entropy for Decision Trees.

\- Build a Decision Tree classifier from scratch using NumPy.

\- Implement a Linear Soft-Margin SVM using hinge loss and gradient descent.

\- Visualize decision boundaries, tree structure, SVM margins, and margin-violating samples.

\- Compare manual implementations with Scikit-learn models.

\- Explore linear, polynomial, and RBF SVM kernels.

\- Evaluate models using accuracy, precision, recall, F1-score, and execution time.



\## 📂 Dataset



\*\*Dataset:\*\* UCI Banknote Authentication



\*\*Source:\*\* https://archive.ics.uci.edu/dataset/267/banknote+authentication



The dataset contains 1,372 samples, four input features, and a binary target.



| Feature | Description |

|---|---|

| Variance | Variance of the Wavelet Transformed image |

| Skewness | Skewness of the Wavelet Transformed image |

| Curtosis | Curtosis of the Wavelet Transformed image |

| Entropy | Entropy of the image |

| Class | Banknote class label (0 or 1) |



\## 🛠️ Technologies Used



\- Python

\- NumPy

\- Pandas

\- Matplotlib

\- Scikit-learn

\- Jupyter Notebook / Google Colab



\## 📁 Project Structure



```text

DS605\_Lab8/

│

├── DS605\_Lab8\_Decision\_Trees\_SVM\_Complete.ipynb

├── README.md

└── data\_banknote\_authentication.txt   (optional)

```



The notebook attempts to download the dataset automatically. If downloading fails, place the dataset file beside the notebook.



\## 🚀 How to Run



1\. Clone or download this repository.

2\. Open the notebook in Google Colab or Jupyter Notebook.

3\. Install the required libraries if they are not already available:



&#x20;  ```bash

&#x20;  pip install numpy pandas matplotlib scikit-learn

&#x20;  ```



4\. Run all notebook cells from top to bottom.

5\. Review the generated visualizations, performance tables, and observations.



\## 🧠 Implementation Details



\### Part A: Decision Tree from Scratch



\- Implements Gini impurity and entropy.

\- Selects feature thresholds using impurity reduction.

\- Builds the tree recursively with stopping conditions.

\- Generates predictions using learned split rules.

\- Visualizes the manual tree and its decision boundaries.

\- Explores different impurity criteria and maximum depths.



\### Part B: Linear SVM from Scratch



\- Standardizes features using training-set statistics.

\- Maps binary labels to -1 and +1.

\- Implements a soft-margin linear SVM using hinge loss and subgradient descent.

\- Visualizes the separating boundary, margins, and margin-violating samples.

\- Studies the effect of regularization, learning rate, and iterations.



\### Part C: Kernel SVM



Uses a nonlinear synthetic dataset generated with `make\_moons`.



The following kernels are compared:



\- Linear

\- Polynomial

\- Radial Basis Function (RBF)



Their decision boundaries and classification performance are visualized and compared.



\### Part D: Final Model Comparison



The final evaluation uses all four features of the original Banknote dataset and a consistent train-test split.



Models evaluated:



1\. Manual Decision Tree

2\. Scikit-learn Decision Tree

3\. Manual Linear SVM

4\. Scikit-learn Linear SVM

5\. Scikit-learn RBF SVM



Evaluation metrics include:



\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Training time

\- Prediction time



\## 📊 Results and Observations



The notebook generates comparison tables and plots when executed.



The main observations to investigate are:



\- Decision Trees provide interpretable feature-threshold rules.

\- Deep trees may overfit the training data.

\- SVM models generally require feature scaling.

\- The SVM regularization parameter `C` affects the trade-off between margin size and classification errors.

\- Nonlinear kernels can model curved decision boundaries.

\- Execution times may vary depending on the computer or runtime environment.



Actual performance values should be taken from the notebook output after running all cells.



\## ✅ Conclusion



This project demonstrates how Decision Trees and Linear SVMs can be implemented from scratch and compared with Scikit-learn. It also explores nonlinear SVM kernels and evaluates different classifiers using consistent metrics.



The project highlights the importance of feature scaling, model complexity, interpretability, and hyperparameter selection in machine learning.



\## 👤 Author



\*\*Name:\*\* Divyesh Padaliya  

\*\*Course:\*\* MSc Data Science  

\*\*Subject:\*\* DS605 — Fundamentals of Machine Learning  

\*\*Assignment:\*\* Lab 8 — Decision Trees and Support Vector Machines



