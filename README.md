🧠 Job Market Analysis & Job Category Classification
📌 Project Overview

This project analyzes job postings and builds machine learning models to classify them into categories such as Software / IT, Business / Management, Healthcare / Medical, Sales / Marketing, and more.

The goal is to explore patterns in the job market and evaluate machine learning models that can automatically categorize job postings based on features like job title, description, and location.

This project demonstrates skills in:

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering (categorical encoding, text processing)

Machine Learning Modeling (Logistic Regression, Random Forest, XGBoost)

Model Evaluation (classification metrics, cross-validation, feature importance)

📂 Dataset

Source: Monster.com Job Sample Dataset

https://www.kaggle.com/datasets/PromptCloudHQ/us-jobs-on-monstercom

Contains job postings with features such as:

Job Title

Location (city/state)

Company

Job Description

Target variable: Job Category (e.g., Software/IT, Healthcare, Business).

⚙️ Methodology
🔹 1. Data Cleaning

Extracted and standardized job state codes from messy location strings.

Removed missing values and duplicates.

🔹 2. Exploratory Data Analysis (EDA)

Top job categories and their frequency distribution.

Geographic analysis of job postings (top states).

Word-level analysis of job descriptions (planned extension).

🔹 3. Feature Engineering

Encoded categorical variables.

Vectorized text features (where applicable).

Label encoded job categories for modeling.

🔹 4. Modeling

Logistic Regression → baseline linear model.

Random Forest → ensemble of decision trees.

XGBoost → gradient boosting model.

🔹 5. Evaluation

Train/Test split with stratification.

Accuracy, Precision, Recall, F1-Score.

Cross-validation for robust performance.

Confusion matrices to evaluate misclassifications.


📊 Results
Model	Test Accuracy	Macro f1	Notes
Logistic Regression	63%	59%	Simple & interpretable baseline
Random Forest	67%	60%	Strong baseline for tabular data
XGBoost	69%	59%	Best performing, risk of overfitting checked


🚀 Conclusions & Next Steps

XGBoost provided the strongest performance, suggesting boosting models are highly effective for this classification task.

Logistic Regression offered interpretability but struggled with minority job classes.

Random Forest provided a balance between interpretability and performance.

🔮 Future Improvements

utilizing an external dataset/api to test model performance

create an application to help filter jobs based on categories 

expand the feature engineering section to include more job types and be able to remove the other category. 


🛠️ Tech Stack

Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-learn

XGBoost

Jupyter Notebook / Google Colab

📖 How to Run

Clone this repository:

git clone https://github.com/PTucc327/job_market_analysis.git
cd job_market_analysis


Install dependencies:

pip install -r requirements.txt


Open the Jupyter Notebook:

jupyter notebook job_market_analysis.ipynb

✍️ Author

Paul Tuccinardi – Data Science Student passionate about applying machine learning to real-world problems.

💼 LinkedIn
https://www.linkedin.com/in/paul-tuccinardi/

📧 paultuccinardi@gmail.com
