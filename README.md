# 🧠 Job Market Analysis & Job Category Classification  
A Data Science Project by **Paul Tuccinardi**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-brightgreen)
![NLTK](https://img.shields.io/badge/NLP-NLTK-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Overview

This project analyzes job postings from Monster.com and builds machine learning models to automatically classify each posting into categories such as:

- **Technology**
- **Business**
- **Healthcare**
- **Logistics**
- **Trades**
- **Marketing**
- **Customer Service**
- **Human Resources**
- **Legal**
- **Education**
- **Sales**
- **Other**

The analysis includes **deep EDA**, **geographical trends**, **salary extraction**, **NLP skill identification**, and **multiple machine learning models**.

---

## 📂 Dataset

**Source:**  
https://www.kaggle.com/datasets/PromptCloudHQ/us-jobs-on-monstercom

**Features include:**
- Job id
- Title  
- Location (messy city/state/zip format)
- Description 
- Organization  
- Salary (in many inconsistent formats)  
- Sector  
- Posted Date  

**Target Variable:**  
🟩 `job_category` (created through keyword extraction and mapping)

---

## ⚙️ Methodology

### 🔹 1. Data Cleaning & Fixing Inconsistent Fields  
- Extracted state abbreviations from messy location strings  
- Created new `city`, `state_clean`, and `region` features  
- Standardized job types (Full Time, Part Time, Contract/Temp, Other)  
- Dropped unusable columns (`country`, `job_board`, etc.)  
- Cleaned and normalized salary strings  
- Handled missing values and duplicates  

---

### 🔹 2. Exploratory Data Analysis (EDA)

Key analyses included:

#### 📍 Location Trends
- **Texas** had the most job postings  
- South & Midwest accounted for the highest hiring demand  

#### 🏢 Organization & Sector Distribution
- Sector and organization fields often mislabeled  
- Healthcare sector dominated volume  

#### 📦 Job Category Distribution  
Custom NLP-based mapping of job titles → job_category.

#### ☁ Word Cloud of Job Titles
Shows common keywords like *Manager, Technician, Engineer, Assistant, Sales*.

---

### 🔹 3. NLP Skill Extraction (New)

Using **NLTK**, the project extracts skills from job descriptions by:

- Tokenizing  
- Cleaning text  
- Stopword removal  
- Lemmatization  
- Matching both **single-word** and **multi-word** skills  

**Detected skills include:**  
`Python, SQL, Excel, AWS, Tableau, Java, Hadoop, Spark, Machine Learning, Agile, Scrum, PowerPoint, JavaScript, HTML, CSS`.

**Findings:**  
- **Excel** and **Microsoft Office** dominate the dataset  
- **SQL** and **Java** lead technical skills  
- Python appears less due to dataset age (2016)

---

### 🔹 4. Salary Parsing & Midpoint Extraction (New)

A specialized parser:

- Removes $, commas, "/year", etc.  
- Extracts numeric ranges  
- Computes midpoint salary  
- Filters valid salary rows  

Result: **3,085 usable salaries**.

Additional analysis:  
- Northeast has higher average salaries  
- Most common salary: **$20k**, then **$25k** and **0k**

---

### 🔹 5. Feature Engineering

- TF-IDF vectorization (5,000 features, unigrams + bigrams)  
- Label encoding of target categories  
- Region, state, job_type_clean, city extracted from text  
- Stratified train/test split  

---

## 🤖 Machine Learning Models

Three models were trained:

| Model | Test Accuracy | Macro F1 | Notes |
|-------|--------------|----------|-------|
| **Logistic Regression** | ~60% | ~0.52 | Strong baseline, interpretable |
| **Random Forest** | ~69% | ~0.61 | Best balance of accuracy & recall |
| **XGBoost** | ~74% | ~0.66-0.67 | Best performing overall |

### Evaluation included:
- Precision, Recall, F1  
- Confusion matrix heatmaps  
- Cross-validation  
- Per-class metrics  

**Key Issue:**  
The “Other” category has high noise → causes most misclassifications.
Insufficient amount of data for other job categories such as healthcare, or trades.
---

## 🚀 Conclusions

- **XGBoost** achieved the strongest performance.  
- **Random Forest** showed consistent accuracy across categories.  
- **Logistic Regression** struggled with minority classes.  
- **Dataset quality significantly limits upper-bound performance.**  
- Many job titles were incorrectly entered resulting in skewed results.

---

## 🔮 Future Improvements

### Based on results from this version:

✔ Improve model performance to get better results.
✔ Use a **modern dataset (2020–2025)** to collect better more accurate data
✔ Build a **web application** to predict job category from pasted text  
✔ Expand dictionary for job title mapping → reduce “Other” category  

---

## 🛠️ Tech Stack

- **Python**  
- **Pandas** & **NumPy**  
- **Matplotlib** & **Seaborn**  
- **NLTK**  
- **scikit-learn**  
- **XGBoost**  
- **TensorFlow / Keras**  
- **Jupyter Notebook** / Google Colab  

---

## 📖 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/PTucc327/job_market_analysis.git
cd job_market_analysis
