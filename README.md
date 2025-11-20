# 📧 Email Spam Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📖 Overview
This project implements a Machine Learning pipeline designed to classify emails (or SMS messages) into two categories: **Spam** (unwanted) and **Ham** (legitimate). 

The goal was to build a robust classifier that processes raw text data and trains multiple models to find the best balance between **Accuracy** and **Safety** (avoiding false positives). The project covers the entire Data Science lifecycle: from data cleaning and NLP preprocessing to model tuning using GridSearch.

## 📂 Dataset Details
The dataset consists of a collection of labeled messages.
- **Total samples:** ~5,500 messages.
- **Labels:** `ham` (legitimate) vs `spam` (unsolicited).
- **Challenge:** The dataset is highly **imbalanced** (significantly more Ham than Spam), requiring careful evaluation metrics beyond simple accuracy.

## 🛠️ Tech Stack
* **Language:** 🐍 Python
* **Data Processing:** 🐼 Pandas, NumPy
* **Visualization:** 📊 Matplotlib, Seaborn
* **Machine Learning:** 🤖 Scikit-Learn (SVC, Naive Bayes, Random Forest)
* **NLP:** 🗣️ NLTK (Stopwords, Stemming)

## ⚙️ Project Workflow

### 1. Data Preprocessing 🧹
Raw text data is messy. I applied a rigorous cleaning pipeline:
- **Cleaning:** Removal of special characters, numbers, and punctuation.
- **Normalization:** Converting all text to lowercase.
- **Stopwords Removal:** Eliminating common words (e.g., "the", "is", "in") that add noise.
- **Stemming:** Reducing words to their root form (e.g., "running" → "run").
- **Vectorization:** Converting text to numbers using **TF-IDF** (Term Frequency-Inverse Document Frequency).

### 2. Model Training & Tuning 🧠
I trained and compared three different algorithms to see which performs best:
1.  **Multinomial Naive Bayes:** The classic baseline for text classification.
2.  **Support Vector Machine (SVC):** Great for high-dimensional data.
3.  **Random Forest:** A robust ensemble method.

**Hyperparameter Tuning:** used `GridSearchCV` to find the optimal parameters for each model (e.g., tuning `alpha` for Bayes, or `C` and `kernel` for SVM).

## 🏆 Results & Evaluation

I evaluated the models based on **Accuracy** and the **Confusion Matrix** (specifically looking at False Positives).

| Model | Accuracy | Key Observation |
| :--- | :--- | :--- |
| **SVC (Support Vector Machine)** | **~97.6%** 🥇 | **Best Overall Accuracy.** Provides the best balance between precision and recall. |
| **Random Forest** | **~97.4%** 🥈 | **Safest Model.** It achieved **Zero False Positives** in testing. Ideal for business contexts where losing a legitimate email is unacceptable. |
| **Naive Bayes** | **~96.6%** 🥉 | Very fast training time, solid baseline performance. |

> **💡 Conclusion:** If the priority is catching *every* spam message, **SVC** is the winner. However, if the priority is *never* accidentally deleting a real email, **Random Forest** is the best choice.

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/spam-classifier.git](https://github.com/your-username/spam-classifier.git)
    cd spam-classifier
    ```

2.  **Install dependencies**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn nltk openpyxl
    ```

3.  **Run the Notebook**
    Open the `.ipynb` file in **VS Code** or **Jupyter Notebook** and run all cells.
    *Note: Ensure the dataset file (`email_spam.xlsx`) is in the same directory as the notebook.*

## 📊 Visuals
The notebook includes detailed visualizations:
- **Target Distribution:** Bar charts showing the class imbalance.
- **Confusion Matrices:** Heatmaps that visualize exactly where each model made errors (False Positives vs False Negatives).

---
**👨‍💻 Author:** [Your Name]  
**🎓 Context:** Artificial Intelligence Technologies (TIA) Course Assignment
