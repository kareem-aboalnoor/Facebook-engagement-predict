# 📊 Facebook Post Engagement Classifier

Analyze and predict engagement levels of Facebook posts using *clustering and machine learning, with an interactive **Streamlit app*.

---

## 🔍 Project Overview

This project leverages machine learning to classify Facebook posts into *engagement levels* based on their reactions, comments, shares, and post type. Posts are categorized into three levels:

- 💤 *Low Engagement*  
- 👍 *Medium Engagement*  
- 🔥 *High Engagement*  

*Purpose:* Empower marketers and content creators to identify high-performing posts and optimize their social media strategy.

---

## 🛠 Tools & Libraries

- *Python* – Core programming language  
- *Pandas & NumPy* – Data manipulation  
- *Scikit-learn* – LabelEncoder, MinMaxScaler, KMeans, PCA, Random Forest  
- *Matplotlib & Seaborn* – Data visualization  
- *Streamlit* – Interactive web application  

---

## ⚡ Project Steps

1. *Data Cleaning & Exploration*  
   - Cleaned the dataset and handled missing values  
   - Performed Exploratory Data Analysis (EDA) to understand trends  

2. *Preprocessing*  
   - Encoded categorical features using LabelEncoder  
   - Scaled numerical features with MinMaxScaler  

3. *Clustering*  
   - Applied *KMeans* to group posts into engagement clusters  
   - Visualized clusters using *PCA* for dimensionality reduction  

4. *Classification*  
   - Assigned cluster labels to the dataset  
   - Trained a *Random Forest Classifier* to predict engagement levels for new posts  

---

## 🚀 How to Run

### 1. Jupyter Notebook

- Open social_media_clustering.ipynb and run the cells to explore data cleaning, preprocessing, clustering, PCA visualization, and Random Forest classification.

### 2. Streamlit App

- Ensure app.py and rf_with_scaler_encoder.pkl are in the same folder.  
- Open a terminal in that folder and run:  
```bash
streamlit run app.py
