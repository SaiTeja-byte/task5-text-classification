# Text Classification Project

## Author

**Bathula Sai Teja**

---

## 📌 Project Overview

This project implements **text classification** using different machine learning models. The workflow covers:

* Exploratory Data Analysis (EDA)
* Text preprocessing
* Feature extraction using TF-IDF
* Model training (Naive Bayes, SGD Classifier, etc.)
* Model comparison
* Prediction on new text data

---

## 📂 Project Structure

```
TASK5-TEXT-CLASSIFICATION/
│── data/                      # Dataset files (if any)  
│── models/                    # (reserved, not used here)  
│── notebooks/                 # Jupyter notebooks  
│   ├── eda.ipynb  
│   ├── Model_Training_Comparison.ipynb  
│   ├── Prediction.ipynb  
│   └── preprocessing.ipynb  
│── screenshots/               # Screenshots for submission  
│── src/                       # Python scripts  
│   ├── create_sample.py  
│   ├── preprocessing.py  
│   ├── utils.py  
│   └── split_data.py  
│── model_nb.pkl  
│── model_sgd.pkl  
│── model_sgd_weighted.pkl  
│── tfidf_vectorizer.pkl  
│── tfidf_vectorizer_weighted.pkl  
│── predictions.csv  
│── requirements.txt  
│── README.md  
```

---

## ⚙️ Installation & Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/TASK5-TEXT-CLASSIFICATION.git
   cd TASK5-TEXT-CLASSIFICATION
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

### 1. Preprocess the data

```bash
python src/preprocessing.py
```

### 2. Split the dataset

```bash
python src/split_data.py
```

### 3. Train models

```bash
python src/train_all_splits.py
```

### 4. Make predictions

```bash
python notebooks/Prediction.ipynb
```

Predictions will be saved in **`predictions.csv`**.

---

## 📸 Screenshots

### 🔍 EDA

![EDA 1](screenshots/task5_eda.ipynb%201.png)
![EDA 2](screenshots/task5_eda.ipynb%202.png)
![EDA 3](screenshots/task5_eda.ipynb%203.png)
![EDA 4](screenshots/task5_eda.ipynb%204.png)

---

### 🛠️ Preprocessing

![Preprocessing 1](screenshots/preprocessing.ipynb%201.png)
![Preprocessing 2](screenshots/preprocessing.ipynb%202.png)
![Preprocessing 3](screenshots/preprocessing.ipynb%203.png)
![Preprocessing Script](screenshots/preprocessing.py.png)

---

### 📊 Model Training & Comparison

![Model Training 1](screenshots/Model_training_comparision.ipynb%201.png)
![Model Training 2](screenshots/Model_training_comparision.ipynb%202.png)
![Model Training 3](screenshots/Model_training_comparision.ipynb%203.png)
![Model Training 4](screenshots/Model_training_comparision.ipynb%204.png)
![Model Training 5](screenshots/Model_training_comparision.ipynb%205.png)

---

### 🔮 Prediction

![Prediction 1](screenshots/task5_prediction.ipynb.png)
![Prediction 2](screenshots/task5_prediction.ipynb%202.png)

---

### ⚡ Other Scripts

![Create Sample](screenshots/create_sample.py.png)
![Split Data](screenshots/split_data.py.png)
![Train Splits 1](screenshots/train_all_splits.py_1.png)
![Train Splits 2](screenshots/train_all_splits.py_2.png)
![Train Splits 3](screenshots/train_all_splits.py_3.png)





---

## ✅ Submission Notes

* All screenshots include **name (Bathula Sai Teja)** and **system date/time** for authenticity.
* Models (`.pkl` files) and vectorizers are saved at the root level.
* Predictions are saved in `predictions.csv`.

---
