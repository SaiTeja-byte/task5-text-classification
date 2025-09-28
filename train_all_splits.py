# # Task 5 - Text Classification
# **Candidate Name:** Bathula Sai Teja  
# **Date/Time:** 2025-09-27

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import glob, os
from src.preprocessing import preprocess_text
import joblib

# -----------------------------
# 1️⃣ Paths and category mapping
# -----------------------------
data_folder = "data"
split_files = glob.glob(os.path.join(data_folder, "complaints_part_*.csv"))

category_map = {
    "Credit reporting, credit repair services, or other personal consumer reports": 0,
    "Debt collection": 1,
    "Consumer Loan": 2,
    "Mortgage": 3
}

classes = [0, 1, 2, 3]

# -----------------------------
# 2️⃣ Initialize vectorizer & SGDClassifier
# -----------------------------
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
model_sgd = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)  # no class_weight

# -----------------------------
# 3️⃣ Fit vectorizer on first file
# -----------------------------
first_file = split_files[0]
df_first = pd.read_csv(first_file)
df_first = df_first[df_first["Product"].isin(category_map.keys())]
df_first["label"] = df_first["Product"].map(category_map)
df_first["text"] = (
    df_first["Product"].astype(str) + " " +
    df_first["Sub-product"].astype(str) + " " +
    df_first["Issue"].astype(str) + " " +
    df_first["Consumer complaint narrative"].astype(str)
)
df_first["clean_text"] = df_first["text"].astype(str).apply(preprocess_text)

vectorizer.fit(df_first["clean_text"])
print("Vectorizer fitted on first batch.")

# -----------------------------
# 4️⃣ Incremental training on all split files with manual sample weights
# -----------------------------
sample_size = 5000  # max rows per category per batch

for file in split_files:
    print(f"Processing {file} ...")
    df = pd.read_csv(file)
    df = df[df["Product"].isin(category_map.keys())]
    df["label"] = df["Product"].map(category_map)
    df["text"] = (
        df["Product"].astype(str) + " " +
        df["Sub-product"].astype(str) + " " +
        df["Issue"].astype(str) + " " +
        df["Consumer complaint narrative"].astype(str)
    )
    df["clean_text"] = df["text"].astype(str).apply(preprocess_text)

    # Balanced sampling per category
    balanced_batch = pd.concat([
        df[df['Product'] == 'Consumer Loan'].sample(min(sample_size, len(df[df['Product'] == 'Consumer Loan'])), random_state=42),
        df[df['Product'] == 'Mortgage'].sample(min(sample_size, len(df[df['Product'] == 'Mortgage'])), random_state=42),
        df[df['Product'] == 'Debt collection'].sample(min(sample_size, len(df[df['Product'] == 'Debt collection'])), random_state=42),
        df[df['Product'] == 'Credit reporting, credit repair services, or other personal consumer reports'].sample(min(sample_size, len(df[df['Product'] == 'Credit reporting, credit repair services, or other personal consumer reports'])), random_state=42)
    ])

    X_batch = vectorizer.transform(balanced_batch["clean_text"])
    y_batch = balanced_batch["label"]

    # Compute sample weights manually
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.array(classes),
        y=y_batch
    )
    class_weights = {cls: w for cls, w in zip(classes, class_weights_array)}
    sample_weight = y_batch.map(class_weights).values

    # Partial fit with sample weights
    model_sgd.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sample_weight)

print("All split files trained successfully.")

# -----------------------------
# 5️⃣ Evaluation on first batch
# -----------------------------
X_test = vectorizer.transform(df_first["clean_text"])
y_test = df_first["label"]

y_pred = model_sgd.predict(X_test)
print("\nSGDClassifier Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# 6️⃣ Save final model and vectorizer
# -----------------------------
joblib.dump(model_sgd, "model_sgd_weighted.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer_weighted.pkl")
print("Final model and vectorizer saved.")

# # Task 5 - Text Classification
# **Candidate Name:** Bathula Sai Teja  
# **Date/Time:** 2025-09-27
