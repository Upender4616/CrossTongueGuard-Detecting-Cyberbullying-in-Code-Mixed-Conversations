import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv(
    r"C:\Users\NAKKA TEJASVI\OneDrive\Documents\Reserachpaper\youtube_text_balanced.csv"
)

df = df.dropna(subset=['clean_text', 'cyberbullying'])
df = df[df['clean_text'].str.strip() != ""]

X = df['clean_text']
y = df['cyberbullying']

# -------------------------------
# 2. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 3. TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------------
# 4. Logistic Regression
# -------------------------------
model = LogisticRegression(
    class_weight='balanced',
    max_iter=500,
    solver='liblinear'
)

model.fit(X_train_tfidf, y_train)

# -------------------------------
# 5. Evaluation
# -------------------------------
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

# -------------------------------
# 6. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Cyberbullying", "Cyberbullying"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# 7. ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.show()

# -------------------------------
# 8. USER INPUT PREDICTION
# -------------------------------
print("\n===== TEXT PREDICTION MODE =====")

while True:
    text = input("\nEnter text (or type 'exit'): ")

    if text.lower() == "exit":
        print("Exiting prediction mode.")
        break

    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf)[0][prediction]

    if prediction == 1:
        print(f"🚨 Cyberbullying Detected (confidence = {confidence:.2f})")
    else:
        print(f"✅ Not Cyberbullying (confidence = {confidence:.2f})")
