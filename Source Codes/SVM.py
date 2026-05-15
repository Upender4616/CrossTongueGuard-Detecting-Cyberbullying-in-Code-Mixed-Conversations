import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# -------------------------------
# 1. Load balanced dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\NAKKA TEJASVI\OneDrive\Documents\Reserachpaper\youtube_text_balanced.csv")

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
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------------
# 4. Train SVM (Linear Kernel)
# -------------------------------
model = SVC(kernel='linear', probability=True)
model.fit(X_train_tfidf, y_train)

# -------------------------------
# 5. Predictions
# -------------------------------
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]

# -------------------------------
# 6. Evaluation Metrics
# -------------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)
# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Cyberbullying", "Cyberbullying"]
)

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix - SVM Classifier")
plt.show()

# -------------------------------
# 8. ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend()
plt.show()

# -------------------------------
# 9. TEXT PREDICTION MODE
# -------------------------------
print("\n===== TEXT PREDICTION MODE =====")

while True:
    text = input("\nEnter text (or type 'exit'): ")
    if text.lower() == 'exit':
        break

    text_tfidf = vectorizer.transform([text])
    pred = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf)[0][pred]

    if pred == 1:
        print(f"🚨 Cyberbullying Detected (confidence = {confidence:.2f})")
    else:
        print(f"✅ Not Cyberbullying (confidence = {confidence:.2f})")
