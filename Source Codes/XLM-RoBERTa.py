import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt

# -------------------------------
# 1. LOAD DATASET (BALANCED)
# -------------------------------
df = pd.read_csv(r"C:\Users\NAKKA TEJASVI\OneDrive\Documents\Reserachpaper\youtube_text_balanced.csv")

X = df["clean_text"].astype(str)
y = df["cyberbullying"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 2. TOKENIZER & MODEL
# -------------------------------
MODEL_NAME = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# -------------------------------
# 3. DATASET CLASS
# -------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

# -------------------------------
# 4. TRAINING ARGUMENTS
# -------------------------------
training_args = TrainingArguments(
    output_dir="./xlmr_results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="no",
    report_to="none"
)

# -------------------------------
# 5. TRAINER
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# -------------------------------
# 6. EVALUATION
# -------------------------------
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_prob = torch.softmax(
    torch.tensor(predictions.predictions), dim=1
).numpy()[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== EVALUATION RESULTS =====")
print(f"Accuracy : {accuracy}")
print(f"Precision: {precision}")
print(f"Recall   : {recall}")
print(f"F1 Score : {f1}")

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (MuRIL)")
plt.show()

# -------------------------------
# 7. ROC CURVE
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XLM-RoBERTa")
plt.legend()
plt.show()

# -------------------------------
# 8. TEXT PREDICTION MODE
# -------------------------------
print("\n===== TEXT PREDICTION MODE =====")

model.eval()

while True:
    text = input("\nEnter text (or type 'exit'): ")
    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

    if pred == 1:
        print(f"🚨 Cyberbullying Detected (confidence = {confidence:.2f})")
    else:
        print(f"✅ Not Cyberbullying (confidence = {confidence:.2f})")
