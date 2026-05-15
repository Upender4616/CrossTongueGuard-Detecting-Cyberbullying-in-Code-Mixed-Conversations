# ================================
# MuRIL Cyberbullying Detection
# ================================

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
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

# -------------------------------
# 1. LOAD DATASET
# -------------------------------
df = pd.read_csv(r"C:\Users\NAKKA TEJASVI\OneDrive\Documents\Reserachpaper\youtube_text_balanced.csv")

# Ensure clean text
df = df.dropna(subset=["clean_text", "cyberbullying"])
df["clean_text"] = df["clean_text"].astype(str)

# -------------------------------
# 2. TRAIN-TEST SPLIT
# -------------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["cyberbullying"]
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# -------------------------------
# 3. TOKENIZER & MODEL
# -------------------------------
MODEL_NAME = "google/muril-base-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("cyberbullying", "labels")
test_dataset = test_dataset.rename_column("cyberbullying", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# -------------------------------
# 4. METRICS FUNCTION
# -------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

# -------------------------------
# 5. TRAINING ARGUMENTS (OPTIMAL)
# -------------------------------
training_args = TrainingArguments(
    output_dir="./muril_results",

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,

    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

# -------------------------------
# 6. TRAINER
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# -------------------------------
# 7. EVALUATION
# -------------------------------
preds = trainer.predict(test_dataset)

y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)
y_prob = torch.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()

print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))

# -------------------------------
# 8. CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (MuRIL)")
plt.show()

# -------------------------------
# 9. ROC CURVE
# -------------------------------
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (MuRIL)")
plt.legend()
plt.show()

# -------------------------------
# 10. TEXT PREDICTION MODE
# -------------------------------
print("\n===== TEXT PREDICTION MODE =====")

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
    confidence, pred = torch.max(probs, dim=1)

    if pred.item() == 1:
        print(f"🚨 Cyberbullying Detected (confidence = {confidence.item():.2f})")
    else:
        print(f"✅ Not Cyberbullying (confidence = {confidence.item():.2f})")
