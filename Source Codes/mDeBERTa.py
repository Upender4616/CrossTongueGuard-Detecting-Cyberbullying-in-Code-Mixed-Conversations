# -------------------------------
# 1. IMPORTS
# -------------------------------
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# -------------------------------
# 2. LOAD DATASET
# -------------------------------
df = pd.read_csv(r"C:\Users\NAKKA TEJASVI\OneDrive\Documents\Reserachpaper\youtube_text_balanced.csv")
df = df[['clean_text', 'cyberbullying']]  # Ensure columns exist

# -------------------------------
# 3. TRAIN-TEST SPLIT
# -------------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['clean_text'], df['cyberbullying'], test_size=0.2, random_state=42, stratify=df['cyberbullying']
)

train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
test_df  = pd.DataFrame({'text': test_texts, 'label': test_labels})

train_dataset = Dataset.from_pandas(train_df)
test_dataset  = Dataset.from_pandas(test_df)

# -------------------------------
# 4. TOKENIZATION
# -------------------------------
model_name = "microsoft/mdeberta-v3-base"  # You can also use "xlm-roberta-base" or "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# -------------------------------
# 5. MODEL INITIALIZATION
# -------------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# -------------------------------
# 6. TRAINING ARGUMENTS
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
# 7. METRICS FUNCTION
# -------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# -------------------------------
# 8. TRAINER
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# -------------------------------
# 9. EVALUATION
# -------------------------------
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids

acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds)
rec = recall_score(labels, preds)
f1 = f1_score(labels, preds)

print("\n===== EVALUATION RESULTS =====")
print(f"Accuracy : {acc}")
print(f"Precision: {prec}")
print(f"Recall   : {rec}")
print(f"F1 Score : {f1}")

# -------------------------------
# 10. CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(labels, preds)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Not Cyberbullying","Cyberbullying"])
plt.yticks([0,1], ["Not Cyberbullying","Cyberbullying"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], horizontalalignment='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.show()

# -------------------------------
# 11. ROC CURVE
# -------------------------------
probs = torch.nn.functional.softmax(torch.tensor(preds_output.predictions), dim=1).numpy()[:,1]
fpr, tpr, thresholds = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# -------------------------------
# 12. TEXT PREDICTION MODE
# -------------------------------
print("\n===== TEXT PREDICTION MODE =====")
threshold = 0.45  # Adjust for sensitivity

while True:
    text = input("\nEnter text (or type 'exit'): ")
    if text.lower() == 'exit':
        break

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    cyber_prob = probs[0][1].item()  # probability of cyberbullying

    if cyber_prob >= threshold:
        print(f"🚨 Cyberbullying Detected (confidence = {cyber_prob:.2f})")
    else:
        print(f"✅ Not Cyberbullying (confidence = {cyber_prob:.2f})")
