import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load original dataset (NO balancing)
# -------------------------------
df = pd.read_csv(r"C:\Users\NAKKA TEJASVI\Downloads\youtube_text_no_nan_no_empty.csv")

# -------------------------------
# 2. Text length feature
# -------------------------------
df['text_length'] = df['clean_text'].astype(str).apply(len)

# -------------------------------
# 3. Class distribution (IMPORTANT)
# -------------------------------
plt.figure()
sns.countplot(x='cyberbullying', data=df)
plt.title("Class Distribution (Original Dataset)")
plt.xlabel("Cyberbullying (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 4. PDF Curve (Text Length vs Cyberbullying)
# -------------------------------
plt.figure()
sns.kdeplot(
    data=df,
    x='text_length',
    hue='cyberbullying',
    fill=True
)
plt.title("PDF Curve of Text Length (Original Dataset)")
plt.xlabel("Text Length")
plt.ylabel("Density")
plt.show()

# -------------------------------
# 5. Box Plot
# -------------------------------
plt.figure()
sns.boxplot(
    x='cyberbullying',
    y='text_length',
    data=df
)
plt.title("Box Plot of Text Length vs Cyberbullying")
plt.xlabel("Cyberbullying (0 = No, 1 = Yes)")
plt.ylabel("Text Length")
plt.show()

# -------------------------------
# 6. Correlation Heatmap (ONLY TWO COLUMNS)
# -------------------------------
corr = df[['intent','harm','aggression','cyberbullying']].corr()
plt.figure()
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)
plt.title("Correlation Heatmap (Text Length vs Cyberbullying)")
plt.show()
