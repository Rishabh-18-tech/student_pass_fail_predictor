import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)

# ── 1. Load Dataset ──────────────────────────────────────────────
df = pd.read_csv("dataset/students.csv")
print("✅ Dataset loaded:", df.shape)
print(df.head())

# ── 2. Check for nulls ───────────────────────────────────────────
print("\n🔍 Null values:\n", df.isnull().sum())

# ── 3. Features & Target ─────────────────────────────────────────
X = df[["hours_studied", "attendance", "prev_score", "assignments_done"]]
y = df["result"]

print(f"\n📊 Pass: {y.sum()} | Fail: {(y==0).sum()}")

# ── 4. Train-Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. Scale Features ────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 6. Train Model ───────────────────────────────────────────────
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("\n✅ Model trained!")

# ── 7. Evaluate ──────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)
auc    = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

print(f"\n📈 Accuracy  : {acc*100:.2f}%")
print(f"📈 ROC-AUC   : {auc:.4f}")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))
print("🔲 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ── 8. Save Model & Scaler ───────────────────────────────────────
os.makedirs("model", exist_ok=True)
pickle.dump(model,  open("model/logistic_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
print("\n💾 Model and scaler saved to /model/")
