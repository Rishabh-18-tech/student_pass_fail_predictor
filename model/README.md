# 🎓 Student Pass/Fail Predictor
### AIML Project — Logistic Regression

---

## 📌 About
This project predicts whether a student will **Pass or Fail** an exam using a
**Logistic Regression** model trained on student performance data.

## 🗂️ Project Structure
```
student_pass_fail/
├── dataset/
│   └── students.csv        ← Training data (60 records)
├── model/
│   ├── logistic_model.pkl  ← Saved trained model
│   └── scaler.pkl          ← Saved StandardScaler
├── train.py                ← Model training script
├── app.py                  ← Streamlit web application
└── README.md
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install streamlit scikit-learn pandas numpy matplotlib
```

### 2. Train the model
```bash
python train.py
```

### 3. Launch the web app
```bash
streamlit run app.py
```

## 📊 Features Used
| Feature | Description |
|---|---|
| `hours_studied` | Daily study hours (0–12) |
| `attendance` | Class attendance % (0–100) |
| `prev_score` | Previous exam score (0–100) |
| `assignments_done` | Assignments completed % (0–100) |

## 📈 Model Performance
- **Algorithm**: Logistic Regression
- **Accuracy**: 100% on test set
- **ROC-AUC**: 1.0

## 🛠️ Tech Stack
- Python 3.x
- Scikit-learn
- Streamlit
- Pandas, NumPy, Matplotlib

---
*Made for AIML Course Project*
