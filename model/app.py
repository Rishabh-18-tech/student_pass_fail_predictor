import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .main { background-color: #f0f4ff; }
    .result-pass {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 8px 24px rgba(0,184,148,0.3);
    }
    .result-fail {
        background: linear-gradient(135deg, #d63031, #e17055);
        color: white;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 8px 24px rgba(214,48,49,0.3);
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .stSlider > div > div > div > div {
        background: #6c63ff;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────
def load_model():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("dataset/students.csv")
    X = df[["hours_studied","attendance","prev_score","assignments_done"]]
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y)
    return model, scaler

model, scaler = load_model()

# ── Header ───────────────────────────────────────────────────────
st.markdown("# 🎓 Student Pass/Fail Predictor")
st.markdown("#### Powered by Logistic Regression | AIML Project")
st.markdown("---")

# ── Sidebar — Input ──────────────────────────────────────────────
st.sidebar.header("📋 Enter Student Details")
student_name = st.sidebar.text_input("Student Name (optional)", placeholder="e.g. Rishabh")

hours       = st.sidebar.slider("📚 Hours Studied per Day",    0, 12, 5)
attendance  = st.sidebar.slider("🏫 Attendance (%)",           0, 100, 75)
prev_score  = st.sidebar.slider("📝 Previous Exam Score",      0, 100, 60)
assignments = st.sidebar.slider("✅ Assignments Completed (%)", 0, 100, 70)

predict_btn = st.sidebar.button("🔍 Predict Result", use_container_width=True)

# ── Main Panel ───────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div style='font-size:28px'>📚</div>
        <b>{hours}h</b><br><small>Study/Day</small>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div style='font-size:28px'>🏫</div>
        <b>{attendance}%</b><br><small>Attendance</small>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div style='font-size:28px'>📝</div>
        <b>{prev_score}</b><br><small>Prev Score</small>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div style='font-size:28px'>✅</div>
        <b>{assignments}%</b><br><small>Assignments</small>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

if predict_btn:
    # Predict
    input_data   = np.array([[hours, attendance, prev_score, assignments]])
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    pass_prob = probability[1] * 100
    fail_prob = probability[0] * 100

    name_display = f"{student_name} will" if student_name else "Student will"

    if prediction == 1:
        st.markdown(f"""<div class="result-pass">
            🎉 {name_display} PASS the exam!
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-fail">
            ❌ {name_display} FAIL the exam.
        </div>""", unsafe_allow_html=True)

    # Probability Bar
    st.markdown("### 📊 Prediction Confidence")
    prog_col1, prog_col2 = st.columns(2)
    with prog_col1:
        st.metric("✅ Pass Probability", f"{pass_prob:.1f}%")
        st.progress(int(pass_prob))
    with prog_col2:
        st.metric("❌ Fail Probability", f"{fail_prob:.1f}%")
        st.progress(int(fail_prob))

    # Feature Importance Chart
    st.markdown("### 📈 How Each Factor Influenced the Prediction")
    features   = ["Hours Studied", "Attendance", "Prev Score", "Assignments"]
    coefs      = model.coef_[0]
    colors     = ["#00b894" if c > 0 else "#d63031" for c in coefs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(features, coefs, color=colors, edgecolor='white', height=0.5)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Coefficient Weight", fontsize=11)
    ax.set_title("Feature Impact on Prediction", fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, coefs):
        ax.text(val + (0.03 if val >= 0 else -0.03),
                bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va='center',
                ha='left' if val >= 0 else 'right', fontsize=10)
    green_patch = mpatches.Patch(color='#00b894', label='Positive (helps pass)')
    red_patch   = mpatches.Patch(color='#d63031', label='Negative (risk factor)')
    ax.legend(handles=[green_patch, red_patch], loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)

    # Tips
    st.markdown("### 💡 Personalized Tips")
    if hours < 4:
        st.warning("📚 Study more! Aim for at least 5-6 hours per day.")
    if attendance < 70:
        st.warning("🏫 Attendance is low! Try to attend at least 75% of classes.")
    if prev_score < 50:
        st.warning("📝 Previous score is low. Focus on weak subjects.")
    if assignments < 60:
        st.warning("✅ Complete more assignments to build understanding.")
    if prediction == 1:
        st.success("🌟 Great performance! Keep it up and you'll do well.")

else:
    # Default info page
    st.markdown("### 👈 Enter details in the sidebar and click **Predict Result**")
    st.info("""
    **This app predicts whether a student will Pass or Fail** based on:
    - 📚 Daily study hours
    - 🏫 Attendance percentage
    - 📝 Previous exam score
    - ✅ Assignment completion rate
    """)

    # Show sample dataset
    st.markdown("### 📂 Sample Training Data")
    try:
        df = pd.read_csv("dataset/students.csv")
        df["result"] = df["result"].map({1: "✅ Pass", 0: "❌ Fail"})
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Dataset: {len(df)} student records | Balanced: 50% Pass, 50% Fail")
    except:
        st.warning("Dataset not found. Run train.py first.")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>🎓 AIML Project | Logistic Regression | Made with Streamlit</small></center>",
    unsafe_allow_html=True
)
