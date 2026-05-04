import streamlit as st
import joblib
from gemma_helper import get_gemma_response


st.set_page_config(
    page_title="SwasthyaAI",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    margin-bottom: 24px;
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 8px;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: #cbd5e1;
}

.info-card {
    padding: 18px;
    border-radius: 16px;
    background-color: #1e293b;
    border: 1px solid #334155;
    margin-bottom: 16px;
    color: white;
    min-height: 90px;
}

.info-card b {
    color: #93c5fd;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-top: 30px;
}

.small-note {
    color: #94a3b8;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load ML model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("health_risk_model.pkl")


def get_confidence(model, text, predicted_risk):
    try:
        probabilities = model.predict_proba([text])[0]
        classes = list(model.classes_)
        risk_index = classes.index(predicted_risk)
        confidence = probabilities[risk_index] * 100
        return round(confidence, 2)
    except Exception:
        return None


model = load_model()


# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">🏥 SwasthyaAI</div>
    <div class="hero-subtitle">
        Safe health triage assistant using Machine Learning + AI safety guidance
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Main Layout
# -----------------------------
left_col, right_col = st.columns([1.2, 0.8])

with left_col:
    st.markdown("## Enter Symptoms")

    symptoms = st.text_area(
        "Describe symptoms in simple words:",
        placeholder="Example: I have chest pain and breathing difficulty",
        height=160
    )

    check_button = st.button("Check Risk", use_container_width=True)

with right_col:
    st.markdown("## Why SwasthyaAI?")

    st.markdown("""
<div class="info-card">
<b>Purpose:</b><br>
Helps users understand symptom urgency and take safer next steps.
</div>

<div class="info-card">
<b>Built for:</b><br>
Low-resource and low-connectivity environments.
</div>

<div class="info-card">
<b>Safety:</b><br>
Does not diagnose disease or prescribe medicine.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Prediction Logic
# -----------------------------
if check_button:
    if symptoms.strip() == "":
        st.warning("Please enter your symptoms first.")
    else:
        risk = model.predict([symptoms])[0]
        confidence = get_confidence(model, symptoms, risk)

        st.markdown("---")
        st.markdown("## Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            if risk == "low":
                st.success("🟢 LOW RISK")
            elif risk == "medium":
                st.warning("🟡 MEDIUM RISK")
            else:
                st.error("🔴 HIGH RISK")

        with col2:
            if confidence is not None:
                st.metric("Model Confidence", f"{confidence}%")
            else:
                st.metric("Model Confidence", "N/A")

        with col3:
            st.metric("Guidance Mode", "Online / Offline Safe")

        st.markdown("## AI Safety Guidance")

        with st.spinner("Generating safety guidance..."):
            explanation = get_gemma_response(symptoms, risk)

        st.write(explanation)

        st.info(
            "⚠️ Disclaimer: SwasthyaAI is not a medical diagnosis tool. "
            "It does not replace doctors. In emergencies, contact a hospital immediately."
        )


# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
Built for Gemma 4 Good Hackathon | ML Risk Classification + AI Safety Guidance + Offline Fallback
</div>
""", unsafe_allow_html=True)