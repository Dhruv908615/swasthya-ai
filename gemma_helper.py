import os
import google.generativeai as genai


def offline_guidance(symptoms, risk_level):
    risk_level = risk_level.upper()

    if risk_level == "HIGH":
        return """
⚠️ Offline Safety Guidance

Risk Level: HIGH

- Your symptoms may need urgent medical attention.
- Do not ignore chest pain, breathing difficulty, unconsciousness, severe bleeding, or blue lips.
- Visit the nearest hospital or call emergency medical help immediately.
- Avoid self-medication unless advised by a doctor.
- Disclaimer: SwasthyaAI does not diagnose disease or prescribe medicine.
"""

    elif risk_level == "MEDIUM":
        return """
⚠️ Offline Safety Guidance

Risk Level: MEDIUM

- Your symptoms should be monitored carefully.
- Consult a doctor if symptoms continue, worsen, or last more than 24–48 hours.
- Rest, drink fluids, and avoid taking medicines without proper medical advice.
- Seek urgent help if breathing difficulty, chest pain, confusion, or severe weakness appears.
- Disclaimer: SwasthyaAI does not diagnose disease or prescribe medicine.
"""

    else:
        return """
⚠️ Offline Safety Guidance

Risk Level: LOW

- Your symptoms appear low-risk based on the ML model.
- Rest, stay hydrated, and monitor your condition.
- If symptoms worsen or new serious symptoms appear, consult a doctor.
- Seek emergency care for chest pain, breathing difficulty, unconsciousness, or severe bleeding.
- Disclaimer: SwasthyaAI does not diagnose disease or prescribe medicine.
"""


def get_gemma_response(symptoms, risk_level):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return offline_guidance(symptoms, risk_level)

    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("models/gemini-2.0-flash")

        prompt = f"""
You are SwasthyaAI, a safe health triage assistant.

User symptoms:
{symptoms}

ML predicted risk level:
{risk_level}

Give simple guidance in 5 bullet points:
1. What this risk level means
2. What the user should do now
3. When to see a doctor
4. Emergency warning signs
5. Clear medical disclaimer

Rules:
- Do not diagnose disease
- Do not prescribe medicine
- Use simple language
- Be calm and helpful
"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        error_msg = str(e).lower()

        if "429" in error_msg or "quota" in error_msg:
            return f"""
⚠️ AI quota limit reached. SwasthyaAI is switching to offline safety mode.

{offline_guidance(symptoms, risk_level)}
"""

        return f"""
⚠️ Online AI service is currently unavailable. SwasthyaAI is switching to offline safety mode.

{offline_guidance(symptoms, risk_level)}
"""