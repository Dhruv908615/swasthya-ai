# SwasthyaAI: Safe Health Triage Assistant

SwasthyaAI is a health triage assistant built for the Gemma 4 Good Hackathon. It helps users understand symptom urgency and take safer next steps using Machine Learning and AI-powered safety guidance.

## Problem

Many people delay medical help because they are unsure whether their symptoms are serious. This delay can be especially risky in low-resource or low-connectivity environments.

SwasthyaAI aims to reduce hesitation by giving users a simple risk level and safe next-step guidance.

## Vision

SwasthyaAI does not replace doctors. Its goal is to help users understand when they should seek medical help faster.

## Key Features

- Symptom-based risk prediction
- Low, Medium, and High risk classification
- ML model using TF-IDF and Logistic Regression
- AI-generated safety guidance
- Offline fallback mode when API quota or internet is unavailable
- Confidence score for model prediction
- Simple Streamlit interface
- Medical safety disclaimer

## How It Works

```text
User Symptoms
     ↓
TF-IDF Feature Extraction
     ↓
Logistic Regression ML Classifier
     ↓
Risk Level: Low / Medium / High
     ↓
AI Safety Guidance
     ↓
Offline Fallback if API fails