# ✈️ Clarity TTS: Data Science & AI Architecture Assessment
**Candidate:** Vinesh J  
**Role:** Data Scientist / ML Engineer  
**Objective:** Deliver a full-stack, production-ready ML and GenAI architecture to resolve B2B travel operational bottlenecks.

---

## 📑 Executive Summary
This repository contains a comprehensive, end-to-end solution for the Clarity TTS assessment. Built strictly within the constrained timeframe, the project demonstrates both traditional predictive modeling and cutting-edge Generative AI orchestration. 

The deliverable is divided into two operational cores:
1. **The Predictive Core (Phase 1 & 2):** A highly tuned XGBoost classification pipeline utilizing Bayesian Target Encoding and Game-Theoretic SHAP analysis to predict B2B booking cancellations.
2. **The Agentic Core (Phase 3):** A multi-agent GenAI triage application built with LangGraph and Streamlit, designed to reduce human handle-time of complex support tickets from minutes to seconds.

---

## 🛠️ Architecture & Tech Stack
* **Machine Learning:** `XGBoost`, `Scikit-Learn`, `Category_Encoders`
* **Explainability:** `SHAP` (SHapley Additive exPlanations)
* **GenAI Orchestration:** `LangGraph`, `LangChain`, `OpenAI (GPT-4o-mini)`
* **Frontend UI:** `Streamlit` (Native SaaS layout)
* **Data Processing:** `Pandas`, `NumPy`

---

## 🧠 Part 1 & 2: Predictive Risk Modeling
**Primary File:** `Clarity_Assessment_Vinesh.ipynb`

The goal of this phase was to build a robust model capable of identifying high-risk booking cancellations before they occur, allowing business operations to proactively manage revenue leakage.

### Step 1: Exploratory Data Analysis & Feature Engineering
Rather than blindly feeding data into a model, I engineered domain-specific features to capture human behavioral nuances:
* **Temporal Dynamics:** Extracted `lead_time_days` to understand how the booking window affects cancellation probability.
* **Pricing Ratios:** Engineered features like `fare_to_avg_route_ratio` to mathematically capture behavioral anomalies (e.g., "buyer's remorse" on high-markup flights versus lack of financial commitment on heavily discounted fares).
* **Bayesian Target Encoding:** Seamlessly handled massive high-cardinality categorical features (like specific flight routes and agency IDs) by replacing them with a blend of posterior probabilities, preventing the dimensionality explosion that ruins standard One-Hot Encoding.

### Step 2: Model Architecture (XGBoost)
* **Threshold Dynamics & Class Imbalance:** Instead of optimizing for raw accuracy, I dynamically calculated `scale_pos_weight` to force the gradient descent to heavily penalize false negatives. In a business context, missing a high-risk cancellation is far more costly than falsely flagging a safe booking.

### Step 3: Game-Theoretic Explainability (SHAP)
Black-box models are unacceptable in enterprise environments. Instead of relying on biased, tree-based feature importance, I implemented SHAP (SHapley Additive exPlanations) to decode the exact marginal contribution of every feature.
* **Global Interpretability:** Allowed us to see exactly which features (like lead time and pricing ratios) drive the macro behavior of the cancellation system.
* **Local Interpretability:** Proved that the model can explain *why* it flagged a specific individual booking, which is critical for operational trust.

---

## 🤖 Part 3: Agentic Support Triage (GenAI Application)
**Primary File:** `app.py`

To solve the Bonus Phase, I engineered a production-ready **Multi-Agent Triage Dashboard**. Rather than deploying a basic clustering algorithm (like K-Means) which only sorts data, this application acts as a digital employee that actively reduces the human support backlog.

### The Problem
B2B travel platforms process thousands of support tickets daily. Human agents waste significant operational hours reading emotional, unstructured emails, manually categorizing them, and hunting for PNRs before they can even begin solving the core issue.

### The LangGraph Multi-Agent Solution
I designed a stateful, directed computational graph using **LangGraph** and **GPT-4o-mini**. The system splits the cognitive load across distinct AI nodes to ensure zero hallucinations and strict data extraction:

1. **Step 1 - The Classifier Node (Agent 1):** Ingests raw, emotional human text. It deterministically strips away the noise and extracts the **Business Category**, assigns a definitive **System Urgency** (High/Medium/Low), and isolates **Key Entities** (PNRs, Flight Numbers).
2. **Step 2 - The Synthesis Node (Agent 2):** Ingests the structured data from Step 1 and generates a cold, concise 3-bullet summary of the operational problem.
3. **Step 3 - The Action Output:** The system concludes by drafting a definitive, bolded **"Recommended Next Action"** for the human support team.

**Business Impact:** This pipeline reduces human ticket-handle-time from ~10 minutes to under 60 seconds, saving massive operational costs and ensuring VIP issues are instantly escalated.

---

## 📸 Dashboard Visuals
*(Screenshots of the Agentic Triage UI in action)*

![UI Overview]("D:\clarity_assessment\assets\dashboard.png")

![Agent Execution]("D:\clarity_assessment\assets\agent_running.png")

---

## 🚀 Quickstart: How to Run Locally

### 1. Environment Setup
Ensure you are using Python 3.10+ and install the strict dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Predictive Pipeline
Open the Jupyter Notebook to review the EDA, XGBoost training, and SHAP visualizations:
```bash
jupyter notebook Clarity_Assessment_Vinesh.ipynb
```

### 3. Running the Agentic Triage App
Launch the LangGraph Streamlit application:
```bash
streamlit run app.py
```
> **Note:** You will need to input your API key (`sk-...`) into the secure left-hand sidebar control panel to initialize the LLM orchestration engine. The application will not process tickets without it.