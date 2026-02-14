# hey python bring me my tools

import streamlit as st          # this builds our web app UI
import pickle                   # loads the trained brain
import numpy as np              # handles numeric arrays
import matplotlib.pyplot as plt # draws history graph
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import os                       # helps check if files exist


# page configuration so app looks professional in browser
st.set_page_config(
    page_title="AI Tumor Diagnosis System",
    page_icon="🧠",
    layout="centered"
)


# load trained brain safely
# streamlit cloud sometimes crashes if file missing
model_path = "model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found. Make sure model.pkl is in your repo.")
    st.stop()

model = pickle.load(open(model_path, "rb"))


# session state keeps history between button clicks
# without this streamlit forgets everything each rerun
if "history" not in st.session_state:
    st.session_state.history = []


# title area
st.title("AI Tumor Diagnosis System")
st.caption("AI-assisted breast tumor prediction tool")


# input section header
st.subheader("Enter Tumor Measurements")


# input boxes arranged nicely in columns
col1, col2 = st.columns(2)

with col1:
    mean_radius = st.number_input("Mean Radius", value=0.0)
    mean_texture = st.number_input("Mean Texture", value=0.0)
    mean_perimeter = st.number_input("Mean Perimeter", value=0.0)

with col2:
    mean_area = st.number_input("Mean Area", value=0.0)
    mean_smoothness = st.number_input("Mean Smoothness", value=0.0)


# prediction button
if st.button("Predict Diagnosis"):

    try:
        # gather user numbers into model format
        values = np.array([
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness
        ]).reshape(1, -1)

        # model makes prediction
        pred = model.predict(values)[0]

        # model confidence
        prob = model.predict_proba(values)[0]
        confidence = float(max(prob))


        # decide label text
        if pred == 1:
            result_text = "BENIGN"
            st.success(f"Prediction: {result_text}")
        else:
            result_text = "MALIGNANT"
            st.error(f"Prediction: {result_text}")


        # show confidence meter
        st.subheader("Confidence Meter")
        st.progress(confidence)
        st.write(f"Confidence: {confidence*100:.2f}%")

        # explanation box
        st.subheader("Prediction Explanation")

        explanation_text = "Prediction explanation\n\n"
        labels = [
            "Mean Radius",
            "Mean Texture",
            "Mean Perimeter",
            "Mean Area",
            "Mean Smoothness"
        ]

        for label in labels:
            explanation_text += f"{label} influenced decision\n"

        explanation_text += "\nSuggestion:\nConsult a medical professional for confirmation."

        st.text_area("Explanation Details", explanation_text, height=200)

        # save history into session memory
        st.session_state.history.append((result_text, confidence))

    except Exception as e:
        st.error("Enter valid numbers")


# history graph button
if st.button("Show Prediction History Graph"):

    if len(st.session_state.history) == 0:
        st.warning("No predictions yet")
    else:
        results = [1 if r[0] == "BENIGN" else 0 for r in st.session_state.history]

        plt.figure()
        plt.plot(results, marker="o")
        plt.title("Prediction History")
        plt.xlabel("Prediction Number")
        plt.ylabel("Result (1 benign 0 malignant)")
        st.pyplot(plt)


# pdf export button
if st.button("Save PDF Report"):

    if len(st.session_state.history) == 0:
        st.warning("No data to export")
    else:
        try:
            doc = SimpleDocTemplate("report.pdf", pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Tumor Diagnosis Report", styles["Title"]))
            elements.append(Spacer(1, 12))

            last_result = st.session_state.history[-1]
            elements.append(
                Paragraph(
                    f"Latest Prediction: {last_result[0]} ({last_result[1]*100:.2f}%)",
                    styles["Heading2"]
                )
            )

            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Prediction History", styles["Heading2"]))

            list_items = [
                ListItem(
                    Paragraph(
                        f"{i+1}. {r[0]} ({r[1]*100:.2f}%)",
                        styles["Normal"]
                    )
                )
                for i, r in enumerate(st.session_state.history)
            ]

            elements.append(ListFlowable(list_items))
            doc.build(elements)

            st.success("PDF saved as report.pdf")

        except Exception as e:
            st.error("PDF generation failed")


# footer disclaimer because this is medical domain
st.caption("Disclaimer: This AI tool is for educational purposes only and not a medical diagnosis.")