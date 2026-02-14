# hey python bring me my tools!
import streamlit as st           # makes beautiful web apps easily
import pickle                    # loads our trained ML brain
import numpy as np               # handles number arrays
import matplotlib.pyplot as plt # draws graphs
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet


# page setup
# this makes the app look modern and full-width
st.set_page_config(
    page_title="AI Tumor Diagnosis System",
    layout="wide"
)


# custom styling
# this gives premium dark theme feel
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# load trained brain
# this brain already studied tumor patterns earlier
model = pickle.load(open("model.pkl", "rb"))


# session history
# this remembers past predictions while app runs
if "history" not in st.session_state:
    st.session_state.history = []


# title section
st.title("AI Tumor Diagnosis System")
st.markdown("### Early risk screening powered by Machine Learning")
st.caption("Educational tool — not a medical diagnosis")


# input area
st.subheader("Enter Tumor Measurements")

col1, col2 = st.columns(2)

with col1:
    radius = st.number_input("Mean Radius", value=14.0)
    texture = st.number_input("Mean Texture", value=20.0)
    perimeter = st.number_input("Mean Perimeter", value=90.0)

with col2:
    area = st.number_input("Mean Area", value=600.0)
    smoothness = st.number_input("Mean Smoothness", value=0.1)


# predict button
if st.button("Predict Diagnosis"):

    # convert inputs into numpy format the model understands
    values = np.array([[radius, texture, perimeter, area, smoothness]])

    # model makes prediction
    pred = model.predict(values)[0]

    # probability confidence
    prob = model.predict_proba(values)[0]
    confidence = max(prob)

    # convert numeric output to human readable
    if pred == 1:
        result_text = "BENIGN"
        color_box = "🟢"
    else:
        result_text = "MALIGNANT"
        color_box = "🔴"

    # metrics row (very professional dashboard look)
    m1, m2 = st.columns(2)

    with m1:
        st.metric("Prediction", result_text)

    with m2:
        st.metric("Confidence", f"{confidence*100:.2f}%")

    # confidence bar
    st.progress(confidence)

    # risk badge interpretation
    if confidence > 0.85:
        st.success("Low Risk Confidence")
    elif confidence > 0.60:
        st.warning("Medium Risk Confidence")
    else:
        st.error("High Risk — Needs Attention")

    # explanation box
    st.subheader("Explanation")

    explanation_text = (
        "The model analyzed the tumor measurements and compared them "
        "with patterns learned from medical data. Higher values in certain "
        "features can influence the prediction.\n\n"
        "Recommendation: Please consult a qualified medical professional "
        "for proper clinical evaluation."
    )

    st.info(explanation_text)

    # save to history
    st.session_state.history.append((result_text, confidence))


# show history graph
if st.button("Show Prediction History Graph"):

    if len(st.session_state.history) == 0:
        st.warning("No history yet")
    else:
        results = [1 if r[0] == "BENIGN" else 0 for r in st.session_state.history]

        fig = plt.figure()
        plt.plot(results, marker="o")
        plt.title("Prediction History")
        plt.xlabel("Prediction Number")
        plt.ylabel("Result (1 benign, 0 malignant)")
        st.pyplot(fig)


# save pdf button
if st.button("Save PDF Report"):

    doc = SimpleDocTemplate("report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Tumor Diagnosis Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    if len(st.session_state.history) > 0:
        last = st.session_state.history[-1]
        elements.append(
            Paragraph(
                f"Latest Prediction: {last[0]} ({last[1]*100:.2f}%)",
                styles["Heading2"]
            )
        )

    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Prediction History", styles["Heading2"]))

    list_items = [
        ListItem(
            Paragraph(f"{i+1}. {r[0]} ({r[1]*100:.2f}%)", styles["Normal"])
        )
        for i, r in enumerate(st.session_state.history)
    ]

    elements.append(ListFlowable(list_items))
    doc.build(elements)

    st.success("PDF saved as report.pdf")