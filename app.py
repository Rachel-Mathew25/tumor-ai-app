# hey python bring me my tools

# streamlit -> turns our Python script into a web app UI
import streamlit as st

# pickle -> loads our trained ML model from file
import pickle

# numpy -> helps handle numerical arrays for prediction
import numpy as np

# matplotlib -> used to draw prediction history graph
import matplotlib.pyplot as plt

# reportlab -> used to generate downloadable PDF report
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# io -> lets us create in-memory files (needed for Streamlit downloads)
import io


# load trained brain
# this brain already studied tumor patterns earlier
# we saved it earlier using pickle during model training
model = pickle.load(open("model.pkl","rb"))


# app title
# this shows at the top of the web app
st.title("AI Tumor Diagnosis System")


# input boxes
# these are tumor measurements user enters
labels = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness"
]

# we collect all user inputs into this list
values = []

# create a number input for each feature
for label in labels:
    val = st.number_input(label, value=0.0)
    values.append(val)


# result text placeholder
# empty container that we update after prediction
result_placeholder = st.empty()


# confidence bar
# shows how confident the model is
st.write("Confidence")
confidence_bar = st.progress(0)


# explanation box
# shows human-readable explanation of prediction
explain_placeholder = st.empty()


# history list stored in session
# session_state keeps data while user interacts with app
if "history" not in st.session_state:
    st.session_state.history = []

# stores last prediction text (needed for PDF)
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# stores last explanation (needed for PDF)
if "last_explain" not in st.session_state:
    st.session_state.last_explain = ""


# prediction function
# runs when user clicks Predict button
def predict():

    try:
        # convert user inputs into numpy array shape expected by model
        arr = np.array(values).reshape(1,-1)

        # model predicts class (0 or 1)
        pred = model.predict(arr)[0]

        # model gives probability for confidence
        prob = model.predict_proba(arr)[0]
        confidence = float(max(prob))

        # convert numeric prediction into human label
        if pred == 1:
            result = "BENIGN"
            color = "green"
        else:
            result = "MALIGNANT"
            color = "red"

        # show prediction result nicely formatted
        text = f"Prediction: {result}   Confidence: {confidence*100:.2f}%"
        result_placeholder.markdown(
            f"<h3 style='color:{color}'>{text}</h3>",
            unsafe_allow_html=True
        )

        # update confidence progress bar
        confidence_bar.progress(confidence)

        # build explanation text
        explanation_text = "Prediction explanation\n\n"

        # simple feature influence explanation
        for i,v in enumerate(arr[0]):
            explanation_text += f"{labels[i]} influenced decision\n"

        explanation_text += "\nSuggestion:\nConsult a medical professional for confirmation."

        # display explanation
        explain_placeholder.text(explanation_text)

        # save for PDF + history
        st.session_state.last_text = text
        st.session_state.last_explain = explanation_text
        st.session_state.history.append((result,confidence))

    except:
        # if user enters invalid data
        result_placeholder.markdown(
            "<h3 style='color:orange'>Enter valid numbers</h3>",
            unsafe_allow_html=True
        )


# graph function
# shows how predictions changed over time
def show_graph():

    # if no predictions yet
    if len(st.session_state.history) == 0:
        st.warning("No history yet")
        return

    # convert BENIGN/MALIGNANT into numeric for plotting
    results = [1 if r[0]=="BENIGN" else 0 for r in st.session_state.history]

    # create matplotlib figure
    fig = plt.figure()
    plt.plot(results, marker="o")
    plt.title("Prediction History")
    plt.xlabel("Prediction Number")
    plt.ylabel("Result (1 benign 0 malignant)")

    # display graph inside Streamlit
    st.pyplot(fig)


# pdf export function
# generates downloadable medical report
def save_pdf():

    # create in-memory file buffer
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # report title
    elements.append(Paragraph("Tumor Diagnosis Report", styles["Title"]))
    elements.append(Spacer(1,12))

    # latest prediction
    elements.append(Paragraph(st.session_state.last_text, styles["Heading2"]))
    elements.append(Spacer(1,12))

    # explanation text
    for line in st.session_state.last_explain.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))

    elements.append(Spacer(1,12))

    # prediction history section
    elements.append(Paragraph("Prediction History", styles["Heading2"]))
    list_items = [
        ListItem(
            Paragraph(f"{i+1}. {r[0]} ({r[1]*100:.2f}%)", styles["Normal"])
        )
        for i,r in enumerate(st.session_state.history)
    ]
    elements.append(ListFlowable(list_items))

    # build PDF
    doc.build(elements)

    buffer.seek(0)

    # download button appears
    st.download_button(
        label="Download PDF Report",
        data=buffer,
        file_name="report.pdf",
        mime="application/pdf"
    )


# buttons

# predict button
if st.button("Predict Diagnosis"):
    predict()

# graph button
if st.button("Show Graph"):
    show_graph()

# pdf button
if st.button("Save PDF Report"):
    save_pdf()