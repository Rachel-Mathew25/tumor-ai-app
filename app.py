# hey python bring me my shiny GUI tools
import customtkinter as ctk  # modern looking tkinter
import joblib  # to load our frozen trained brain
import numpy as np  # for reshaping input properly


# load the trained brain we saved earlier
model = joblib.load("breast_cancer_model.pkl")


# dark mode because we are professionals
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# create main window
app = ctk.CTk()
app.title("Breast Cancer Prediction System")
app.geometry("600x500")


# title
title_label = ctk.CTkLabel(
    app,
    text="Breast Cancer Prediction System",
    font=("Arial", 22)
)
title_label.pack(pady=20)


# input box (expects 30 values)
input_entry = ctk.CTkEntry(
    app,
    width=500,
    placeholder_text="Enter 30 comma-separated values"
)
input_entry.pack(pady=15)


# result label
result_label = ctk.CTkLabel(app, text="", font=("Arial", 16))
result_label.pack(pady=10)


# explanation box (multi-line text)
explanation_label = ctk.CTkLabel(
    app,
    text="",
    wraplength=550,
    justify="left"
)
explanation_label.pack(pady=10)


# store last probability globally
last_probability = None
last_prediction = None


# predict function
def predict():
    global last_probability, last_prediction

    try:
        user_input = input_entry.get()
        values = [float(x) for x in user_input.split(",")]

        if len(values) != 30:
            result_label.configure(text="Please enter exactly 30 values.")
            return

        input_array = np.array(values).reshape(1, -1)

        # prediction
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)

        last_probability = probability
        last_prediction = prediction

        if prediction[0] == 1:
            result_label.configure(text="Prediction: BENIGN 🟢")
        else:
            result_label.configure(text="Prediction: MALIGNANT 🔴")

        explanation_label.configure(text="")  # clear old explanation

    except:
        result_label.configure(text="Invalid input. Please enter proper numbers.")
        explanation_label.configure(text="")


# insight function
def explain():
    if last_prediction is None:
        explanation_label.configure(
            text="Please make a prediction first."
        )
        return

    malignant_prob = round(last_probability[0][0] * 100, 2)
    benign_prob = round(last_probability[0][1] * 100, 2)

    if last_prediction[0] == 0:
        explanation_text = (
            f"The model noticed patterns that are more commonly associated "
            f"with malignant tumors.\n\n"
            f"Confidence level:\n"
            f"Malignant: {malignant_prob}%\n"
            f"Benign: {benign_prob}%\n\n"
            f"This does NOT mean a confirmed diagnosis. "
            f"It simply means the measurements resemble patterns "
            f"that often require further medical evaluation.\n\n"
            f"Recommendation:\n"
            f"It would be wise to consult a medical professional "
            f"for proper clinical testing and reassurance."
        )
    else:
        explanation_text = (
            f"The model noticed patterns that are more commonly associated "
            f"with benign (non-dangerous) tumors.\n\n"
            f"Confidence level:\n"
            f"Benign: {benign_prob}%\n"
            f"Malignant: {malignant_prob}%\n\n"
            f"This suggests the measurements look similar to cases "
            f"that were non-cancerous in the training data.\n\n"
            f"Recommendation:\n"
            f"Regular checkups are always a good practice, "
            f"but this result appears reassuring."
        )

    explanation_label.configure(text=explanation_text)


# predict button
predict_button = ctk.CTkButton(
    app,
    text="Predict",
    command=predict
)
predict_button.pack(pady=10)


# explanation button
explain_button = ctk.CTkButton(
    app,
    text="Why this result?",
    command=explain
)
explain_button.pack(pady=10)


# run app
app.mainloop()