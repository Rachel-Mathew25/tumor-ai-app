# hey python bring me my tools!

import tkinter as tk   # creates window + buttons + inputs
import joblib   # loads saved ML brain


# load trained model
# this loads the already trained pipeline from file
# meaning we DON'T train again (fast + professional way)
model = joblib.load("model.pkl")


# create window
# this is the main app window users see
window = tk.Tk()
window.title("Tumor Predictor AI")
window.geometry("500x650")


# heading
# big title text at top
title = tk.Label(window, text="Tumor Prediction System", font=("Arial", 18))
title.pack(pady=10)


# labels for inputs
# these are tumor measurement names
labels = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness"
]

entries = []   # will store input boxes


# create input boxes dynamically
# loop creates label + textbox for each feature
for text in labels:
    label = tk.Label(window, text=text)
    label.pack()

    entry = tk.Entry(window)
    entry.pack()

    entries.append(entry)


# result text label
# this will display prediction result
result_label = tk.Label(window, text="", font=("Arial", 14))
result_label.pack(pady=20)


# explanation label
# gives user friendly explanation
explain_label = tk.Label(window, text="", wraplength=400, font=("Arial", 11))
explain_label.pack(pady=10)


# prediction function
# runs when user presses button
def predict():

    # get user input from textboxes
    # convert text → float numbers
    values = [float(e.get()) for e in entries]

    # model expects 2D list format
    # so we wrap values inside []
    prediction = model.predict([values])

    # probability scores
    # tells how confident model is
    probability = model.predict_proba([values])

    # probability of malignant (class 0)
    risk = probability[0][0]


    # logic for result display
    if prediction[0] == 0:

        result = "Malignant ⚠"
        color = "red"

        explanation = (
            "The measurements match patterns seen in risky tumors.\n"
            "This does NOT mean you have cancer.\n"
            "Please consult a medical professional for proper diagnosis."
        )

    else:

        result = "Benign ✅"
        color = "green"

        explanation = (
            "The measurements match patterns seen in non-dangerous tumors.\n"
            "This is a good sign, but always confirm with a doctor."
        )


    # show result text
    # also show probability score
    result_label.config(
        text=f"{result}\nRisk Score: {risk:.2f}",
        fg=color
    )


    # show explanation message
    explain_label.config(text=explanation)



# predict button
# when clicked → calls predict()
btn = tk.Button(window, text="Predict", command=predict)
btn.pack(pady=20)


# run window loop
# keeps window alive until user closes it
window.mainloop()
