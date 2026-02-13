# hey python bring me my tools
import customtkinter as ctk   # modern looking GUI
import pickle                 # loads trained brain
import numpy as np            # handles numbers
import matplotlib.pyplot as plt   # draws graph
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet


# load trained brain
# this brain already studied tumor patterns earlier
model = pickle.load(open("model.pkl","rb"))


# app window
app = ctk.CTk()
app.title("AI Tumor Diagnosis System")
app.geometry("700x900")


# scrollable area so UI never cuts off
container = ctk.CTkScrollableFrame(app,width=650,height=850)
container.pack(pady=10)


# title
title = ctk.CTkLabel(container,text="AI Tumor Diagnosis System",font=("Arial",28,"bold"))
title.pack(pady=20)


# input boxes
# these are tumor measurements user enters
labels = ["Mean Radius","Mean Texture","Mean Perimeter","Mean Area","Mean Smoothness"]
entries = []

for label in labels:
    ctk.CTkLabel(container,text=label,font=("Arial",16)).pack(pady=5)
    entry = ctk.CTkEntry(container,width=300,height=35)
    entry.pack(pady=5)
    entries.append(entry)


# result text
result_label = ctk.CTkLabel(container,text="",font=("Arial",18))
result_label.pack(pady=15)


# confidence bar
confidence_label = ctk.CTkLabel(container,text="Confidence",font=("Arial",16))
confidence_label.pack()

confidence_bar = ctk.CTkProgressBar(container,width=400)
confidence_bar.pack(pady=10)
confidence_bar.set(0)


# explanation box
explain_box = ctk.CTkTextbox(container,width=500,height=120)
explain_box.pack(pady=15)


# history list
history = []


# prediction function
def predict():

    try:
        # get numbers from entry boxes
        values = [float(e.get()) for e in entries]
        values = np.array(values).reshape(1,-1)

        # predict class
        pred = model.predict(values)[0]

        # probability confidence
        prob = model.predict_proba(values)[0]
        confidence = max(prob)

        # show result text
        if pred == 1:
            result = "BENIGN"
            color="lightgreen"
        else:
            result = "MALIGNANT"
            color="red"

        result_label.configure(text=f"Prediction: {result}   Confidence: {confidence*100:.2f}%",text_color=color)

        # update bar
        confidence_bar.set(confidence)

        # explanation text
        explain_box.delete("1.0","end")
        explain_box.insert("end","Prediction explanation\n\n")

        for i,v in enumerate(values[0]):
            explain_box.insert("end",f"{labels[i]} influenced decision\n")

        explain_box.insert("end","\nSuggestion:\nConsult a medical professional for confirmation.")

        # save history
        history.append((result,confidence))

    except:
        result_label.configure(text="Enter valid numbers",text_color="orange")



# graph function
def show_graph():

    if len(history)==0:
        return

    results=[1 if r[0]=="BENIGN" else 0 for r in history]

    plt.plot(results,marker="o")
    plt.title("Prediction History")
    plt.xlabel("Prediction Number")
    plt.ylabel("Result (1 benign 0 malignant)")
    plt.show()



# pdf export function
def save_pdf():

    doc = SimpleDocTemplate("report.pdf",pagesize=letter)
    styles = getSampleStyleSheet()
    elements=[]

    elements.append(Paragraph("Tumor Diagnosis Report",styles["Title"]))
    elements.append(Spacer(1,12))

    elements.append(Paragraph(result_label.cget("text"),styles["Heading2"]))
    elements.append(Spacer(1,12))

    text = explain_box.get("1.0","end")
    for line in text.split("\n"):
        elements.append(Paragraph(line,styles["Normal"]))

    elements.append(Spacer(1,12))

    elements.append(Paragraph("Prediction History",styles["Heading2"]))
    list_items = [ListItem(Paragraph(f"{i+1}. {r[0]} ({r[1]*100:.2f}%)",styles["Normal"])) for i,r in enumerate(history)]
    elements.append(ListFlowable(list_items))

    doc.build(elements)

    result_label.configure(text="PDF saved as report.pdf",text_color="cyan")



# buttons
predict_btn = ctk.CTkButton(container,text="Predict Diagnosis",command=predict,width=250,height=40)
predict_btn.pack(pady=10)

graph_btn = ctk.CTkButton(container,text="Show Graph",command=show_graph,width=250,height=40)
graph_btn.pack(pady=10)

pdf_btn = ctk.CTkButton(container,text="Save PDF Report",command=save_pdf,width=250,height=40)
pdf_btn.pack(pady=10)


# run app
app.mainloop()
