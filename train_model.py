# hey python bring me my tools!
from sklearn.datasets import load_breast_cancer  # gives us a real medical dataset
from sklearn.model_selection import train_test_split  # splits data into practice + exam
from sklearn.linear_model import LogisticRegression  # the ML BRAIN
from sklearn.metrics import accuracy_score  # checks how smart it is
from sklearn.metrics import confusion_matrix  # shows where it messed up
from sklearn.metrics import classification_report  # detailed performance breakdown
import joblib  # lets us freeze and save the trained brain


# load the real dataset
# this dataset contains information about tumors
# the goal is to predict if a tumor is malignant (dangerous) or benign (safe)
data = load_breast_cancer()

# features (clues)
# these are numerical measurements of the tumor (like size, texture, smoothness etc.)
# X is basically all the input columns
X = data.data  

# target (answers)
# 0 = malignant
# 1 = benign
# Y is what we want the model to predict
Y = data.target  


# Split into training and testing
# 70% is the training set (learning)
# 30% (0.3) is the testing set (exam)
# 42 means use the same random shuffle every time so results don’t randomly change
# train_test_split basically takes your data -> shuffle it -> break into 2 parts
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)


# create the model (THE BRAIN)
# logistic regression draws a smart boundary between malignant and benign
# internally it uses a sigmoid function
# probability > 0.5 → class 1
# probability < 0.5 → class 0
# max_iter=5000 means "keep adjusting weights up to 5000 times if needed"
model = LogisticRegression(max_iter=5000)


# Train model
# this is where the model learns by adjusting weights (gradient descent happening behind the scenes)
# it studies patterns between tumor measurements and final diagnosis
model.fit(X_train, y_train)


# make prediction
# now we give it the test data (which it has NEVER seen before)
# it must decide 0 or 1 based on learned patterns
predictions = model.predict(X_test)


# check accuracy
# accuracy = (correct predictions) / (total predictions)
# if accuracy = 0.95 → it means 95% correct on unseen data
accuracy = accuracy_score(y_test, predictions)


# confusion matrix
# helps detect:
# true positives
# true negatives
# false positives
# false negatives
# VERY important in medical systems because wrong prediction can be serious
cm = confusion_matrix(y_test, predictions)


# print results
print("Predictions:", predictions)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

print("\nDetailed Report:")
print(classification_report(y_test, predictions))


# save the trained brain so we don’t retrain every time
# this creates a file: breast_cancer_model.pkl
joblib.dump(model, "breast_cancer_model.pkl")

print("\nModel trained and saved successfully.")