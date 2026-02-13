# hey python bring me my tools!

from sklearn.datasets import load_breast_cancer   # gives us real medical dataset
from sklearn.model_selection import train_test_split   # splits data into practice + exam
from sklearn.linear_model import LogisticRegression   # the ML BRAIN
from sklearn.metrics import accuracy_score   # checks how smart it is
from sklearn.metrics import confusion_matrix   # shows where it messed up
from sklearn.pipeline import Pipeline   # automatic machine that runs steps in order
from sklearn.preprocessing import StandardScaler   # scales numbers so model understands better


# load real dataset
# contains tumor measurements
# goal → predict if tumor is malignant or benign
data = load_breast_cancer()


# features (clues)
# these are measurements like radius, texture, smoothness etc.
# X = input information
X = data.data


# target (answers)
# 0 = malignant (dangerous)
# 1 = benign (safe)
# Y = what model must predict
Y = data.target


# split data into training + testing
# 70% learning material
# 30% exam paper
# 42 = same shuffle every time so results stay consistent
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)


# create pipeline (SMART MACHINE)
# pipeline = steps connected in order
# step1 → scaler (normalizes numbers)
# step2 → model (learns patterns)
pipeline = Pipeline([
    ("scaler", StandardScaler()),   # makes values comparable
    ("model", LogisticRegression(max_iter=5000))   # brain that learns
])


# train pipeline
# internally this does:
# scale training data → then train model
pipeline.fit(X_train, y_train)


# make predictions
# pipeline automatically:
# scales test data → sends to model → gives prediction
predictions = pipeline.predict(X_test)


# check accuracy
# accuracy = correct predictions / total predictions
accuracy = accuracy_score(y_test, predictions)


# confusion matrix
# shows exactly where model was right and wrong
# helps detect:
# true positives
# true negatives
# false positives
# false negatives
cm = confusion_matrix(y_test, predictions)


# print results
print("Predictions:", predictions)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

# save trained pipeline to file
# joblib is used to store trained ML models
import joblib

joblib.dump(pipeline, "model.pkl")

print("Model saved successfully!")

