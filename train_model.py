# hey python bring me my tools 😌
from sklearn.datasets import load_breast_cancer   # gives us real tumor data
from sklearn.pipeline import Pipeline   # lets us build a smart processing machine
from sklearn.preprocessing import StandardScaler   # fixes number scale drama
from sklearn.linear_model import LogisticRegression   # the prediction brain 🧠
import pickle   # used to store our trained genius safely


# ok python go fetch medical dataset
# this dataset contains measurements of tumors
# our mission → predict if tumor is dangerous or safe
data = load_breast_cancer()


# choosing features
# frontend only sends 5 numbers
# so model MUST learn from same 5 numbers
# otherwise model will throw tantrum and crash
X = data.data[:, [0,1,2,3,4]]


# answers column
# 0 = malignant (danger 😬)
# 1 = benign (safe 😌)
y = data.target


# building smart pipeline machine
# input goes in → scaler fixes values → brain predicts
model = Pipeline([

    # scaler makes sure all numbers are balanced
    # because area can be 600
    # while smoothness is like 0.09
    # without scaling big numbers bully small ones
    ("scaler", StandardScaler()),

    # logistic regression = decision maker
    # draws smart boundary between benign and malignant
    ("brain", LogisticRegression(max_iter=5000))
])


# training time 😤📚
# model studies patterns between tumor measurements and diagnosis
model.fit(X, y)


# save the trained brain
# so we don't retrain every time app runs
pickle.dump(model, open("model.pkl", "wb"))


# dramatic success message 😎
print("model trained successfully — brain saved as model.pkl")
