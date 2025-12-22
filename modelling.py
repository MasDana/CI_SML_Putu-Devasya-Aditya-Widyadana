import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

mlflow.set_experiment("CI Wine Quality Training")

df = pd.read_csv("C:/Users/Microsoft/Documents/CI_SML_PUTU DEVASYA ADITYA WIDYADANA/winequality-white_preprocessing.csv")

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)