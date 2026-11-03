import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")
test_df = pd.read_csv("Iris - Iris.csv")

X = test_df.drop(columns=["Id", "Species"])
y = test_df["Species"]
pred = model.predict(X)
accuracy = accuracy_score(y, pred)
print("Accuracy Score:", accuracy)

assert accuracy > 0.8, "Accuracy is less than 80%"
assert len(pred) == len(y), "Number of predictions does not match number of samples"