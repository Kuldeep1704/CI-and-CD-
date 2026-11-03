import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
file_path = "Iris - Iris.csv"
df = pd.read_csv(file_path)

# Prepare feature and target variables
X = df.drop(columns=["Id", "Species"])
y = df["Species"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("Accuracy Score:", accuracy)

# Save model to match the test script
joblib.dump(model, "model.pkl")
print("Model saved successfully")
