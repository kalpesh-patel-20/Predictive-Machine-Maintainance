from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

app = Flask(__name__)

# Globals
MODEL_PATH = "model/decision_tree.pkl"
scaler = StandardScaler()
encoder = LabelEncoder()
model = None
df = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    global df
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("upload.html", message="No file selected!")

        file = request.files['file']
        if file.filename == '':
            return render_template("upload.html", message="No file selected!")

        if file.filename.endswith(".csv"):
            try:
                # Load dataset
                df = pd.read_csv(file)
                return redirect(url_for("train_page"))
            except Exception as e:
                return render_template("upload.html", message=f"Error reading file: {e}")
        else:
            return render_template("upload.html", message="Please upload a valid CSV file!")
    return render_template("upload.html")

@app.route("/train", methods=["GET", "POST"])
def train_page():
    global df, model, scaler, encoder

    if df is None:
        return render_template("train.html", message="No dataset uploaded. Please upload a dataset first!")

    if request.method == "POST":
        try:
            # Data Preprocessing
            data_cleaned = df.drop(columns=["UDI", "Product ID","Failure Type","Type"])

            numeric_columns = ["Air temperature [K]", "Process temperature [K]", 
                               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
            data_cleaned[numeric_columns] = scaler.fit_transform(data_cleaned[numeric_columns])  

            # Split data into features and target
            X = data_cleaned.drop(columns=["Target"])
            y = data_cleaned["Target"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Train a Decision Tree model
            model = DecisionTreeClassifier(random_state=42, max_depth=5)
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            # Save the model
            os.makedirs("model", exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)

            return render_template("train.html", 
                                   message="Model trained successfully!", 
                                   accuracy=round(accuracy, 4), 
                                   report=report)
        except Exception as e:
            return render_template("train.html", message=f"Error during training: {e}")
    return render_template("train.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    global model, scaler, encoder

    if model is None:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        else:
            return render_template("predict.html", message="No trained model found. Please train the model first!")

    if request.method == "POST":
        try:
            # Get input data from the form
            input_data = {
                "Air temperature [K]": float(request.form["Air temperature [K]"]),
                "Process temperature [K]": float(request.form["Process temperature [K]"]),
                "Rotational speed [rpm]": int(request.form["Rotational speed [rpm]"]),
                "Torque [Nm]": float(request.form["Torque [Nm]"]),
                "Tool wear [min]": int(request.form["Tool wear [min]"])
            }

            input_df = pd.DataFrame([input_data])

            numeric_columns = ["Air temperature [K]", "Process temperature [K]", 
                               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            confidence = max(model.predict_proba(input_df)[0])

            return render_template("predict.html", 
                                   message="Prediction completed!",
                                   prediction="Failure" if prediction == 1 else "No Failure",
                                   confidence=round(confidence, 2),
                                   air_temp=input_data["Air temperature [K]"],
                                   process_temp=input_data["Process temperature [K]"],
                                   speed=input_data["Rotational speed [rpm]"],
                                   torque=input_data["Torque [Nm]"],
                                   tool_wear=input_data["Tool wear [min]"])
        
        except Exception as e:
            return render_template("predict.html", message=f"Error: {str(e)}")
    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
