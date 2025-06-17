import pandas as pd
import mlflow
import mlflow.sklearn
import os
import argparse
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load .env
load_dotenv()

# Konfigurasi MLflow
mlflow.set_tracking_uri("https://dagshub.com/nfvalenn/mental-health-Nabila-Febriyanti-Valentin.mlflow")
mlflow.set_experiment("Mental Health Prediction")

# Parsing argument CLI
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Set environment variable untuk autentikasi MLflow
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Load dataset
url = "https://raw.githubusercontent.com/nfvalenn/Eksperimen_Nabila-Febriyanti-Valentinn/main/preprocessing/mental_health_cleaned.csv"
df = pd.read_csv(url)

X = df.drop("treatment", axis=1)
y = df["treatment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)

# Nonaktifkan autolog, gunakan log manual untuk mencegah overload request
mlflow.sklearn.autolog(disable=True)

try:
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Logging manual model, parameter, dan metrik
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_metric("accuracy", acc)

        # Simpan model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Logging laporan klasifikasi sebagai file text
        report = classification_report(y_test, preds)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        print(f"Training complete. Accuracy: {acc:.4f}")

except Exception as e:
    print(f"Training gagal. Error: {e}")
