# train_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow

def train_and_save():
    mlflow.start_run()
    
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "model.joblib")
    print(f"Model saved with accuracy: {acc:.4f}")

    mlflow.end_run()

if __name__ == "__main__":
    train_and_save()
