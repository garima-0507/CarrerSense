import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os

SKILL_KEYWORDS = [
    "python", "ml", "pandas", "numpy",
    "sql", "excel", "statistics", "data",
    "html", "css", "javascript", "react",
    "java", "android", "kotlin",
    "networking", "linux", "security", "firewall",
    "aws", "cloud", "docker", "azure"
]

INTEREST_KEYS = ["ml", "web", "data", "android", "cyber", "cloud"]

# ---- Helper: convert text -> feature vector ----
def build_features(skills_str: str, interest_str: str) -> list:
    skills_list = [s.strip().lower() for s in skills_str.split(",")]

    # skill part (binary)
    skill_vector = [1 if kw in skills_list else 0 for kw in SKILL_KEYWORDS]

    interest_str = interest_str.strip().lower()
    # one-hot for interests
    interest_vector = [1 if interest_str == key else 0 for key in INTEREST_KEYS]

    return skill_vector + interest_vector


def main():
    # Load dataset
    df = pd.read_csv("dataset/careers.csv")

    X = []
    y = []

    for _, row in df.iterrows():
        features = build_features(row["skills"], row["interests"])
        X.append(features)
        y.append(row["role"])

    X = np.array(X)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create model directory if not exists
    os.makedirs("model", exist_ok=True)

    # Save model
    with open("model/career_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved to model/career_model.pkl")
    print("Classes:", model.classes_)


if __name__ == "__main__":
    main()
