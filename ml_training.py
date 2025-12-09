import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os

# ---- SKILL & INTEREST KEYWORDS ----
SKILL_KEYWORDS = [
    "python", "ml", "pandas", "numpy",
    "sql", "excel", "statistics", "data",
    "html", "css", "javascript", "react",
    "java", "android", "kotlin",
    "networking", "linux", "security", "firewall",
    "aws", "cloud", "docker", "azure"
]

INTEREST_KEYS = ["ml", "web", "data", "android", "cyber", "cloud"]


# ---- FEATURE BUILDING FUNCTION ----
def build_features(skills_str, interest_str):
    """
    Converts user’s skills + interest into numerical ML features.
    """
    # Convert string to list and lowercase
    skills_list = [s.strip().lower() for s in skills_str.split(",")]

    # Skill vector (binary)
    skill_vector = [1 if kw in skills_list else 0 for kw in SKILL_KEYWORDS]

    # Interest one-hot encoding
    interest_str = interest_str.strip().lower()
    interest_vector = [1 if interest_str == key else 0 for key in INTEREST_KEYS]

    return skill_vector + interest_vector


# ---- MAIN TRAINING PROCESS ----
def main():
    print("Loading dataset...")

    df = pd.read_csv("dataset/careers.csv")

    X = []
    y = []

    print("Processing rows...")
    for _, row in df.iterrows():
        features = build_features(row["skills"], row["interests"])
        X.append(features)
        y.append(row["role"])

    X = np.array(X)

    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X, y)

    # Create model folder if needed
    os.makedirs("model", exist_ok=True)

    # Save trained model
    with open("model/career_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\n🎉 Model trained successfully!")
    print("📁 Saved as: model/career_model.pkl")
    print("📌 Classes:", model.classes_)


# Run training
if __name__ == "__main__":
    main()
