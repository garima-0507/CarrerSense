from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ---- Same constants as training ----
SKILL_KEYWORDS = [
    "python", "ml", "pandas", "numpy",
    "sql", "excel", "statistics", "data",
    "html", "css", "javascript", "react",
    "java", "android", "kotlin",
    "networking", "linux", "security", "firewall",
    "aws", "cloud", "docker", "azure"
]

INTEREST_KEYS = ["ml", "web", "data", "android", "cyber", "cloud"]

# Load trained model
with open("model/career_model.pkl", "rb") as f:
    model = pickle.load(f)

def build_features(skills_str: str, interest_str: str) -> list:
    skills_list = [s.strip().lower() for s in skills_str.split(",") if s.strip()]

    skill_vector = [1 if kw in skills_list else 0 for kw in SKILL_KEYWORDS]

    interest_str = interest_str.strip().lower()
    interest_vector = [1 if interest_str == key else 0 for key in INTEREST_KEYS]

    return skill_vector + interest_vector


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form.get("name")
        education = request.form.get("education")
        skills = request.form.get("skills")
        interest = request.form.get("interests")

        features = build_features(skills, interest)
        features = np.array(features).reshape(1, -1)

        # Top 3 role suggestions
        probs = model.predict_proba(features)[0]
        classes = model.classes_
        top_idx = np.argsort(probs)[::-1][:3]

        recommendations = []
        for i in top_idx:
            recommendations.append({
                "role": classes[i],
                "score": round(float(probs[i] * 100), 1)
            })

        return render_template(
            "result.html",
            name=name,
            education=education,
            skills=skills,
            interest=interest,
            recommendations=recommendations
        )

    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
