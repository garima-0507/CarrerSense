from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

SKILL_KEYWORDS = [
    "python", "ml", "pandas", "numpy",
    "sql", "excel", "statistics", "data",
    "html", "css", "javascript", "react",
    "java", "android", "kotlin",
    "networking", "linux", "security", "firewall",
    "aws", "cloud", "docker", "azure"
]

INTEREST_KEYS = ["ml", "web", "data", "android", "cyber", "cloud"]

with open("model/career_model.pkl", "rb") as f:
    model = pickle.load(f)

# Career Information Data
ROLE_DETAILS = {
    "Machine Learning Engineer": {
        "description": "ML Engineers build models and AI systems.",
        "skills": ["Python", "ML", "Statistics", "TensorFlow", "Pandas"],
        "salary": "5–15 LPA (India) / $90k–150k (US)",
        "roadmap": [
            "Learn Python",
            "Master ML Algorithms",
            "Learn TensorFlow/PyTorch",
            "Build ML Projects"
        ]
    },
    "Data Scientist": {
    "description": "Data Scientists analyze data to find insights and build predictive models.",
    "skills": ["Python", "Statistics", "Machine Learning", "SQL", "Data Visualization"],
    "salary": "6–20 LPA (India) / $100k–160k (US)",
    "roadmap": [
        "Learn Python & SQL",
        "Understand Statistics & Probability",
        "Learn Machine Learning",
        "Work on Data Analysis & Visualization Projects"
        ]
    },

    "Frontend Developer": {
        "description": "Frontend developers create website interfaces.",
        "skills": ["HTML", "CSS", "JavaScript", "React"],
        "salary": "3–12 LPA (India) / $70k–130k (US)",
        "roadmap": [
            "Learn HTML/CSS",
            "JavaScript Basics",
            "Build Web Projects",
            "Learn React"
        ]
    },
    "Data Analyst": {
        "description": "Data analysts analyze business data.",
        "skills": ["SQL", "Excel", "Python", "Power BI"],
        "salary": "3–10 LPA (India) / $60k–110k (US)",
        "roadmap": [
            "Learn Excel",
            "Master SQL",
            "Python for Data",
            "Dashboard building"
        ]
    },
    "Cybersecurity Analyst": {
        "description": "Cybersecurity analysts protect networks.",
        "skills": ["Networking", "Linux", "Firewalls"],
        "salary": "4–14 LPA (India)",
        "roadmap": [
            "Networking basics",
            "Linux commands",
            "Security tools"
        ]
    }
}


def build_features(skills_str, interest_str):
    skills_list = [s.strip().lower() for s in skills_str.split(",")]
    skill_vector = [1 if kw in skills_list else 0 for kw in SKILL_KEYWORDS]

    interest_vector = [1 if interest_str == key else 0 for key in INTEREST_KEYS]

    return skill_vector + interest_vector


# ROUTES
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        name = request.form.get("name")
        skills = request.form.get("skills")
        interest = request.form.get("interests")

        features = np.array(build_features(skills, interest)).reshape(1, -1)
        probs = model.predict_proba(features)[0]
        classes = model.classes_

        top_idx = np.argsort(probs)[::-1][:3]

        recommendations = [
            {"role": classes[i], "score": round(probs[i] * 100, 2)}
            for i in top_idx
        ]

        return render_template("result.html", name=name, recommendations=recommendations)

    return render_template("test.html")


@app.route("/career/<role>")
def career_details(role):
    role = role.replace("%20", " ")
    details = ROLE_DETAILS.get(role)

    if not details:
        return "Career details not found"

    return render_template("career_details.html", role=role, details=details)


@app.route("/careers")
def all_careers():
    return render_template("careers.html", roles=ROLE_DETAILS)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
