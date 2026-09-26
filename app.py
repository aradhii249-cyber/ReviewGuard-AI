from flask import Flask, render_template, request
import joblib
import os
import re

app = Flask(__name__)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "fake_review_model.pkl"
)

model = joblib.load(MODEL_PATH)


# -------------------------------------------------
# ANALYZE REVIEW
# -------------------------------------------------

def analyze_review(review):

    prediction = model.predict([review])[0]

    if prediction == "deceptive":
        result = "LIKELY DECEPTIVE"
        result_type = "suspicious"
    else:
        result = "LIKELY TRUTHFUL"
        result_type = "truthful"

    words = re.findall(r"\b\w+\b", review)

    word_count = len(words)
    character_count = len(review)

    sentence_count = len(
        re.findall(r"[.!?]+", review)
    )

    return {
        "result": result,
        "result_type": result_type,
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count
    }


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# -------------------------------------------------
# TEXT REVIEW ANALYSIS
# -------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    review = request.form.get(
        "review",
        ""
    ).strip()

    if not review:

        return render_template(
            "index.html",
            error="Please enter a review."
        )

    analysis = analyze_review(
        review
    )

    return render_template(
        "index.html",
        review=review,
        analysis=analysis
    )


# -------------------------------------------------
# OCR ROUTE
# -------------------------------------------------

@app.route("/ocr", methods=["POST"])
def ocr():

    # Check whether a file was uploaded

    if "screenshot" not in request.files:

        return render_template(
            "index.html",
            error="Please upload a screenshot."
        )

    screenshot = request.files["screenshot"]

    # Check filename

    if screenshot.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )

    # For now, confirm that the image was received.
    # OCR will be connected after the upload section
    # is confirmed to be working.

    return render_template(
        "index.html",
        ocr_message="Screenshot uploaded successfully!",
        filename=screenshot.filename
    )


# -------------------------------------------------
# START FLASK
# -------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)