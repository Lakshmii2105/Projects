from flask import Flask, render_template, request
import joblib

from phishing_module import URLPreprocessor, FeatureExtractor

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize classes
pre = URLPreprocessor()
extractor = FeatureExtractor()


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        user_url = request.form["url"]

        # Clean URL
        clean_url = pre.clean_url(user_url)

        # Extract Features
        features = extractor.extract_features([clean_url])

        # Scale Features
        features_scaled = scaler.transform(features)

        # Predict
        result = model.predict(features_scaled)

        # Similarity Detection
        similarity = extractor.similarity_to_known_sites(clean_url)

        # Rule-Based Detection
        if similarity > 0.80 and similarity < 1.0:
            prediction = "⚠️ Phishing Website Detected"

        elif extractor.has_at_symbol(clean_url):
            prediction = "⚠️ Suspicious URL"

        elif extractor.has_ip(clean_url):
            prediction = "⚠️ IP-Based Suspicious Website"

        elif result[0] == 1:
            prediction = "⚠️ Phishing Website Detected"

        else:
            prediction = "✅ Legitimate Website"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)