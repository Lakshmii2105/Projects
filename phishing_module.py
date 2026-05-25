import pandas as pd
import re
import numpy as np
import joblib

from difflib import SequenceMatcher
from urllib.parse import urlparse

# =========================================
# MODULE 1: DATA COLLECTION & PREPROCESSING
# =========================================

class URLPreprocessor:

    def __init__(self):

        self.data = [

            # Legitimate URLs
            {"url": "https://www.google.com", "label": 0},
            {"url": "https://www.amazon.in", "label": 0},
            {"url": "https://www.facebook.com", "label": 0},
            {"url": "https://www.youtube.com", "label": 0},
            {"url": "https://www.wikipedia.org", "label": 0},
            {"url": "https://www.instagram.com", "label": 0},
            {"url": "https://www.microsoft.com", "label": 0},
            {"url": "https://www.github.com", "label": 0},

            # Phishing URLs
            {"url": "https://secure-login-bank.xyz", "label": 1},
            {"url": "http://192.168.1.1/login", "label": 1},
            {"url": "https://verify-paypal-account-alert.ru", "label": 1},
            {"url": "https://update-bank-info-secure.com", "label": 1},
            {"url": "https://login-account-verify-security.net", "label": 1},
            {"url": "https://googgle-login-security.com", "label": 1},
            {"url": "http://google.com@fake-login.xyz", "label": 1},
            {"url": "https://faceboook-security-alert.com", "label": 1}
        ]

    def load_dataset(self):
        return pd.DataFrame(self.data)

    def clean_url(self, url):

        url = url.lower()

        if not url.startswith("http"):
            url = "https://" + url

        return url

    def preprocess(self):

        data = self.load_dataset()

        data['url'] = data['url'].apply(self.clean_url)

        print("✅ Dataset Loaded & Cleaned\n")

        return data


# =========================================
# MODULE 2: FEATURE EXTRACTION
# =========================================

class FeatureExtractor:

    # URL Length
    def url_length(self, url):
        return len(url)

    # Count dots
    def count_dots(self, url):
        return url.count('.')

    # Count special characters
    def count_special_chars(self, url):
        return len(re.findall(r"[-@?=_%]", url))

    # Detect IP Address
    def has_ip(self, url):
        return 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0

    # Suspicious Keywords
    def suspicious_words(self, url):

        keywords = [
            "login",
            "verify",
            "update",
            "secure",
            "bank",
            "alert",
            "account",
            "password",
            "signin"
        ]

        return sum(word in url for word in keywords)

    # Check HTTPS
    def has_https(self, url):
        return 1 if url.startswith("https") else 0

    # Count Hyphens
    def count_hyphens(self, url):
        return url.count('-')

    # Count Subdomains
    def count_subdomains(self, url):

        domain = urlparse(url).netloc

        return domain.count('.')

    # Detect @ Symbol
    def has_at_symbol(self, url):
        return 1 if '@' in url else 0

    # Extract Domain
    def extract_domain(self, url):

        domain = urlparse(url).netloc

        domain = domain.replace("www.", "")

        return domain

    # Similarity Detection
    def similarity_to_known_sites(self, url):

        legit_sites = [
            "google.com",
            "amazon.in",
            "facebook.com",
            "youtube.com",
            "instagram.com",
            "microsoft.com",
            "github.com"
        ]

        domain = self.extract_domain(url)

        max_similarity = 0

        for site in legit_sites:

            ratio = SequenceMatcher(None, domain, site).ratio()

            max_similarity = max(max_similarity, ratio)

        return max_similarity

    # Feature Extraction
    def extract_features(self, urls):

        features = []

        for url in urls:

            features.append([

                self.url_length(url),
                self.count_dots(url),
                self.count_special_chars(url),
                self.has_ip(url),
                self.suspicious_words(url),
                self.has_https(url),
                self.count_hyphens(url),
                self.count_subdomains(url),
                self.has_at_symbol(url),
                self.similarity_to_known_sites(url)

            ])

        return np.array(features)


# =========================================
# MODULE 3: DATA PREPROCESSING
# =========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(X, y):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        random_state=42
    )

    print("✅ Data Scaled & Split\n")

    return X_train, X_test, y_train, y_test, scaler


# =========================================
# MODULE 4: MODEL TRAINING
# =========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("✅ Model Trained Successfully\n")

    return model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("===== MODEL EVALUATION =====\n")

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))


# =========================================
# MAIN PROGRAM
# =========================================

if __name__ == "__main__":

    print("\n===== PHISHING URL DETECTION SYSTEM =====\n")

    # Step 1: Preprocessing
    pre = URLPreprocessor()

    data = pre.preprocess()

    # Step 2: Feature Extraction
    extractor = FeatureExtractor()

    X = extractor.extract_features(data['url'])

    y = data['label']

    # Step 3: Scaling + Split
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Step 4: Train Model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate Model
    evaluate_model(model, X_test, y_test)

    # Step 6: Save Model and Scaler
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("\n✅ Model and Scaler Saved Successfully")

    # =========================================
    # REAL-TIME DETECTION
    # =========================================

    print("\n===== REAL-TIME URL DETECTION =====")

    while True:

        user_url = input("\nEnter URL (or 'exit' to stop): ")

        if user_url.lower() == "exit":
            break

        # Clean URL
        clean_url = pre.clean_url(user_url)

        # Extract Features
        features = extractor.extract_features([clean_url])

        # Scale Features
        features_scaled = scaler.transform(features)

        # ML Prediction
        result = model.predict(features_scaled)

        # Similarity Score
        similarity = extractor.similarity_to_known_sites(clean_url)

        # Domain Extraction
        domain = extractor.extract_domain(clean_url)

        # Rule-Based Checks

        if similarity > 0.80 and similarity < 1.0:

            print("\n⚠️ Phishing Detected (Typosquatting Attack)")

        elif extractor.has_at_symbol(clean_url):

            print("\n⚠️ Suspicious '@' Symbol Found")

        elif extractor.has_ip(clean_url):

            print("\n⚠️ IP-Based Suspicious URL")

        elif (
            extractor.suspicious_words(clean_url) >= 2
            and extractor.count_hyphens(clean_url) >= 2
        ):

            print("\n⚠️ Highly Suspicious URL Detected")

        elif result[0] == 1:

            print("\n⚠️ Phishing / Deceptive Website Detected")

        else:

            print("\n✅ Legitimate Website")

        # Extra Information
        print("\n===== URL ANALYSIS =====")

        print("Domain:", domain)

        print("Similarity Score:", round(similarity, 2))

        print("URL Length:", extractor.url_length(clean_url))

        print("Special Characters:",
              extractor.count_special_chars(clean_url))

        print("Suspicious Keywords:",
              extractor.suspicious_words(clean_url))