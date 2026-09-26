import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --------------------------------
# STEP 1: Read the broken CSV safely
# --------------------------------

filename = "deceptive-opinion.csv"

with open(filename, "r", encoding="utf-8", errors="replace") as file:
    lines = file.read().splitlines()

records = []
current = None

for line in lines:

    # New review starts with truthful or deceptive
    if re.match(r"^(truthful|deceptive),", line):

        if current is not None:
            records.append(current)

        parts = line.split(",", 4)

        if len(parts) == 5:
            current = parts
        else:
            current = None

    else:
        # Continuation of the review text
        if current is not None:
            current[4] += "\n" + line

# Add last review
if current is not None:
    records.append(current)


# --------------------------------
# STEP 2: Create DataFrame
# --------------------------------

df = pd.DataFrame(
    records,
    columns=["deceptive", "hotel", "polarity", "source", "text"]
)

# Remove accidental header row
df = df[df["deceptive"].isin(["truthful", "deceptive"])]

# Clean text
df["text"] = df["text"].astype(str).str.strip()

# Remove unnecessary quotation marks
df["text"] = df["text"].apply(
    lambda x: x[1:-1]
    if len(x) >= 2 and x.startswith('"') and x.endswith('"')
    else x
)

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["deceptive"].value_counts())


# --------------------------------
# STEP 3: Input and target
# --------------------------------

X = df["text"]
y = df["deceptive"]


# --------------------------------
# STEP 4: Train/Test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# STEP 5: TF-IDF + SVM
# --------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000
        )
    ),

    (
        "classifier",
        LinearSVC()
    )
])


# --------------------------------
# STEP 6: Train
# --------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)


# --------------------------------
# STEP 7: Test
# --------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------
# STEP 8: Save model
# --------------------------------

joblib.dump(model, "fake_review_model.pkl")

print("\nModel saved successfully!")
print("File: fake_review_model.pkl")


# --------------------------------
# STEP 9: Test our own reviews
# --------------------------------

reviews = [

    "The room was clean and comfortable. "
    "The staff were friendly and helpful. "
    "The breakfast was good.",

    "BEST HOTEL EVER!!! AMAZING!!! "
    "Everyone MUST stay here!!! "
    "This is the best hotel in the world!!!"
]

print("\nSample Predictions:")

for review in reviews:

    prediction = model.predict([review])[0]

    print("\nReview:")
    print(review)

    print("Prediction:", prediction)