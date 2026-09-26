import joblib

# Load trained model
model = joblib.load("fake_review_model.pkl")

print("===================================")
print("       FAKE REVIEW DETECTOR")
print("===================================")

while True:

    review = input("\nEnter a review (type 'exit' to stop): ")

    if review.lower() == "exit":
        print("Program stopped.")
        break

    prediction = model.predict([review])[0]

    print("\nPrediction:", prediction.upper())