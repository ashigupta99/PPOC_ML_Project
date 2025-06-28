import pandas as pd
import pickle

# Load model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

# Load test data
test_df = pd.read_csv("test.csv")
texts = test_df['text'].fillna("")

# Transform and predict
X_test = vectorizer.transform(texts)
predictions = model.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    "id": test_df["id"],
    "label": predictions
})

# Save to CSV
submission.to_csv("my_submission.csv", index=False)
print("✅ Prediction saved to my_submission.csv")
