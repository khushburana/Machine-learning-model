import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
data = pd.read_csv("spam_dataset_realistic_10000.csv", encoding="latin-1")
data = data[["Email Text", "Label"]]

# Visualize spam vs not_spam
plt.figure(figsize=(8, 5))
plot = data["Label"].value_counts().plot(
    kind="bar",
    color=["#4CAF50", "#F44336"],
    edgecolor="black",
    width=0.7
)
for bar in plot.patches:
    plot.annotate(f'{bar.get_height()}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                  ha='center', va='bottom')
plt.title("Spam vs Not Spam Email Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Not Spam", "Spam"])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Convert text labels to numeric
data["Label"] = data["Label"].map({"not_spam": 0, "spam": 1})
data = data.dropna()

# Train-test split
X = data["Email Text"]
y = data["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
predictions = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Save model and vectorizer
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Load for real-time prediction
def detect_spam(email_text):
    vectorized_input = vectorizer.transform([email_text])
    result = model.predict(vectorized_input)
    return "Spam!" if result[0] == 1 else "Not Spam!"

# Example (Uncomment to test)
# print(detect_spam("Claim your free prize now!"))
# print(detect_spam("Let’s meet for the project discussion."))
