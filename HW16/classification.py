import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load your text data and labels from the CSV file
data = pd.read_csv("IMDB Dataset.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["review"], data["sentiment"], test_size=0.2, random_state=42)

# Load the SpaCy NLP pipeline
nlp = spacy.load("en_core_web_sm")

# Tokenize and preprocess the text data using SpaCy (you may need additional preprocessing steps)
X_train_processed = [nlp(text) for text in X_train]
X_test_processed = [nlp(text) for text in X_test]

# Feature extraction (you can use TF-IDF or other methods)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform([str(doc) for doc in X_train_processed])
X_test_tfidf = tfidf_vectorizer.transform([str(doc) for doc in X_test_processed])

# Choose and train a classification model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report for more detailed evaluation metrics
print(classification_report(y_test, y_pred))
