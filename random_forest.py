import os
import glob
import email
import re
import nltk
from nltk.data import find
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

# Pre-download NLTK resources and handle exceptions
def download_nltk_resources():
    try:
        find('corpora/stopwords.zip', paths=['./nltk_data'])
    except LookupError:
        print("Stopwords not found locally. Downloading...")
        nltk.download('stopwords', download_dir='./nltk_data')

download_nltk_resources()


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.strip()
    return text

def predict_spam(vectorizer, classifier, message):
    # Preprocess the message
    message = preprocess_text(message)
    
    # Transform the message using the vectorizer
    message_tfidf = vectorizer.transform([message])
    
    # Predict using the classifier
    prediction = classifier.predict(message_tfidf)
    
    # Return result
    return "Spam" if prediction[0] == 1 else "Not Spam"

def load_emails_from_folder(folder):
    emails = []
    for filepath in glob.glob(os.path.join(folder, '*')):
        with open(filepath, 'r', encoding='latin1') as f:
            try:
                msg = email.message_from_file(f)
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            emails.append(part.get_payload())
                else:
                    emails.append(msg.get_payload())
            except Exception as e:
                print(f"Error reading email: {e}")
    return emails

def train_random_forest_model(ham_folder, spam_folder):
    ham_emails = load_emails_from_folder(ham_folder)
    spam_emails = load_emails_from_folder(spam_folder)

    ham_emails_clean = [preprocess_text(email) for email in ham_emails if preprocess_text(email).strip() != '']
    spam_emails_clean = [preprocess_text(email) for email in spam_emails if preprocess_text(email).strip() != '']

    ham_labels = [0] * len(ham_emails_clean)
    spam_labels = [1] * len(spam_emails_clean)

    emails = ham_emails_clean + spam_emails_clean
    labels = ham_labels + spam_labels

    if not emails:
        raise ValueError("No valid documents to process")

    X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifier = RandomForestClassifier()
    classifier.fit(X_train_tfidf, y_train)

    return vectorizer, classifier, X_test_tfidf, y_test

def evaluate_model(vectorizer, classifier, X_test_tfidf, y_test):
    predictions = classifier.predict(X_test_tfidf)
    return accuracy_score(y_test, predictions)
