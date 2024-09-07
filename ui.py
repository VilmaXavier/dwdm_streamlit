import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from classifier import train_model, predict_spam
from random_forest import train_random_forest_model
from naive_bayes import train_naive_bayes_model
from gbm import train_gbm_model

# Define a function to select and train the model based on the chosen algorithm
def load_model(algorithm, ham_folder, spam_folder):
    if algorithm == 'SVM':
        vectorizer, classifier, X_test_tfidf, y_test = train_model(ham_folder, spam_folder)
    elif algorithm == 'Random Forest':
        vectorizer, classifier, X_test_tfidf, y_test = train_random_forest_model(ham_folder, spam_folder)
    elif algorithm == 'Naive Bayes':
        vectorizer, classifier, X_test_tfidf, y_test = train_naive_bayes_model(ham_folder, spam_folder)
    elif algorithm == 'Gradient Boosting':
        vectorizer, classifier, X_test_tfidf, y_test = train_gbm_model(ham_folder, spam_folder)
    else:
        raise ValueError("Invalid algorithm selected")
    
    # Calculate accuracy for the model
    accuracy = evaluate_model(vectorizer, classifier, X_test_tfidf, y_test)
    return vectorizer, classifier, accuracy

def evaluate_model(vectorizer, classifier, X_test_tfidf, y_test):
    from sklearn.metrics import accuracy_score
    predictions = classifier.predict(X_test_tfidf)
    return accuracy_score(y_test, predictions) * 100  # Convert to percentage

# Streamlit UI
st.title("Email Spam Detector")

# Dropdown to select the algorithm
algorithm = st.selectbox("Select Algorithm", ["SVM", "Random Forest", "Naive Bayes", "Gradient Boosting"])

# Load the model and vectorizer based on the selected algorithm
ham_folder = 'datasets/ham'
spam_folder = 'datasets/spam'
vectorizer, classifier, accuracy = load_model(algorithm, ham_folder, spam_folder)

# Input fields for email address and message
email_address = st.text_input("Email Address:")
message_text = st.text_area("Message:")

# Button to make prediction
if st.button("Predict"):
    if not message_text.strip():
        st.warning("Please enter a message to classify.")
    else:
        result = predict_spam(vectorizer, classifier, message_text)
        st.write(result)

# Display training metrics
st.subheader("Model Performance Comparison")

# Collecting performance data for visualization
algorithms = ["SVM", "Random Forest", "Naive Bayes", "Gradient Boosting"]
accuracies = []

# Collecting accuracy data
for algo in algorithms:
    try:
        _, _, accuracy = load_model(algo, ham_folder, spam_folder)
        accuracies.append(accuracy)
    except Exception as e:
        st.error(f"Error loading model for {algo}: {e}")
        accuracies.append(None)

# Print accuracies for debugging
st.write("Accuracies:", accuracies)

# Plotting the accuracy comparison
fig, ax = plt.subplots()
sns.barplot(x=algorithms, y=accuracies, ax=ax, palette="viridis")
ax.set_xlabel('Algorithm')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Comparison of Model Accuracies')

# Set y-axis range from 95.0 to 99.0
ax.set_ylim(95.0, 99.0)

# Display the plot
st.pyplot(fig)
