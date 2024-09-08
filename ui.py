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

# Sidebar with model descriptions
st.sidebar.title("Model Descriptions")
st.sidebar.markdown("""
### SVM (Support Vector Machine)
SVM is a supervised learning algorithm which is mainly used for classification problems. It works by finding a hyperplane that best divides a dataset into classes.

### Random Forest
Random Forest is an ensemble method that builds multiple decision trees and merges them together to get a more accurate and stable prediction.

### Naive Bayes
Naive Bayes is a simple but powerful algorithm based on the Bayes theorem. It assumes that the features are independent, which makes it particularly suited for spam filtering.

### Gradient Boosting
Gradient Boosting is a machine learning technique for regression and classification problems, which builds a prediction model in the form of an ensemble of weak prediction models, typically decision trees.
""")

# Streamlit UI for email spam detector
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

# Dynamically set y-axis range
if accuracies:
    min_acc = min([acc for acc in accuracies if acc is not None])
    max_acc = max([acc for acc in accuracies if acc is not None])
    ax.set_ylim(min_acc - 1, max_acc + 1)  # Adding some padding for better visibility

# Display the plot
st.pyplot(fig)
