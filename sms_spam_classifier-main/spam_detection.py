# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data/spam.csv', encoding='ISO-8859-1')

# Clean the dataset (drop unnecessary columns)
data = data[['v1', 'v2']]  # 'v1' is the label, 'v2' is the message
data.columns = ['label', 'message']

# Encode the labels (spam = 1, ham = 0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Split the dataset into features (X) and target (y)
X = data['message']
y = data['label']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use TF-IDF vectorizer to convert text data into numerical data
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Naive Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

# Logistic Regression Classifier
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_tfidf, y_train)
y_pred_log_reg = log_reg_model.predict(X_test_tfidf)

# Support Vector Machine Classifier
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)

# Evaluate the models
print("Naive Bayes Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb)}")
print(classification_report(y_test, y_pred_nb))

print("Logistic Regression Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg)}")
print(classification_report(y_test, y_pred_log_reg))

print("Support Vector Machine Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(classification_report(y_test, y_pred_svm))

# Plot confusion matrix for SVM
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.show()
