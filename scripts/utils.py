import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset of symptoms to disease
disease_data_path = '/home/mustar/catkin_ws/src/med_buddy/data/Symptom2Disease.csv'
data1 = pd.read_csv(disease_data_path, encoding='utf-8')  # Ensure proper encoding

# Drop unnecessary columns if they exist
if "Unnamed: 0" in data1.columns:
    data1.drop(columns=["Unnamed: 0"], inplace=True)

# Concise summary of DataFrame
data1.info()

# Check for null values
print(data1.isnull().sum())

# Extracting 'label' and 'text' columns from the 'data1' DataFrame
labels = data1['label']  # Contains the labels or categories associated with the text data
symptoms = data1['text']  # Contains the textual data (e.g., symptoms, sentences) for analysis

# Text Preprocessing
stop_words = set(stopwords.words('english'))  # Set of stopwords in English

# Text preprocessing function
def preprocess_text(text):
    # Handle Unicode text properly
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    # Tokenization: split the text into words
    words = word_tokenize(text.lower())
    # Removing stopwords and non-alphabetic characters
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)  # Join words back into a single string

# Apply preprocessing to symptoms
preprocessed_symptoms = symptoms.apply(preprocess_text)

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1500)  # Convert text data to TF-IDF features
tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_symptoms).toarray()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)

# KNN Model Training
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # KNN classifier with 5 neighbors
knn_classifier.fit(X_train, y_train)  # Train the model

# Predictions
predictions = knn_classifier.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, predictions)  # Calculate accuracy
print('Accuracy: {:.2f}'.format(accuracy))
print(classification_report(y_test, predictions))  # Print classification report

# TF-IDF function for new input
def tfidf(preprocessed_text, preprocessed_symptoms):
    tfidf_vectorizer = TfidfVectorizer(max_features=1500)  # Recreate the vectorizer
    tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_symptoms).toarray()
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text]).toarray()  # Transform new text to TF-IDF
    return text_tfidf

# Get the disease from the symptoms function 
def getDisease(symptom_tfidf):
    predicted_disease = knn_classifier.predict(symptom_tfidf)  # Predict the disease
    if predicted_disease.size > 0:
        return predicted_disease[0]
    else:
        return np.nan  # Return NaN if no prediction is made

# Get the suggestion from the predicted disease function
def getAction(data, predicted_disease):
    action_row = data.loc[data.disease == predicted_disease]  # Find the row with the predicted disease
    if not action_row.empty:
        return action_row['action'].values[0]  # Return the action for the predicted disease
    else:
        return "I'm sorry, I couldn't find any action for the predicted disease. Please visit a doctor for your problem."  # Return a default message

# Check if input contains any recognizable symptoms function
def validate_input(input_text, preprocessed_symptoms):
    # Split the preprocessed input sentence into individual words
    input_words = set(input_text.split())

    # Initialize a set to store all symptom words by splitting and flattening the symptoms list
    all_symptom_words = set(' '.join(preprocessed_symptoms).split())

    # Find the intersection of input words and symptom words
    valid_symptom_words = input_words.intersection(all_symptom_words)

    # Count the number of valid symptom words found
    valid_words_count = len(valid_symptom_words)

    # Check if the input is valid based on the number of valid words
    if valid_words_count == 0:
        return False, 'Not Valid'  # No valid symptoms found
    elif valid_words_count < 3:
        return False, 'Insufficient'  # Not enough valid symptoms found. A minimum of 2 keywords is needed for a more accurate prediction.
    return True, ' '  # Input is valid

