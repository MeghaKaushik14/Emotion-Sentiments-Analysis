import numpy as np
import pandas as pd
import streamlit as st
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Streamlit app title
st.title('Emotion Sentiments Analysis')

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Combined_Data.csv")

df = load_data()

# Clean the data: drop missing values and reset index
df_clean = df.dropna().reset_index(drop=True)

# Add a serial number ('SNO') column
df_clean['SNO'] = df_clean.index + 1

# Ensure 'statement' is of string type
df_clean['statement'] = df_clean['statement'].astype(str)

# Preprocess function using PorterStemmer
ps = PorterStemmer()

def preprocess_data(statement):
    # Remove non-alphabet characters, convert to lowercase, and apply stemming
    cleaned_data = re.sub(r'[^a-zA-Z\s]', ' ', statement)
    lower_data = cleaned_data.lower()
    splitted_data = lower_data.split()
    stemmed_data = [ps.stem(word) for word in splitted_data]
    return ' '.join(stemmed_data)

# Apply preprocessing to 'statement' column
df_clean['statement'] = df_clean['statement'].apply(preprocess_data)

# Features and target variable
x = df_clean['statement'].values
y = df_clean['status'].values

# Vectorize the text data using TF-IDF with a reduced number of features
vector = TfidfVectorizer(stop_words='english', max_features=3000)
x = vector.fit_transform(x)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Cache the model training process to improve speed
@st.cache_resource
def train_model(_x_train, _y_train):
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(_x_train, _y_train)
    return model

# Call the function with the updated signature
model = train_model(x_train, y_train)


# Predict on test set
y_test_pred = model.predict(x_test)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)


# Streamlit UI for emotion sentiment prediction
st.header('Emotion Sentiments Prediction')
input_text = st.text_input('Enter your emotional text')

# Prediction function
def prediction(input_text):
    preprocessed_input = preprocess_data(input_text)
    input_data = vector.transform([preprocessed_input])
    pred = model.predict(input_data)
    return pred[0]

# Button for prediction
if st.button('Predict'):
    if input_text:
        pred = prediction(input_text)
        st.write(f"Predicted Sentiment: {pred}")
    else:
        st.write("Please enter some text to analyze.")
# provide correct provide for sentiments analyze
