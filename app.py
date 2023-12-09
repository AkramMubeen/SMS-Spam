import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Function to transform and preprocess the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set page title and favicon
st.set_page_config(
    page_title="Spam Classifier",
    page_icon=":email:",
)

# Set app header with custom styles
st.title("📧 Email/SMS Spam Classifier")
st.markdown(
    """
    <style>
        div.stTextInput > div > div > input {
            border-radius: 10px;
            border: 2px solid #3498db;
            padding: 10px;
            font-size: 16px;
            box-shadow: 5px 5px 10px #888888;
        }
        div.stButton > button:first-child {
            background-color: #2ecc71;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
        }
        div.stButton > button:hover {
            background-color: #27ae60;
        }
        div.stHeader > p {
            font-size: 24px;
            font-weight: bold;
            color: #2ecc71;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# User input for SMS
input_sms = st.text_area("Enter the message")

# Button to predict
if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display result
    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✉️ Not Spam")
