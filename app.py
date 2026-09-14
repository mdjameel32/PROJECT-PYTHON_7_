import streamlit as st
import pickle
import re
import string

# 1. Page Configuration
st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨", layout="centered")

# 2. Load Models (Cached for performance so they only load once)
@st.cache_resource
def load_assets():
    model = pickle.load(open('disaster_tweet_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except FileNotFoundError:
    st.error("Error: Model files (`disaster_tweet_model.pkl` or `tfidf_vectorizer.pkl`) not found. Please ensure they are in the same directory.")
    st.stop()

# 3. Text Preprocessing Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 4. UI Layout
st.title("🚨 Disaster Tweet Classifier")
st.write("Enter a tweet below to predict if it describes a real-world disaster.")

# Form container for user input
with st.form(key='tweet_form'):
    tweet_input = st.text_area("Tweet Text", placeholder="Type or paste tweet here...", height=100)
    submit_button = st.form_submit_button(label='Analyze Tweet')

# 5. Prediction Logic
if submit_button:
    if tweet_input.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        # Preprocess and predict
        cleaned = clean_text(tweet_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]
        
        # Format results
        prediction = 'Real Disaster' if pred == 1 else 'Not a Disaster'
        confidence = round(proba * 100, 2) if pred == 1 else round((1 - proba) * 100, 2)
        
        # Display Results
        st.subheader("Analysis Result")
        if pred == 1:
            st.error(f"**Prediction:** {prediction} (Confidence: {confidence}%)")
        else:
            st.success(f"**Prediction:** {prediction} (Confidence: {confidence}%)")
