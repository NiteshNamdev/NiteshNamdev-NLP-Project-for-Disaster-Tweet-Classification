import streamlit
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the logistic regression model and TF-IDF vectorizer
# Make sure these files are in the same directory as this script, or provide their full paths
with open('log_reg_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Title for the Streamlit web app
streamlit.title("Disaster Tweet Classification")
streamlit.write("This app classifies whether a tweet is about a disaster or not.")

# Text input for the user's tweet
user_input = streamlit.text_area("Enter the tweet text:", "")

# Check if there's text input, then predict
if user_input:
    # Preprocess and transform the input text
    tweet_vector = vectorizer.transform([user_input])
    
    # Get the prediction and probability
    prediction = model.predict(tweet_vector)
    probability = model.predict_proba(tweet_vector)[0][1]
    
    # Display the result
    result = "Disaster" if prediction[0] == 1 else "Non-Disaster"
    streamlit.write(f"**Classification:** {result}")
    streamlit.write(f"**Confidence Score:** {probability:.2f}")

    # Optional: Display an alert-style message based on prediction
    if result == "Disaster":
        streamlit.warning("This tweet is classified as related to a disaster.")
    else:
        streamlit.success("This tweet is classified as not related to a disaster.")
