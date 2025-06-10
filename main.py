import streamlit as st 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense,Embedding,SimpleRNN
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
import tensorflow as tf

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
model = load_model('simpleRNN.h5')
word_index = imdb.get_word_index()
reverse_word_review = {value:key for key,value in word_index.items()}
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write("Enter a movie review below and get the predicted sentiment (positive or negative).")
user = st.text_area("📝 Type your review here:")
if(st.button('Predict')):
    preprocessed_input = preprocess_text(user)
    prediction = model.predict(preprocessed_input)
    sentiment = ' 🟢 Positive' if prediction[0][0] > 0.5 else ' 🔴 Negative'

    st.markdown(f"### Sentiment: {sentiment}")
    st.write("Confidence Score: ",prediction)
else:
    st.write("please write a movie review to continue!")
