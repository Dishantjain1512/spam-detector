import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

@st.cache_resource
def train_model():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
        sep='\t', header=None, names=['label', 'message']
    )
    data['label_num'] = data.label.map({'ham': 0, 'spam': 1})
    X_train, _, y_train, _ = train_test_split(
        data.message, data.label_num, test_size=0.2, random_state=1
    )
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

model, vectorizer = train_model()

st.title("Spam Email Detector")
st.write("Write any message and check if it’s spam or not.")

user_input = st.text_area("Type your message here...")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        message_vec = vectorizer.transform([user_input])
        prediction = model.predict(message_vec)[0]
        if prediction == 1:
            st.error("This message is SPAM.")
        else:
            st.success("This message is NOT spam.")
