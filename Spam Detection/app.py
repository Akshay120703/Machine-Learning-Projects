from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Load the model and TF-IDF vectorizer
model = joblib.load('spam_detection_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Spam_Detection.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    message_tfidf = tfidf.transform([message])
    prediction = model.predict(message_tfidf)
    result = 'Spam' if prediction[0] == 1 else 'Not Spam'
    return render_template('Spam_Detection.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
