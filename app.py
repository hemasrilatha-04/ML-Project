from flask import Flask, render_template, request
import re
import nltk
import pickle
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

# SAME cleaning as training
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z ]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['job_text']
    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    prob = model.predict_proba(vector)[0]
    confidence = max(prob) * 100

    if prediction == 1:
        result = f"⚠ Fake Job ({confidence:.2f}%)"
    else:
        result = f"✅ Real Job ({confidence:.2f}%)"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)