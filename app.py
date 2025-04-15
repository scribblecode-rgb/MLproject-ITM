from flask import Flask, render_template, request
import pickle


# Load model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']
    input_vector = vectorizer.transform([email_text])
    prediction = model.predict(input_vector)[0]

    if prediction == 0:
        result = "Ham (Not Spam)"
        color = "success"
    else:
        result = "Spam"
        color = "danger"

    return render_template('index.html', prediction=result, color=color, email=email_text)

if __name__ == '__main__':
    app.run(debug=True)
