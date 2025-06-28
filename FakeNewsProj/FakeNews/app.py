from flask import Flask, render_template, request, session
import pickle

# Load model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

app = Flask(__name__)
app.secret_key = 'your_secret_key'  

@app.route('/')
def index():
    # Clear session history on first page load
    session['history'] = []
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news'].strip()

    if not input_text or len(input_text.split()) < 3:
        label = "⚠️ Please enter a longer news sentence."
    else:
        transformed = vectorizer.transform([input_text])
        pred_label = model.predict(transformed)[0]
        label = "✅ Real Disaster News" if pred_label == 1 else "❌ Fake Disaster News"

        # Store to session history
        if 'history' not in session:
            session['history'] = []
        session['history'].append({"text": input_text, "label": label})
        session.modified = True

    return render_template('index.html', prediction=label, input_text=input_text, history=session.get('history', []))

if __name__ == '__main__':
    app.run(debug=True)
