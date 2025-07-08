📰 Fake News Detection App
A powerful fake news classifier built with BERT, enhanced by temperature scaling for calibrated confidence, and further supported with explanations from Google Gemini.

Deployed at: https://fakenews-detection-app.streamlit.app/
Run locally at: http://localhost:8501

🚀 Features
1. Detects fake vs real news headlines
2. Uses BERT for deep understanding of text
3. Applies temperature scaling to improve confidence calibration
4. Explains results using Gemini LLM 
5. Confidence adjustment if explanation shows uncertainty
6. Polished UI with custom CSS & background
7. Tech Stack
- transformers, datasets, accelerate, torch
- scikit-learn, pandas, numpy
- streamlit, python-dotenv, huggingface_hub
- Google Gemini API (via REST)

📦 Installation

1. Clone the repo
git clone ...
cd fake-news-detector

2. Install dependencies
pip install -r requirements.txt

3. Set up environment
Create a .env file and add your Google Gemini API key
GEMINI_API_KEY=your_api_key_here

4. Run the app locally
streamlit run app.py


🧪 Model Training Pipeline

Load Data: Real (True.csv) and fake (Fake.csv) news from Google Drive.

Preprocess: Combine title and text, then label and shuffle.

Split Dataset: 80% train / 10% val / 10% test.

Tokenize using BERT tokenizer.

Train BERT using Hugging Face Trainer.

Temperature Scaling: Improve calibration on validation set.

Evaluate: Accuracy, precision, recall, F1, and confusion matrix.

Save Model: Export model, tokenizer, and temperature for use in Streamlit.

🖥️ Streamlit App Features

Paste a headline or short article

See prediction (Fake or Real) + confidence %

Get a short explanation from Gemini LLM

Automatically lowers confidence if LLM expresses uncertainty (e.g. speculative, rumor)

🤖 Example Predictions

"COVID-19 vaccine causes magnetic arms in humans."
→ Fake (Confidence: 99.9%) + Explanation

"NASA’s Perseverance rover begins mission on Mars to collect rock samples."
→ Real (Confidence: 96.4%) + Explanation

📁 Model on Hugging Face Hub
Model and temperature file hosted at:
🔗 https://huggingface.co/aashi219/fake-news-bert

🎨 UI & Styling

Responsive design
Light-themed layout with news-like styling
Custom CSS and background image
