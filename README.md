🛡️ PhishGuard

PhishGuard is a robust machine learning-based phishing detection system that utilizes BERT (Bidirectional Encoder Representations from Transformers) to classify emails as Safe or Phishing. It features a fine-tuned transformer model on real-world email datasets, integrated with a modern React frontend and Python backend (PyTorch).

🔍 Features

✅ Detects phishing emails using fine-tuned BERT

📊 Visualizes evaluation metrics (Accuracy, Precision, Recall, F1)

🔄 Supports training resumption via checkpoints

🧠 Extracts BERT embeddings + trains SVM for alternate classification

🎯 Real-time email prediction through frontend

⚡ CUDA acceleration supported for training/prediction

🧪 Dataset Format

The dataset is a CSV file with the following structure:

Email Text	Email Type
"Click here to update your info"	Phishing Email
"Your report is attached"	Safe Email

Mapped labels:

Safe Email → 0

Phishing Email → 1

🧠 Model Training

We use a pre-trained bert-base-uncased model and fine-tune it using the HuggingFace Transformers library.

🔧 Training Setup

Tokenization using BertTokenizer

Classification model: BertForSequenceClassification

Custom Dataset and DataLoader

Optimizer: AdamW

Scheduler: get_linear_schedule_with_warmup

Loss: CrossEntropyLoss

Checkpointing: Saved periodically (bert_fine_tuned_checkpoint.pth)

✍️ To Train:

cd backend/

run bert.ipynb file

Intermediate checkpoints are saved every few batches to handle interruptions.

📈 Evaluation Metrics

The model is evaluated using:

Accuracy, Precision, Recall, F1 Score, Confusion Matrix (Plotted using Seaborn)

🔮 Email Prediction

Run predictions on new email text using:

from predict import predict_email

email = "Dear user, your account has been compromised. Click here to reset."
prediction = predict_email(email, model, tokenizer)

print("Phishing" if prediction else "Safe")

💻 React Frontend

The frontend is built using React.js and interacts with the Python backend via REST APIs. It allows users to input email text and receive phishing/safe predictions with real-time UI feedback.

To Start Frontend:

cd frontend/react-app/
npm install
npm start

🚀 Deployment
You can deploy the backend using:

Flask / FastAPI (for serving the BERT model)

Docker (optional for containerization)

Integration with EC2 + Jenkins for CI/CD

⚙️ Tech Stack
Frontend: React.js

Backend: Python, PyTorch, Transformers

ML Model: bert-base-uncased (HuggingFace)

Evaluation: Scikit-learn, Matplotlib, Seaborn

Training: GPU-accelerated fine-tuning, SVM classifier on BERT embeddings


