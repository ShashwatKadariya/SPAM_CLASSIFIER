from pipeline import preprocess, model

def classify_email(text, model):
    tokens = preprocess(text)
    prediction = model.predict(tokens)
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    print(f"Email: \"{text}\"\nPrediction: {label}")


sample_email = "Congratulations! You’ve won a free vacation to the Bahamas. Click here to claim now."
classify_email(sample_email, model)

# Try another one
sample_email_2 = "Hi team, just a reminder about the meeting tomorrow at 10 AM."
classify_email(sample_email_2, model)

