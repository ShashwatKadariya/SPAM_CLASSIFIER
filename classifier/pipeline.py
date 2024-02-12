import pandas as pd
import re
from sklearn.model_selection import train_test_split
from NaiveBayesClassifier import NaiveBayesClassifier

df = pd.read_csv("./Data/spam_ham.csv")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

df['tokens'] = df['text'].apply(preprocess)

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


model = NaiveBayesClassifier()
model.train(train_data)

correct = 0
for _, row in test_data.iterrows():
    prediction = model.predict(row['tokens'])
    if prediction == row['spam']:
        correct += 1

accuracy = correct / len(test_data)
print(f"Accuracy: {accuracy:.2f}")

