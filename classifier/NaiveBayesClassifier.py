import math
from collections import defaultdict, Counter



class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {}
        self.word_probs = {}
        self.vocab = set()
    
    def train(self, data):
        spam_words = []
        ham_words = []
        spam_count = 0
        ham_count = 0
        
        for _, row in data.iterrows():
            if row['spam'] == 1:
                spam_words.extend(row['tokens'])
                spam_count += 1
            else:
                ham_words.extend(row['tokens'])
                ham_count += 1
        
        self.priors[1] = spam_count / len(data)
        self.priors[0] = ham_count / len(data)
        
        spam_freq = Counter(spam_words)
        ham_freq = Counter(ham_words)
        self.vocab = set(spam_freq.keys()).union(set(ham_freq.keys()))
        vocab_size = len(self.vocab)
        
        # Laplace smoothing
        self.word_probs[1] = {word: (spam_freq[word] + 1) / (len(spam_words) + vocab_size) for word in self.vocab}
        self.word_probs[0] = {word: (ham_freq[word] + 1) / (len(ham_words) + vocab_size) for word in self.vocab}
    
    def predict(self, tokens):
        log_prob_spam = math.log(self.priors[1])
        log_prob_ham = math.log(self.priors[0])
        
        for word in tokens:
            if word in self.vocab:
                log_prob_spam += math.log(self.word_probs[1].get(word, 1e-6))
                log_prob_ham += math.log(self.word_probs[0].get(word, 1e-6))
        
        return 1 if log_prob_spam > log_prob_ham else 0
