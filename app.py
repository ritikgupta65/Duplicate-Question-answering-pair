from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np  
import re
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz

app = Flask(__name__)

# Load the trained model and other necessary objects
model_path = 'model.pkl'
cv_path = 'cv.pkl'
stopwords_path = 'stopwords.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

cv = pickle.load(open(cv_path, 'rb'))
STOP_WORDS = pickle.load(open(stopwords_path, 'rb'))
# import nltk
# from nltk.corpus import stopwords
# STOP_WORDS = stopwords.words("english")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    q1 = request.form['question1']  # Question 1
    q2 = request.form['question2']  # Question 2

    # Generate feature vector from the input questions
    query_features = query_point_creator(q1, q2)

    # Make the prediction
    prediction = model.predict(query_features)
    output = 'Duplicate' if prediction == 1 else 'Non-Duplicate'

    return render_template('index.html', prediction_text=f'Prediction: {output}')


# Feature extraction functions

def test_common_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def test_total_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1) + len(w2)

def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 2

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    return length_features

def test_fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features

def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent').replace('$', ' dollar ').replace('₹', ' rupee ').replace('€', ' euro ').replace('@', ' at ')
    q = q.replace('[math]', '')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    contractions = { "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is", "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it's": "it is", "let's": "let us", "ma'am": "madam", "shan't": "shall not", "she's": "she is", "there's": "there is", "we'd": "we would", "we're": "we are", "weren't": "were not", "what's": "what is", "who's": "who is", "won't": "will not", "you're": "you are"}
    
    q_decontracted = [contractions[word] if word in contractions else word for word in q.split()]
    q = ' '.join(q_decontracted).replace("'ve", " have").replace("n't", " not").replace("'re", " are").replace("'ll", " will")
    
    q = BeautifulSoup(q, 'html.parser').get_text()  # Remove HTML tags
    q = re.sub(r'\W', ' ', q).strip()  # Remove punctuations
    
    return q

def query_point_creator(q1, q2):
    input_query = []

    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query.append(len(q1))
    input_query.append(len(q2))
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

    token_features = test_fetch_token_features(q1, q2)
    input_query.extend(token_features)

    length_features = test_fetch_length_features(q1, q2)
    input_query.extend(length_features)

    fuzzy_features = test_fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)

    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 21), q1_bow, q2_bow))


if __name__ == "__main__":
    app.run(debug=True)
    # numpy ==1.24.4
    # python ==3.8.4
    # pandas==2.0.3
    # scikit-learn==1.3.2
