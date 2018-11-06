from utils import get_data, format_sequence, generate_output, seed_sequence
from keras.models import load_model
import numpy as np
import json
from flask import request
import flask
import io

app = flask.Flask(__name__)
model = None

def load_keras_model():
    """Load in the pre-trained model"""
    global model
    model = load_model('../models/train-embeddings-rnn.h5')
    

@app.route('/query-example')
def query_example():
    r = request.args.get('language')
    return f'<h1>{r}</h1>'


@app.route("/predict", methods = ["POST"])
def predict():
    """Make a prediction with the model"""
    data = {'success': False}
    
    if flask.request.method == "POST":
        s = request.args.get('sequence', None)
#         s = request
#         return(f'<h1>{s.args}</h1')
        word_idx = json.load(open('../data/word-index.json'))
        idx_word = {idx: word for word, idx in word_idx.items()}
        out = seed_sequence(model, s, word_idx, idx_word)
        data['success'] = True
        data['out'] = out
        
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
          "please wait until server has fully started"))
    load_keras_model()
    app.run()