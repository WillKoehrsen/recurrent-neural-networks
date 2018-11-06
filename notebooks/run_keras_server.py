from utils import get_data, format_sequence, generate_output, seed_sequence, addContent, header, box, remove_spaces
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy as np
import json
from flask import request
import flask
import io

app = flask.Flask(__name__)

def load_keras_model():
    """Load in the pre-trained model"""
    K.clear_session()
    global model
    model = load_model('../models/train-embeddings-rnn.h5')
    global graph
    graph = tf.get_default_graph()
   
def generate_output(s, words_to_generate = 50, diversity = 0.75):
    """Generate output from a sequence"""
    # Mapping of words to integers
    word_idx = json.load(open('../data/word-index.json'))
    idx_word = {idx: word for word, idx in word_idx.items()}
    
    # Original formated text
    start = format_sequence(s).split()
    gen = []
    s = start[:]
    
    with graph.as_default():
        
        # Generate output
        for i in range(words_to_generate):
            # Conver to array
            x = np.array([word_idx.get(word, 0) for word in s]).reshape((1, -1))

            # Make predictions
            preds = model.predict(x)[0].astype(float)

            # Diversify
            preds = np.log(preds) / diversity
            exp_preds = np.exp(preds)
            # Softmax
            preds = exp_preds / np.sum(exp_preds)

            # Pick next index
            next_idx = np.argmax(np.random.multinomial(1, preds, size = 1))
            s.append(idx_word[next_idx])
            gen.append(idx_word[next_idx])
    
    # Formatting in html
    start = remove_spaces(' '.join(start)) + ' '
    gen = remove_spaces(' '.join(gen)) 
    html = ''
    html = addContent(html, header('Input Seed ', color = 'black', gen_text = 'Network Output'))
    html = addContent(html, box(start, gen))
    return html

@app.route('/query-example')
def query_example():
    r = request.args.get('language')
    return f'<h1>{r}</h1>'


@app.route("/predict")
def predict():
    """Make a prediction with the model"""
    data = {'success': False}
    s = request.args.get('sequence', None)
    out = generate_output(s)
    data['success'] = True
    data['out'] = out
        
    return data['out']

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
          "please wait until server has fully started"))
    load_keras_model()
    app.run(host="0.0.0.0", port=10000)
