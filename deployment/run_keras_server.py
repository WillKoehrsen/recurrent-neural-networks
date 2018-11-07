from utils import generate_output, generate_from_seed
from keras.models import load_model
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField,  DecimalField, IntegerField
import flask
import io

DEBUG = False
app = flask.Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):

    seed = TextField('Enter a seed string or random:', validators=[
                     validators.InputRequired(message='A seed string is required')])
    diversity = DecimalField('Enter diversity:', default=0.8, validators=[validators.InputRequired(
    ), validators.NumberRange(min=0.5, max=5.0, message='Diversity must be between 0.5 and 5.')])
    words = IntegerField('Enter number of words to generate:', default=50, validators=[validators.InputRequired(
    ), validators.NumberRange(min=10, max=100, message='Number of words must be between 10 and 100')])
    submit = SubmitField("Send")


def load_keras_model():
    """Load in the pre-trained model"""
    global model
    model = load_model('../models/train-embeddings-rnn.h5')
    global graph
    graph = tf.get_default_graph()


@app.route("/", methods=['GET', 'POST'])
def home():
    form = ReusableForm(request.form)

    if request.method == 'POST':

        seed = request.form['seed']
        diversity = float(request.form['diversity'])
        words = int(request.form['words'])

        if form.validate():
            if seed == 'random':
                return generate_output(model, graph,
                                       new_words=words, diversity=diversity)
            else:
                return generate_from_seed(model, graph, seed, new_words=words, diversity=diversity)

        # else:
        #     flash('All the form fields are required. ')

    return render_template('home.html', form=form)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_keras_model()
    app.run(host="0.0.0.0", port=10000)
