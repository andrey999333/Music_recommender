
import numpy as np
import pandas as pd
from sklearn.externals import joblib

import flask
from flask import request

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
# #### initialize our Flask application and the Keras model



app = flask.Flask(__name__)


def load_mod():
    # load the pre-trained Keras model and all necessary additions to organize the data
    global graph,model,scaler_age,cat_dict,features
    graph = tf.get_default_graph()
    model = load_model('music_rec.hdf5')
    scaler_age = joblib.load("scaler.pkl")
    cat_dict=joblib.load("cat_map.pkl")
    features=joblib.load("features.pkl")

def data_prepare(data):
    #convert data form json dict into a list and encode categories as int for feeding itno model

    data1={}

    #categorical data
    for key in cat_dict:
        data1[key]=np.array([cat_dict[key][el] if el in cat_dict[key] else cat_dict[key]['unknown'] for el in data[key]])

    #numerical data
    for key in data:
        if key not in cat_dict:
            data1[key]=np.array(data[key])

    #scale age
    data1['age'] = np.reshape(data1['age'],(-1,1))
    data1['age'] = scaler_age.transform(data1['age'])

    return [data1[feature] for feature in features]

@app.route('/prediction', methods=['POST'])
def prediction():

    #get data
    data = request.json
    data = data_prepare(data)

    #make prediction
    with graph.as_default():
        predict=model.predict_on_batch(data)

    #get prediction to be in binary mode and json serializable
    predict=np.squeeze(np.floor(predict+1./2).astype(int)).tolist()

    #return prediction
    return flask.jsonify(predict)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))

    load_mod()
    app.run()
