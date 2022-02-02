from distutils.command.clean import clean
from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib as plt
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# import the Flask class, instance of this class will be WSGI (Web Server Gateway Interace)
import pickle

# instance of Flask class
# __name__ is a convenient shortcut to the app's module or package 
# tells Flask where to look for resources and templates
app = Flask(__name__)

# gets the model that I made in the notebook
    # TODO: train() here
clean_data = pickle.load(open("../ufo-clean-data.pkl", "rb"))
# model = pickle.load(open("../ufo-model.pkl", "rb"))

#Challenge train inside of the app instead of notebook
# model = 

@app.route("/")
def home():
    return render_template("index.html")
# The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned.
# The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.
@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    # converting to numpy type array 
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    #
    # myList = []
    # for i in range(10):
    #   myList.append(i)
    # myList = [i for i in range(10)] <= same as above for loop 
    # gets the index of the 
    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]  

    # render template: give name of template, and the variables
    # {} are for formatting the string, means just place them in
    return render_template("index.html", prediction_text="Likely country: {}".format(countries[output]))

    
# TODO: Add a train function to train the model rather than grabbing from pickle model
@app.route("/train", methods=["POST"])
def train():
    selected_features = ["Seconds", "Latitude", "Longitude"]
    X = clean_data[selected_features]
    y = clean_data["Country"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression()
    model.fit(X_train.values, y_train.values)

    int_features = [int(x) for x in request.form.values()]
    # converting to numpy type array 
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]  

    # render template: give name of template, and the variables
    # {} are for formatting the string, means just place them in
    return render_template("index.html", prediction_text="Likely country: {}".format(countries[output]))
 
if __name__ == "__main__":
    app.run(debug=True)

"""
    Pros: all done in this file and runs from one place so easy to do 
    - no need for extra files

    Cons: every time the button is presed then we have to train every time
"""