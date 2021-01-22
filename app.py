from flask import Flask, render_template,session,flash,request
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
def get_pos(word):
    tag=nltk.pos_tag([word])[0][1][0].upper()
    tag_dict={"J":wordnet.ADJ,"N":wordnet.NOUN,"V":wordnet.VERB,"R":wordnet.ADV}
    return tag_dict.get(tag,wordnet.NOUN)
lemmatizer=WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
with open("vector.pkl","rb") as f:
     vecorizer=pickle.load(f)
best_model=keras.models.load_model("My_model (5).h5")
best_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
app=Flask(__name__)
app.secret_key="kunu_lucky_pintu"
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method=="POST":
        linkedin_slogan=request.form["lslogan"]
        linkedin_overview=request.form["loverview"]
        linkedin_industry=request.form["lindustry"]
        linkedin_specialities=request.form["lspeciality"]
        crunchbase_slogan=request.form["cslogan"]
        crunchbase_industries=request.form["cindustry"]
        crunchbase_overview=request.form["coverview"]
        text=linkedin_industry+linkedin_overview+linkedin_slogan+linkedin_specialities+crunchbase_industries+crunchbase_slogan+crunchbase_slogan
        if len(text)<4:
           flash("Please Write Something","info")
           return render_template("home.html")
        sen = re.sub("[^A-Za-z]", " ", text)
        sen = sen.lower()
        words = sen.split()
        words = [lemmatizer.lemmatize(word, get_pos(word)) for word in words if word not in set(stopwords.words("english"))]
        text=" ".join(words)
        text_array=vecorizer.transform(text).toarray()
        val=best_model.predict(text_array)
        if val[0][0]>0.5:
            flash("The Company is a Social company", "info")
        else:
            flash("The Company is an Unsocial company", "info")
        return render_template("home.html")


if __name__=="__main__":
    app.run(debug=True)