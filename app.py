from flask import Flask, render_template,session,flash,request
import re
import nltk
import pickle
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from langdetect import detect
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import CountVectorizer
def get_pos(word):
    tag=nltk.pos_tag([word])[0][1][0].upper()
    tag_dict={"J":wordnet.ADJ,"N":wordnet.NOUN,"V":wordnet.VERB,"R":wordnet.ADV}
    return tag_dict.get(tag,wordnet.NOUN)
lemmatizer=WordNetLemmatizer()

with open("vectorizer.pkl","rb") as f:
     vecorizer=pickle.load(f)
best_model=Sequential([
                  Dense(20,activation="relu",input_dim=10000),
                  Dropout(0.2),
                  Dense(1,activation="sigmoid")
])
best_model.load_weights("My_model (6).h5")
best_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
app=Flask(__name__)
app.secret_key="kunu_lucky_pintu"
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/predict",methods=["POST","GET"])
def predict():
  #  data = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)
   # from nltk.corpus import stopwords
  #from nltk import wordnet
   # from nltk.stem import WordNetLemmatizer
   # corpus = []
   # punc = """:;""?!#@&.,"""
   # lema = WordNetLemmatizer()
   # for i in range(0, 1000):
   #     mess = [w for w in data.Review[i] if w not in punc]
   #     mess = "".join(mess)
   #     mess = mess.lower()
   #     all_stop_word = stopwords.words("english")
   #     all_stop_word.remove('not')
   #     all_stop_word.remove('no')
   #     all_stop_word.remove("didn't")
   #     all_stop_word.remove("won't")
   #     all_stop_word.remove("shan't")

      #  message = [lema.lemmatize(word) for word in mess.split() if word not in set(all_stop_word)]
      #  message = " ".join(message)
      #  corpus.append(message)
    #from sklearn.feature_extraction.text import CountVectorizer
    #cv = CountVectorizer(max_features=1500)
    #X = cv.fit_transform(corpus).toarray()
    #y = data.iloc[:, -1].values
    #from sklearn.linear_model import LogisticRegression
    #classifier = LogisticRegression()
    #classifier.fit(X, y)
    #model = [cv, classifier]
    #with open("model_nlp.pkl", "wb") as f:
    #    pickle.dump(model, f)
    if request.method=="POST":
        linkedin_slogan=request.form["lslogan"]
        linkedin_overview=request.form["loverview"]
        linkedin_industry=request.form["lindustry"]
        linkedin_specialities=request.form["lspeciality"]
        crunchbase_slogan=request.form["cslogan"]
        crunchbase_industries=request.form["cindustry"]
        crunchbase_overview=request.form["coverview"]
        text=linkedin_industry+" "+linkedin_overview+" "+linkedin_slogan+" "+linkedin_specialities+" "+crunchbase_industries+" "+crunchbase_slogan+" "+crunchbase_slogan
        if len(text)<10:
           flash("Please Write Something","info")
           return render_template("home.html")
        else:
             sen = re.sub("[^A-Za-z]", " ", text)
             sen = sen.lower()
             words = sen.split()
             words = [lemmatizer.lemmatize(word, get_pos(word)) for word in words if word not in set(stopwords.words("english"))]
             text=" ".join(words)
             if detect(text)!="en":
                flash("Sorry! Please either change the language to english or write something meaningful","info")
                return render_template("home.html")
             text_array=vecorizer.transform([text]).toarray()
             val=best_model.predict(text_array)
             if val[0][0]>0.5:
                 flash("The Company is a Social company", "info")
             else:
                 flash("The Company is an Unsocial company", "info")
             return render_template("home.html")
    
if __name__=="__main__":
    app.run(debug=True)
