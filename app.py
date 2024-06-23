import nltk
import mysql.connector

nltk.download('popular')#  Natural Language Toolkit (NLTK) library installed will download and
# install a collection of popular NLTK data packages, including corpora,
# models etc
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/chatbot')
def chatbot():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="jj@1405",
        database="mydatabase"
    )

    cursor = connection.cursor()
    query = "SELECT * FROM chatbot_conversations;"
    cursor.execute(query)
    conversations = cursor.fetchall()
    cursor.close()
    connection.close()
    
    return render_template('chatbot.html', conversations = conversations)

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    response = chatbot_response(userText)


    # Establish a connection to the MySQL database
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="jj@1405",
        database="mydatabase"
    )

    # Create a cursor object to execute SQL queries
    cursor = connection.cursor()

    # Example: Insert a user message into the table
    user_id = 1
    message_type= "ChatBot"
    content = response
    cursor.execute("INSERT INTO chatbot_conversations (user_id, message_type, content) VALUES (%s, %s, %s),(%s, %s, %s);", (user_id, message_type, content, user_id, "you", userText))


    # print("hellllllllo")
    # Commit the changes to the database
    connection.commit()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    return response


if __name__ == "__main__":
    app.run()