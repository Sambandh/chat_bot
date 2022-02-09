import os
import numpy as np
import nltk
import random
import string
from nltk.chat.util import Chat,reflections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages
#nltk.download('punkt')
#nltk.download('wordnet')
from io import StringIO
import sys

import os
flag = True

import requests
from requests import get
from lxml import html
from bs4 import BeautifulSoup
#import web_query as wq
#import combine_pdf
import gpt2_sentiment

f = open(r"C:\Users\samby.DESKTOP-N5KBK1O\Desktop\skema\chatbot\chatbot.txt", errors = 'ignore') #importing the corpus which in our case is the wiki page for chatbot
raw = f.read()
raw = raw.lower()

#Tokenizing the corpus to a list of sentences
sent_tokens = nltk.sent_tokenize(raw)
#Tokenizing the corpus to a list of words
word_tokens = nltk.word_tokenize(raw)
#Preprocessing the tokens (stemming all the words so that they get converted to stem form and then lemmatize them so that these stem words can be validated as to whether they exist from the dictionary)
#using WordNet to carry out lemmatization
lemmer = nltk.stem.WordNetLemmatizer()
#function to carry out lemmatization
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

#punctuation dictionary to store punctuations in a dictionary
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

#Normalizing the text tokens (i.e. removing the punctuations from the corpus and then tokeizing them by word. After this, we pass these word tokens to the lemmatization function 
#to find out lemmatized tokens)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#user defined function as a greeting function
GREETING_INPUTS = ("hello","hi","greetings","sup","what's up","hey",)
GREETING_OUTPUTS = ["hi","hey there","*nods*","hello","I am glad talking to you"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_OUTPUTS)

#using the TF-IDF vectorizer
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf,tfidf[-1])
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    idx = vals.argsort()[0][-2]
    if (req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you!"
        return robo_response
    else:
        robo_response=robo_response+sent_tokens[idx]
        return robo_response


set_pairs = [
[r"my name is (.*)",
["Hello %l, How are you doing today?",]
],
[r"Hi|hey|hello",
["Hello","Hey there",]
],
[r"how are you?",
["my name is anthony consalves",]
],
[r"I am fine, thank you",
["Great, How can I help you?",]
],
[r"(.*)Thank you so much, It was helpful",
["I am happy to help",]
],
[r"quit",
["Bye, take care",]
],
]

def chatbot():
    print("Hi, This is your automated assistant")
    chat = Chat(set_pairs,reflections)
    user_input = quit
    try:
        user_input=input(">")
    except EOFError:
        print(user_input)
    if user_input:
        user_input=user_input[:-1]
        if chat.respond(user_input)!=None:
            print(chat.respond(user_input))
        else:
            user_response = user_input
            user_response = user_response.lower()
            if (user_response!='bye'):
                if (user_response=='thanks' or user_response=='thank you'):
                    flag=False
                    print ("Bot: You are welcome")
                else:
                    if (greeting(user_response)!=None):
                        print ("Bot:" +greeting(user_response))
                    else:
                        if("python" in user_response):
                            print ("Bot:", end="")
                            print (response(user_response))
                            sent_tokens.remove(user_response)
                        elif("combine" and "file" in user_response):
                            for i in range(100):
                                print ("\n")
                            print ("Entering file combining mode. Please enter the exact directory we will be combining")
                            directory = input(">")
                            os.chdir(directory)
                            print ("Thank you. Now enter keyword to identify target files by title")
                            keyword = input(">")
                            reg_pattern = r'(.*'+keyword+'.+\.pdf)|(^'+keyword+'_.*\.pdf)|(.*_'+keyword+'\.pdf)|(.*'+keyword+'.+\.pdf)'
                            #Instantiate a bot object eg:bot1
                            bot1 = combine_pdf.FileBots(directory, reg_pattern)
                            #use the .locate method to find files of interest
                            bot1.locate(show=True)
                            bot1.pdf_merge_tree()
                        else:
                            print("Bot:",end="")
                            print(gpt2_sentiment.interact_model('run1',None,1,1,2,1,0,'./checkpoint')
                            )
            else:
                flag=False
                print("Bot: Bye!  Take care")

            

                

if __name__=="__main__":
    chatbot()
