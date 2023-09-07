#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications

"""
#  Initialise NLTK Inference
#######################################################
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring
# Initialise csv and sklearn libraries
#######################################################
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#######################################################
# Initialise fish agent
#######################################################
import json, requests
#######################################################
#  Initialise AIML agent
#######################################################
import aiml
#######################################################
#  Initialise Knowledgebase. 
#######################################################
import pandas
kb=[]
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
#######################################################
#  Initialise BoW and TF-IDF
#######################################################
corpus = []
questions = []
answers = []

with open('QA.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        corpus.append(row[0])
        questions.append(row[0])
        answers.append(row[1])

count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(corpus)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")
#######################################################
# Welcome user
#######################################################
print("Welcome to this chat bot. Please feel free to ask questions from me!")
#######################################################
# Main loop
#######################################################
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
        
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 2:
            succeeded = False
            api_url = "https://fishbase.ropensci.org/species?limit=10&offset=0"
            headers = {"Accept": "application/vnd.ropensci.v6+json"}
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json:
                    fish_name = response_json["data"][0]["Species"]
                    habitat = response_json["data"][0]["Habitat"]
                    print(f"The {fish_name} fish lives in {habitat}.")
                    succeeded = True
            if not succeeded:
                print("Sorry, I couldn't find the fish you are looking for")
             
        # Here are the processing of the new logical component:
        elif cmd == 31: # if input pattern is "I know that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            # >>> ADD SOME CODES HERE to make sure expr does not contradict 
            # with the KB before appending, otherwise show an error message.
            kb.append(expr) 
            print('OK, I will remember that',object,'is', subject)
        elif cmd == 32: # if the input pattern is "check that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               print('It is not true') 
             
        elif cmd == 99:
            answer.split('$')
            answer = answer[1];
            # use cosine similarity to find closest matching question
            userInput_tfidf = tfidf_transformer.transform(count_vect.transform([userInput]))
            similarities = (X_tfidf * userInput_tfidf.T).toarray()
            closest_match_index = similarities.argmax()
            
            if similarities[closest_match_index][0] > 0.6:
                answer = answers[closest_match_index]
                print(answer)
            else:
                answer = "I'm sorry, I did not get that"
                print(answer)
    else:
        print(answer)
        