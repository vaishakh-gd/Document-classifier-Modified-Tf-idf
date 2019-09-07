from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.tokenize import PunktSentenceTokenizer

import logging

import boto3

import pandas as pd

s31=boto3.resource('s3')
some_binary_data = 'ninna nanna manavu'
object = s31.Object('com-wonderful', 'filename.txt')
object
object.put(Body=some_binary_data)

# i-090f0ef66640bae59



f=open("logtext1.txt","w+")

logging.basicConfig(filename='example.log',level=logging.DEBUG)


from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


from nltk.stem import wordnet

import os

stopWords = set(stopwords.words('english'))


import re

dict_p={}
dict_n={}
object.put(Body=str(dict_p))

stemmer = SnowballStemmer("english")

def returnStemmedWords(words):
    w=[stemmer.stem(word) for word in words]
    return w





def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    for i in filters:
        text.replace(i," ")

    return text









def goodwordsreturn(document,filename,sentiment):
    document=clean_text(document)
    words=word_tokenize(document)
    logging.info(str(document))
    words=[word for word in words if word not in stopWords]
    words = [word.lower() for word in words]
    words = returnStemmedWords(words)
    logging.info(str(words))

    for word in words:
        val = re.search('[a-zA-Z]+', word)

        if(sentiment=="neg" and val!=None):
            if(len(dict_n) < 1):
                dict_n[word] = 0

            elif (word not in dict_n.keys()):
                dict_n[word] = 1/12500


            else:
                dict_n[word] += 1/12500

        elif(val!=None):
            if (len(dict_p) < 1):
                dict_p[word] = 0

            elif(word not in dict_p.keys()):
                dict_p[word] = 1/12500


            else:
                dict_p[word]+=1/12500










#aclImdb


data_dir="aclImdb/"

for split in ["train"]:
    for sentiment in ["neg", "pos"]:
        score = 1 if sentiment == "pos" else 0

        path = os.path.join(data_dir, split, sentiment)
        file_names = os.listdir(path)
        for f_name in file_names:
            with open(os.path.join(path, f_name), "r") as f:
                review = goodwordsreturn(f.read(),f_name,sentiment)







#df = pd.DataFrame.from_dict(d, orient='index',columns=['token_category_frequency'])
#s3_url = f"s3://com-wonderful/dict_p/output.parquet"
#df.to_parquet('s3://com-wonderful/dict_p/output.parquet')










print("all ok")



print(str(dict_p))
print("------------------------------------------")
print(str(dict_n))


terms_considered_positive=[]


def term_frequency_pos(word):
    som=0
    #logging.info("---------------------------------------------------")
    for i in dict_p.keys():
        if(word==i):
            som+=dict_p[i]

            #logging.info(sup)
    #logging.info("---------------------------------------------------")


    return som

def term_frequency_neg(word):
    som=0
    #logging.info("---------------------------------------------------")
    for i in dict_n.keys():
        if(word==i):
            som+=dict_n[i]

            #logging.info(sup)
    #logging.info("---------------------------------------------------")

    return som






#(1/number_of_words_gone_previous)*Tf((+),word)*Tf((+),previous)+1/number_of_words_gone_previous*Tf((-),word)*Tf((-),previous)



list_for_df=[]

def foo(doc):
    document = clean_text(doc)
    words = word_tokenize(document)
    words = [word for word in words if word not in stopWords]
    words = [word.lower() for word in words]
    words = returnStemmedWords(words)
    som=0
    som=term_frequency_pos(words[0])-term_frequency_neg(words[0])
    logging.info("PRINTED SOM IS  ")
    logging.info(str(som))
    xispos=0
    for i in range(len(words)):
        c=i
        count=1
        while(c>=0 and i!=0):
            c-=1
            count+=1
            if(c>=0):

                prod_pos=(1/count)*term_frequency_pos(words[c])*term_frequency_pos(words[i])
                prod_neg=(1/count)*term_frequency_neg(words[c])*term_frequency_neg(words[i])
                x=prod_pos-prod_neg

                if(x>=0):
                    xispos+=1
                else:
                    xispos-=1


    return som,xispos








for split in ["test"]:
    for sentiment in ["pos","neg"]:

        score = 1 if sentiment == "pos" else 0

        path = os.path.join(data_dir, split, sentiment)
        file_names = os.listdir(path)
        for f_name in file_names:
            with open(os.path.join(path, f_name), "r") as f:
                review,xispos = foo(f.read())
                list_for_df.append([f_name,sentiment,review,xispos])






df6 = pd.DataFrame(list_for_df,columns=['f_name','sentiment','review_som','xispos'])

s3_url = f"s3://com-wonderful/dict_p/output.parquet"
df6.to_parquet('s3://com-wonderful/megadataframe/output.snappy.parquet',
              compression='snappy')
print("all ok")


































#[word for word in tokenized_words if word not in stop_words]
 #this be the best





from nltk.stem import LancasterStemmer
lst=LancasterStemmer()




words = ["game", "gaming", "gamed", "games","gone","went","going","reached","grouped","seductive"]


 

for word in words:
    print(stemmer.stem(word))



