import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from collections import Counter
import os

"""
    first :
            i will make a Dictionary So it contains every Word on the Document 
            and Then Remove unwanted items .
            unwanted items => Less Common and Have Numbers .
            
            So Lets Do a Function Called dic_process()


"""
TRAIN_DIR = "./train-mails"
TEST_DIR = "./test-mails"

def dic_process(root_dir):
    all_words = []
    emails= [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    for email in emails :
        with open (email) as m:
            for line in m :
                words = line.split()
                all_words+=words
    dictionary = Counter(all_words)
    list_to_remove = list(dictionary.keys())

    for item in list_to_remove:

        if item.isalpha() == False:# remove the word which has the numbers from the the dictionary 
            del dictionary[item]
        elif len(item) == 1:# remove the alone letter  
            del dictionary[item] 
    dictionary = dictionary.most_common() # this the final dictinoary

    return dictionary



"""
    Second :
            i will extract the features and put them in a matrix
            
            So Lets Do a Function Called ext_features()


"""



def ext_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    feature_matrix = np.zeros(len(files))# why this ?
    
    train_labels = np.zeros(len(files))
    count = 0
    docId = 0

    for fil in files :
        with open(fil) as fi:
            for i,line in enumerate(fi):
                if i == 2 :
                    words = line.split()
                    for word in words :
                        wordId=0
                        for i,d in enumerate(dictionary):
                            if d[0] == word:
                                wordId = i
                                feature_matrix[docId]= words.count(word)
            train_labels[docId] = 0
            filepathTokens = fil.split('/')
            lastToken = filepathTokens[len(filepathTokens) - 1]
            if lastToken.startswith("spmsg"):
                train_labels[docId] = 1
                count = count + 1
            docId = docId+1
    return feature_matrix , train_labels

dictionary = dic_process(TRAIN_DIR)
print ("reading and processing emails from file.")
features_matrix =np.reshape(ext_features(TRAIN_DIR),(1404,1))
labels =np.reshape(ext_features(TRAIN_DIR),(1404,1))

test_feature_matrix =np.reshape(ext_features(TEST_DIR),(520,1))
test_labels = np.reshape(ext_features(TEST_DIR),(520,1))


clf = GaussiaInNB()
print("Training Model :")
clf.fit(features_matrix,labels)

predicted_labels = clf.predict(test_feature_matrix)

print (accuracy_score(test_labels,predicted_labels))






