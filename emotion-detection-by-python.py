
import nltk
import re
import string 
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
import csv 
stemmer = PorterStemmer()

def file_write(test_predict,test_id): # this function generate our output file in output floder

    #print(test_id)
    #print(test_predict)

    # with open(file_name, 'w') as f: 
    #     for line in test_predict: 
    #         f.write('%s\n' % line)

    with open('predict.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(test_id,test_predict))

def preprocessing(documents): 
    
    examples = []
    stop_words = set(stopwords.words("english")) # set to the english stop words
    for line in documents:

        #print(line)
        #tokens = line.split()
        
        str1 = " "  # it is empty string variable
        str2 = str1.join(line)  # convert tokens list to string
      
        #this lower() method convert lower case letter
        text_lower = str2.lower()

        # this method remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        text2 = text_lower.translate(translator)
       
        #$convert unicode to ascii char or remonve
        to_ascii = unicodedata.normalize('NFKD', text2).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        #print(to_ascii)


        #remove_num = re.sub(r'\d+', '', text_lower) # this re.sub() remove number

        #step10 : remove special characters
        remove_sp_char = " ".join(re.findall(r"[a-zA-Z0-9]+",to_ascii))

        # step11 : remove default stop words
        word_tokens = word_tokenize(remove_sp_char)
        filtered_text = [word for word in word_tokens if word not in stop_words] 
        #print(filtered_text)

        stems = [stemmer.stem(word) for word in filtered_text]

        str3 = " " 
        str4 = str3.join(stems) 

        examples.append(str4)

    return examples

def read_files():
    train_data= []
    dev_data= []
    test_data = []

    # reads the training data
    with open('train.txt', 'r' , encoding='utf-8',errors='ignore') as train_file:
        for line in train_file:
            train_data.append(line)
    
    # reads the development data
    with open('dev.txt', 'r' ,encoding='utf-8',errors='ignore') as dev_files:
        for line in dev_files:
            dev_data.append(line)
    
    # reads the test data
    with open('test.txt', 'r' ,encoding='utf-8',errors='ignore') as test_file:
        for line in test_file:
            test_data.append(line)
    
    return train_data, dev_data, test_data


def separate_labels(data):

    documents = []
    labels = []

    for line in data:
        #splitted_line = line.split('\t', 2)
        # separate the labels and examples (docs) in different list
        tokens = line.split()

        labels.append(tokens[-1:])
        documents.append(tokens[1:-2])

        #labels.append(line[-1:])
        #documents.append(line[1:-2])

    return documents, labels


def separate_lbls_2(data):
    test_id = []
    documents = []
    for line in data:

        tokens = line.split()
        test_id.append(tokens[0])
        documents.append(tokens[1:])

    return documents, test_id


def identity(X):
    return X


def vectorization(is_tfidf=True):
    
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if is_tfidf:
        vec = TfidfVectorizer(preprocessor = identity, lowercase=True, analyzer='char', 
                            tokenizer = identity, ngram_range=(2,5))

    else:
        vec = CountVectorizer(preprocessor = identity, lowercase=True, analyzer='char', 
                            tokenizer = identity, ngram_range=(2,5))

    return vec

def vectorization2(is_tfidf=True):
    
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if is_tfidf:
        vec = TfidfVectorizer(preprocessor = identity,tokenizer = identity)
    else:
        vec = CountVectorizer(preprocessor = identity, tokenizer = identity)

    return vec

def SVM_with_tfidf(train_docs, train_lbls, dev_docs, dev_lbls,test_docs,test_id):
    
    # calls the vectorization function
    vec = vectorization(is_tfidf=True)

    # combine the vectorizer with a SV classifier
    classifier = Pipeline( [('vec', vec),('cls', SVC(kernel='linear'))] )

    #Fit SVM classifier according to train_docs , train_lbls
    # train_docs : Training vectors, where n_samples is the number of samples. 
    # train_lbls : Target values or binary class
    classifier.fit(train_docs, train_lbls)

    # Perform classification on an array of test vectors Xtest. 
    predict = classifier.predict(dev_docs)

    predict_for_text = classifier.predict(test_docs)
    # file write for test prediction
    file_write(predict_for_text, test_id)

    # show the results of the evaluation
    results(dev_lbls, predict, classifier)


def SVM_with_count(train_docs, train_lbls, dev_docs, dev_lbls,test_docs,test_id):
    
    # calls the vectorization function
    vec = vectorization(is_tfidf=False)
    
    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline( [('vec', vec),('cls', SVC(kernel='linear'))] )

    #Fit SVM classifier according to train_docs , train_lbls
    # train_docs : Training vectors, where n_samples is the number of samples. 
    # train_lbls : Target values or binary class
    classifier.fit(train_docs, train_lbls)

    # Perform classification on an array of test vectors Xtest.
    predict = classifier.predict(dev_docs)

    #predict_for_text = classifier.predict(test_docs)
    ## file write for test prediction
    #file_write(predict_for_text, test_id)

    # show the results of the evaluation
    results(dev_lbls, predict, classifier)


def NB_with_tfidf(train_docs, train_lbls, dev_docs, dev_lbls,test_docs,test_id):
    
    # calls the vectorization function
    vec = vectorization2(is_tfidf=True)

    # combine the vectorizer with a NB classifier
    classifier = Pipeline([('vec', vec),('cls', MultinomialNB())] )

    #Fit SVM classifier according to train_docs , train_lbls
    # train_docs : Training vectors, where n_samples is the number of samples. 
    # train_lbls : Target values or binary class
    classifier.fit(train_docs, train_lbls)

    # Perform classification on an array of test vectors Xdev. 
    predict = classifier.predict(dev_docs)

    # show the results of the evaluation
    #print(predict)

    #predict_for_text = classifier.predict(test_docs)
    ## file write for test prediction
    #file_write(predict_for_text, test_id)

    results(dev_lbls, predict, classifier)

def NB_with_count(train_docs, train_lbls, dev_docs, dev_lbls,test_docs,test_id):
    
    # calls the vectorization function
    vec = vectorization2(is_tfidf=False)
    
    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec),('cls', MultinomialNB())] )

    #Fit SVM classifier according to train_docs , train_lbls
    # train_docs : Training vectors, where n_samples is the number of samples. 
    # train_lbls : Target values or binary class
    classifier.fit(train_docs, train_lbls)

    # Perform classification on an array of test vectors Xtest.
    predict = classifier.predict(dev_docs)



    #print(dev_lbls) 

    #predict_for_text = classifier.predict(test_docs)
    ## file write for test prediction
    #file_write(predict_for_text, test_id)

    # show the results of the evaluation
    results(dev_lbls, predict, classifier)

def results(dev_lbls, predict, classifier):

    # Compare the accuracy of the output (predict) with the class labels of the original test set (test_lbls)
    print("Accuracy = ",accuracy_score(dev_lbls, predict))

    # Report on the precision, recall, f1-score of the output (Yguess) with the class labels of the original test set (Ytest)
    print(classification_report(dev_lbls,predict,labels=classifier.classes_,digits=3))
   
    print("\nConfusion Matrix : ")
    print(classifier.classes_)
    print(confusion_matrix(dev_lbls, predict, labels=classifier.classes_))
    print("\n")


def main():

    print('Reading The Dataset....')
    train_data,dev_data,test_data= read_files()

    train_docs, train_lbls = separate_labels(train_data)
    dev_docs, dev_lbls = separate_labels(dev_data)

    test_docs, test_id = separate_lbls_2(test_data)

    #print(train_docs)
   
    print('\nPreprocessing....')
    preprocessed_train_data = preprocessing(train_docs)
    preprocessed_dev_data = preprocessing(dev_docs)
    preprocessed_test_data = preprocessing(test_docs)

    print('\nTraining The SVM Classifier and TfidfVectorizer....\n\n')
    SVM_with_tfidf(preprocessed_train_data, train_lbls, preprocessed_dev_data, dev_lbls, preprocessed_test_data, test_id)

    print('\nTraining The SVM Classifier and CountVectorizer....\n\n')
    SVM_with_count(preprocessed_train_data, train_lbls,  preprocessed_dev_data, dev_lbls, preprocessed_test_data, test_id)

    print('\nTraining The NB Classifier and TfidfVectorizer....\n\n')
    NB_with_tfidf(preprocessed_train_data, train_lbls, preprocessed_dev_data, dev_lbls, preprocessed_test_data, test_id)

    print('\nTraining The NB Classifier and CountVectorizer....\n\n')
    NB_with_count(preprocessed_train_data, train_lbls, preprocessed_dev_data, dev_lbls, preprocessed_test_data, test_id)

   # print(test_id)

if __name__ == "__main__":
    
    main()
    