# Emotion Detection based on Textual Data

**Dataset**

For this projects , I used some datasets which is train dataset for training model, development dataset for testing purpose, and  test dataset which is actually use for predicting label.
In train dataset and development dataset has three label that are id, document, and topic label. here id is not necessary and we used document and topic in our code. And development dataset has also three label which is id ,document,topic label and here id is not necessary and we used document and topic label in our code. Test dataset has two label which is id and document and where document is used for predicting labels.

Dataset Statices:

**Training dataset**\
disgust  = 603\
surprise = 3113\
fear     = 2269\
joy      = 6677\
anger    = 1237\
sadness  = 3101

**Development dataset**\
disgust  = 103\
surprise = 469\
fear     = 346\
joy      = 988\
anger    = 196\
sadness  = 449

**Implementation**

For this project, I used machine learning algorithm like SVM and NB 

**Support Vector Machine(SVM)** :\
 It is a supervised machine learning algorithm which will be used for both classification or regression challenges.

In my code, I used svc algorithm which is class capable of performing multiclass classification on our datasets.

**Navie Bayes(NB)** :\
It is a  supervised machine learning algorithms based on applying Bayes theorem with the ‘naive’ assumption of conditional independence between every pair of features given the value of the class variable.

In my code, I used Multinomical Naive Bayes Classifier.It implements the navie Bayes algorithm for multinomially distributed data, and is one of the classic navie bayes varients used in text-classification. 

**Preprocessing**\
I used some preprocessing technique like text-lower case, remove ascii character, remove punctuation, remove special character, remove unicode, remove default stop word and apply word stemming.

In order to use textual data for predictive modeling, the text must be parsed to remove certain words – this process is called Tokenization. 

The Tokenization words need to be encoded as integers, or floating-point values, for use as inputs in machine learning algorithms. This process is called feature extraction (or vectorization). So, I used two feature extraction method which is Tfidf-vectorizer and Count-Vectorizer.

**Result**:\
Here I show the **Accuracy**, **Precision**, **Recall**,  **F1-score**, and **Confusion Matrix**

**Training The SVM Classifier and TfidfVectorizer....**

Accuracy =  0.6056448451587613

              precision    recall  f1-score   support

       anger      0.600     0.291     0.392       196
     disgust      0.552     0.155     0.242       103
        fear      0.794     0.566     0.661       346
         joy      0.598     0.865     0.707       988
     sadness      0.512     0.430     0.467       449
    surprise      0.610     0.486     0.541       469

    accuracy                          0.606      2551
    macro avg     0.611     0.466     0.502      2551
    weighted avg  0.610     0.606     0.585      2551

Confusion Matrix :\
['anger' 'disgust' 'fear' 'joy' 'sadness' 'surprise']\
[[ 57   3  12  76  27  21]\
 [  5  16   4  44  27   7]\
 [  7   2 196  83  32  26]\
 [ 11   3  11 855  46  62]\
 [ 13   4  11 198 193  30]\
 [  2   1  13 173  52 228]]


**Training The SVM Classifier and CountVectorizer....**

Accuracy =  0.5762446099568796

              precision    recall  f1-score   support

       anger      0.390     0.352     0.370       196
     disgust      0.408     0.301     0.346       103
        fear      0.590     0.616     0.603       346
         joy      0.644     0.730     0.684       988
     sadness      0.497     0.441     0.468       449
    surprise      0.567     0.507     0.535       469

    accuracy                          0.576      2551
    macro avg     0.516     0.491     0.501      2551
    weighted avg  0.568     0.576     0.570      2551


Confusion Matrix :\
['anger' 'disgust' 'fear' 'joy' 'sadness' 'surprise']\
[[ 69   7  22  52  25  21]\
 [  8  31  12  26  17   9]\
 [ 15   6 213  54  30  28]\
 [ 40  17  49 721  76  85]\
 [ 27  11  28 146 198  39]\
 [ 18   4  37 120  52 238]]



**Training The NB Classifier and TfidfVectorizer....**

Accuracy =  0.3888671109368875

              precision    recall  f1-score   support

       anger      0.000     0.000     0.000       196
     disgust      0.000     0.000     0.000       103
        fear      0.000     0.000     0.000       346
         joy      0.388     0.997     0.559       988
     sadness      0.700     0.016     0.031       449
    surprise      0.000     0.000     0.000       469

    accuracy                          0.389      2551
    macro avg     0.181     0.169     0.098      2551
    weighted avg  0.274     0.389     0.222      2551

Confusion Matrix :\
['anger' 'disgust' 'fear' 'joy' 'sadness' 'surprise']\
[[  0   0   0 196   0   0]\
 [  0   0   0 103   0   0]\
 [  0   0   0 344   1   1]\
 [  0   0   0 985   1   2]\
 [  0   0   0 442   7   0]\
 [  0   0   0 468   1   0]]



**Training The NB Classifier and CountVectorizer....**

Accuracy =  0.4331634653077225

              precision    recall  f1-score   support

       anger      0.182     0.020     0.037       196
     disgust      0.333     0.029     0.054       103
        fear      0.408     0.306     0.350       346
         joy      0.460     0.760     0.574       988
     sadness      0.354     0.245     0.289       449
    surprise      0.412     0.279     0.333       469

    accuracy                          0.433      2551
    macro avg     0.358     0.273     0.273      2551
    weighted avg  0.399     0.433     0.387      2551


Confusion Matrix :\
['anger' 'disgust' 'fear' 'joy' 'sadness' 'surprise']\
[[  4   0  22 124  25  21]\
 [  2   3   7  65  19   7]\
 [  5   0 106 169  38  28]\
 [  5   5  71 751  72  84]\
 [  2   0  32 258 110  47]\
 [  4   1  22 264  47 131]]

