import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import joblib

#ML libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# # Importing the text transcript dataset

#####Importing all the text dataset transcript into one df
df = pd.DataFrame(columns = ["Target", "Transcript"])
root_folder = "./Dataset"

for subdir, dirs, files in os.walk(root_folder):
    for FileName in files:
        #Get the current txt file
        file_path = root_folder + "/" + FileName
        with open(file_path) as file:
            lines = file.readlines()
            file.close()
        
        doctor_lines = ""
        patient_lines = ""
        
        for current_line in lines:
            #Skip blank lines
            if current_line == "\n":
                continue
            
            #Extracting the transcript only (wihtout the label)
            transcript = re.findall(r"[:;] \s*(.*)", current_line)
            if len(transcript) == 0:
                continue
            
            #Add the transcript to the doctor lines if Label is doctor (first character is "D")
            if current_line[0] == "D":
                doctor_lines = doctor_lines + " " + transcript[0]
            else:
                patient_lines = patient_lines + " " + transcript[0]
        
        #Add both the doctor and patient transcript to dataframe
        df.loc[len(df)] = ["Doctor", doctor_lines]
        df.loc[len(df)] = ["Patient", patient_lines]
        
print("Number of samples:", len(df))


# # Text Cleaning

# Text Processing Steps:
# 
# 1.) Lower Casing
# 
# 2.) Punctuation Removal
# 
# 3.) Tokenization
# 
# 4.) Stop Word Removal
# 
# 5.) Lemmatization

#Lower casing and punctuation removal
def text_clean(text):
    #Lower Casing
    text_clean = text.lower()
    
    #Remove punctuations
    punctuations = r"[^\w\s]"
    text_clean = re.sub(punctuations, " ", text_clean)

    #Tokenization
    text_clean = word_tokenize(text_clean)

    #Remove stop words
    stop_words = set(stopwords.words("english"))
    text_clean = [word for word in text_clean if word not in stop_words]

    #Lemmatization
    wn = nltk.WordNetLemmatizer()
    text_lemmatized = []
    for word in text_clean:
        text_lemmatized.append(wn.lemmatize(word))

    #Remove filler words like "um" "like" and basic greetings like "hi", "hey", "hello"
    filtered_list = []
    for token in text_lemmatized:
        if token in ["um", "like", "uhm", "uh", "hi", "hey", "hello"]:
            continue
        #also remove digits
        if token.isdigit():
            continue
            
        filtered_list.append(token)

    text_clean = filtered_list

    #Concat all tokens into a single string
    text_clean = " ".join(text_clean)

    return text_clean

df["Processed"] = df["Transcript"].apply(lambda x: text_clean(x))
df.head(15)


# # Pre- Processing

#Label Encoder to transform target
le = LabelEncoder()
le.fit(df["Target"])

target = le.transform(df["Target"])
le.classes_

#Split train test
x_train, x_test, y_train, y_test = train_test_split(df["Processed"], target, test_size = 0.25, random_state = 1, shuffle = True)

#Feature extraction (Text to numeric) using TF-IDF
tfidf = TfidfVectorizer(max_features = 1000,
                        lowercase = True,
                        analyzer = "word",
                        )

#Fit
tfidf_obj = tfidf.fit(x_train)
x_train_vector = tfidf_obj.transform(x_train.tolist())
x_test_vector = tfidf_obj.transform(x_test.tolist())

#Print the total number of features extracted
sample, features = x_train_vector.shape
print("Total number of features extracted: ", sample)

# # LSTM
#Preprocess text data by converting to tokens -> sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 5000, oov_token = "<OOV>")
tokenizer.fit_on_texts(x_train)

train_seq = tokenizer.texts_to_sequences(x_train)
test_seq = tokenizer.texts_to_sequences(x_test)

#Pad the sequences to ensure that they are all of the same length
max_length = max(len(seq) for seq in train_seq)

train_padded = pad_sequences(train_seq, maxlen = max_length, padding ='post')
test_padded = pad_sequences(test_seq, maxlen = max_length, padding ='post')

#BI-LSTM init
keras.utils.set_random_seed(0)
vocab_size = 5000
embedding_dim = 128

lstm_model = Sequential()
lstm_model.add(Embedding(vocab_size, embedding_dim, input_length = train_padded.shape[1]))

#First LSTM Layer
lstm_model.add(Bidirectional(LSTM(32, dropout = 0.5, return_sequences = False)))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.5))

#Dense Layer
lstm_model.add(Dense(32, activation = "relu"))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.5))

lstm_model.add(Dense(1, activation = 'sigmoid'))

lstm_model.compile(
    loss = 'binary_crossentropy',
    optimizer = Adam(learning_rate = 0.001),
    metrics = ['accuracy']
)

#Add early stopping to prevent overfitting during training
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2, restore_best_weights = True)

history = lstm_model.fit(
    train_padded,
    y_train,
    epochs = 15,
    batch_size = 16,
    validation_split = 0.2,
    callbacks = [early_stopping]
)

#Plotting Training and Validation Accuracy over epooch
plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], label = "Testing Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Testing Accuracy over Epoch")
plt.show()


# # Test

#Get the test predictions
###LSTM
lstm_predictions = lstm_model.predict(test_padded)
lstm_predictions = (lstm_predictions > 0.5).astype("int")

from sklearn.metrics import accuracy_score
lstm_testscore = accuracy_score(y_test, lstm_predictions)
print("LSTM Test accuracy score is: ", lstm_testscore * 100)

#Classification Reports
from sklearn.metrics import classification_report
classes = dict(zip(le.transform(le.classes_), le.classes_,))

lstm_classificationreport = classification_report(y_test, lstm_predictions, target_names = classes.values())
print("LSTM Classification Report: \n", lstm_classificationreport)

#Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, lstm_predictions)
plt.figure(figsize = (7, 5))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = "Blues", cbar = False, 
            xticklabels = classes.values(), yticklabels = classes.values())
plt.title("LSTM Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

#Create a table of performance metrics (Accuracy, recall, precision, F1-Score) of all the models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

columns = ["Accuracy", "Recall", "Precision", "F1-Score"]
scores = pd.DataFrame(columns = columns)

Accuracy = accuracy_score(y_test, lstm_predictions)
Recall = recall_score(y_test, lstm_predictions, average = "macro")
Precision = precision_score(y_test, lstm_predictions, average = "macro")
F1 = f1_score(y_test, lstm_predictions, average = "macro")
row = [Accuracy * 100, Recall * 100, Precision * 100, F1 * 100]
#Store the results of the model in the dataframe
scores.loc["LSTM"] = row

scores.style.set_caption("Performance Scores of LSTM")

#Create the ROC-AUC
from sklearn.metrics import RocCurveDisplay
lstm_predict_proba = lstm_model.predict(test_padded)
ROC_curve = RocCurveDisplay.from_predictions(
    y_test.ravel(),
    lstm_predict_proba.ravel(),
    name = "LSTM",
    #Plot the chance level
    plot_chance_level = (True if 0 == 0 else False),
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC of LSTM model")
plt.show()