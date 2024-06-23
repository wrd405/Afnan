Importing Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Embedding, Flatten, LSTM, Dense,Dropout, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
/opt/conda/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.24.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
Reading and Preparing Data
# Read data from csv file
Data = pd.read_csv('comments.csv')
#Make sure that all the reviews are in text format
Data['clean text'] = Data['clean text'].astype(str)
Data.head()
review_text	review_link	review_rating	review_timestamp	review_datetime_utc	review_likes	reviews_id	is Noise	pos_count	neg_list	sentiment	clean text
# No. of rows in each class
Data['sentiment'].value_counts()

#Remove the Neutral Class
Data = Data[Data['sentiment'] != "Neutral"]
#Converting each label to corresponding number
Data['sentiment'] =  Data['sentiment'].map({'Negative':0, 'Positive': 1})
#Printing the shape of the data
Data.shape

#Tokenization of the sequences in the data
tokenize = Tokenizer()
tokenize.fit_on_texts(Data['clean text'].values)
Data['clean text'] = tokenize.texts_to_sequences(Data['clean text'].values)
vocab_size = len(tokenize.word_index) + 1
print(vocab_size)

X = Data['clean text'].values.tolist()
y = Data['sentiment'].values.tolist()
#Compute class weights to give more importance to the minority class 
class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(y), y= y)
print(class_weights)

#Padding all sequences to be in the same length
max_len = 100
X = pad_sequences(X, padding='post', truncating='post', maxlen=max_len)
#convert y to numpy array
y = np.array(y)
#Splitting the data to train and text
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)
Model Building and Training
#Build 2 Layer LSTM model
model = Sequential()
model.add(Input(shape=(max_len,)))
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

#setting the optimization function, learning rate, loss and metrics of the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
#Use earlystopping to prevent overfitting
#Use Validation split to tweak the data to converge without using test data 
earlystopping = EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)
​
class_weights = {0: 1.99895519, 1: 0.66678284}
history = model.fit(X_train, y_train, epochs = 50, batch_size = 32, class_weight=class_weights, validation_split = 0.15, callbacks=earlystopping)

Model Evaluation
fig, axs = plt.subplots(2, 1, figsize=(15,15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(['Train', 'Val'])
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['Train', 'Val'])

#Calculate Test accuracy
score = model.evaluate(X_test, y_test) 
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predict X_test and compare the predicted results with original y_test to produce classification report
y_predict = model.predict(X_test)
y_predict = np.round(y_predict).astype(int)

#Calculate the precision
precision = precision_score(y_test, y_predict)
print(precision)

#Calculate the recall
recall = recall_score(y_test, y_predict)
print(recall)

#Calculate the f1-score
f1_score = f1_score(y_test, y_predict)
print(f1_score)

#Calculate the precision, recall and f1-score for each class
print(classification_report(y_test, y_predict))
              

#Save the Model
saved_model = model.save('ArabicReviews-DL.keras')
