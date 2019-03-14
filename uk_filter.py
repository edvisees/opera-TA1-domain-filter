from sklearn.datasets import fetch_20newsgroups
import os
import codecs
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
import pickle as pkl
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from stop_words import get_stop_words
import gc
from sklearn.metrics import f1_score
ru_stopWords = get_stop_words('uk')
def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6
def load_data(folder, label):
    data = []
    labels = []
    for file in os.listdir(folder):
        with codecs.open(os.path.join(folder, file), encoding='utf-8') as f:
            content = "".join([line for line in f])

            data.append(content)
            labels.append(label)
    return data, labels


pos_data, pos_label = load_data("data/pos/uk", 1)
print len(pos_data)
neg_data, neg_label = load_data("data/neg/uk", 0)
print len(neg_data)
neg_data = neg_data[:len(pos_data)]
neg_label = neg_label[:len(pos_data)]
#trace_data, trace_label = load_data("data/relevant_documents/english", 1)
#trace_data = np.array(trace_data)
#trace_label = np.array(trace_label)
#
print('split')
all_data = []
all_data.extend(pos_data + neg_data)
all_labels = []
all_labels.extend(pos_label + neg_label)
print len(all_labels), len(all_data)
all_data = np.array(all_data)
all_labels = np.array(all_labels)
print('split')
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

print('split')
idx = 0
batch_size = 64
num_classes = 2
epochs = 5

filepath="uk_best.hdf5"   

#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

for train_index, test_index in sss.split(all_data, all_labels):
    X_train, X_test = all_data[train_index], all_data[test_index]
    y_train, y_test = all_labels[train_index], all_labels[test_index]
    y_f1_test = y_test
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('feature extraction')
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=ru_stopWords,
    							 analyzer='word', max_features=30000)
    #with open('tfidf0.pkl') as f:
        #clf = pkl.load(f)
    #clf = MLPClassifier(solver='adam', alpha=1e-5, activation='tanh',
                    #hidden_layer_sizes=(5000, 500, 50), random_state=1,
                    #learning_rate_init=.05, verbose=10, tol=1e-4, max_iter=20)
    #clf = SVC(class_weight='balanced', C=2.5)
    print('training processing')
    X_train_fea = vectorizer.fit_transform(X_train)
    print X_train_fea[1].shape
    model = Sequential()
    model.add(Dense(4096, activation='tanh', input_shape=(X_train_fea.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=False)
    callbacks_list = [checkpoint]
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
                  #)
    X_test_fea = vectorizer.transform(X_test)
    history = model.fit(X_train_fea, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test_fea, y_test),
                    callbacks=callbacks_list)
    
    score = model.evaluate(X_test_fea, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    predicte = model.predict_classes(X_test_fea)
    print predicte
    print f1_score(y_f1_test, predicte, average=None)  
    with open(''.join(('tfidf', str(idx), 'uk.pkl')), 'w') as f:
        pkl.dump(vectorizer, f)
    '''
    clf.fit(X_train_fea, y_train) 
    X_test_fea = vectorizer.transform(X_test)
    predicted = clf.predict(X_test_fea)
    print predicted
    print(np.mean(y_test==predicted))
    #trace_fea = vectorizer.transform(trace_data)
    #predicted = clf.predict(trace_fea)
    #print(np.mean(trace_label==predicted))
    with open(''.join(('tfidf', str(idx), 'uk.pkl')), 'w') as f:
        pkl.dump(vectorizer, f)
    with open(''.join(('filter', str(idx), 'uk.pkl')), 'w') as f:
        pkl.dump(clf, f)
    '''
