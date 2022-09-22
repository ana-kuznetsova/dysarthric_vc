from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import os
import numpy as np
from sklearn import svm

def evaluate(inp_dir, model='speaker_encoder'):
    if model=='speaker_encoder':
        X = np.load(os.path.join(inp_dir, 'X.npy'))
        y = np.load(os.path.join(inp_dir, 'Y.npy'))
        target_names = [f"spk {i}" for i in np.unique(y)]

        num_classes = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print(f'> Training classifier...')
        classifier = svm.SVC(kernel="linear", probability=True, random_state=42, verbose=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        print(classification_report(y_test, y_pred, target_names=target_names))
    
