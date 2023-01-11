import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

from imblearn.over_sampling import SMOTE
import scipy.stats as stats
from datetime import datetime 
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import scipy.stats as stats

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
def clustering(df):
    X_clus= df.drop(["icon"],axis=1)
    y_clus = df["icon"]
    colnames = X_clus.columns

    #Label encode
    label_encoder = preprocessing.LabelEncoder()
    for col in X_clus:
        if X_clus[col].dtype == "object" and col!="icon":
            X_clus[col]=label_encoder.fit_transform(X_clus[col])

    X_clus =stats.zscore(X_clus)
    

    neighbors = st.sidebar.number_input('Insert the maximum number of neighbors')
    sampling = st.sidebar.selectbox("Choose a sampling technique.",("None", "SMOTE", "Oversampling","Undersampling"))

    if st.sidebar.button("Run model", key='run'):
        if sampling == "None":
            none_option(neighbors,X_clus,y_clus)

        if sampling == 'SMOTE':
            smote_option(neighbors,X_clus,y_clus)


def none_option(neighbors, X_clus, y_clus):
            neighbors_range = np.arange(1, neighbors)
            train_accuracy = np.empty(len(neighbors_range))
            x_train, x_test, y_train, y_test = train_test_split(X_clus, y_clus, test_size=0.3, random_state=0)

            # Loop over different values of k
            for i, k in enumerate(neighbors_range):
                # Setup a k-NN Classifier with k neighbors: knn
                knn = KNeighborsClassifier(n_neighbors=int(k))

                # Fit the classifier to the training data
                cv_scores = cross_val_score(knn, X_clus, y_clus, cv=5)
    
                #Compute accuracy on the training set
                train_accuracy[i] = np.mean(cv_scores)

    

            # Generate plot
            fig = plt.figure()
            plt.title('k-NN: Varying Number of Neighbors')
            #plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
            plt.plot(neighbors_range, train_accuracy, label = 'Training Accuracy')
            plt.legend()
            plt.xlabel('Number of Neighbors')
            plt.ylabel('Accuracy')
            st.pyplot(fig)


            st.write("We ue Grid Search to verify that the best number of neighbors is the same as above")
            #create a dictionary of all values we want to test for n_neighbors
            param_grid = {'n_neighbors': np.arange(1, 40)}
            #use gridsearch to test all values for n_neighbors
            knn_gscv = GridSearchCV(knn, param_grid, cv=5)
            #fit model to data
            knn_gscv.fit(x_train, y_train)

            st.write("The best number of neighbor is ", knn_gscv.best_params_)
            st.write("The best score is ", knn_gscv.best_score_)

            st.write("Then we plot the confusion matrix, ROC Curve and precision recall curve uing the best number of neighbors")
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(knn_gscv.best_estimator_, x_test, y_test)
            st.pyplot()

            

            st.subheader("Precision Recall Curve")
            st.subheader("ROC Curve")


def smote_option(neighbors, X_clus, y_clus):
            neighbors_range = np.arange(1, neighbors)
            train_accuracy = np.empty(len(neighbors_range))
            oversample = SMOTE()
            X_clus, y_clus = oversample.fit_resample(X_clus, y_clus)
            x_train, x_test, y_train, y_test = train_test_split(X_clus, y_clus, test_size=0.3, random_state=0)

            # Loop over different values of k
            for i, k in enumerate(neighbors_range):
                # Setup a k-NN Classifier with k neighbors: knn
                knn = KNeighborsClassifier(n_neighbors=int(k))

                # Fit the classifier to the training data
                cv_scores = cross_val_score(knn, X_clus, y_clus, cv=5)
    
                #Compute accuracy on the training set
                train_accuracy[i] = np.mean(cv_scores)

    

            # Generate plot
            fig = plt.figure()
            plt.title('k-NN: Varying Number of Neighbors')
            #plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
            plt.plot(neighbors_range, train_accuracy, label = 'Training Accuracy')
            plt.legend()
            plt.xlabel('Number of Neighbors')
            plt.ylabel('Accuracy')
            st.pyplot(fig)


            st.write("We ue Grid Search to verify that the best number of neighbors is the same as above")
            #create a dictionary of all values we want to test for n_neighbors
            param_grid = {'n_neighbors': np.arange(1, 40)}
            #use gridsearch to test all values for n_neighbors
            knn_gscv = GridSearchCV(knn, param_grid, cv=5)
            #fit model to data
            knn_gscv.fit(x_train, y_train)

            st.write("The best number of neighbor is ", knn_gscv.best_params_)
            st.write("The best score is ", knn_gscv.best_score_)

            st.write("Then we plot the confusion matrix, ROC Curve and precision recall curve uing the best number of neighbors")
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(knn_gscv.best_estimator_, x_test, y_test)
            st.pyplot()

            

            st.subheader("Precision Recall Curve")
            st.subheader("ROC Curve")
