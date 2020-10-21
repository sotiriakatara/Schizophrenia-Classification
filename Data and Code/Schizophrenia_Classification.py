import csv
import pandas 
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut


def readFile(choice):
        df_labels = pandas.read_csv('train_labels.csv')
        df_FNC = pandas.read_csv('train_FNC.csv')
        df_SBM = pandas.read_csv('train_SBM.csv')
        df_concat = pandas.concat([df_FNC, df_SBM], axis=1)
        df_real = df_concat.drop('Id', axis=1)  # cut id column
        # print(df_real)
        labels = df_labels['Class'].values
        # print(labels)

        # scaler = preprocessing.StandardScaler()
        # scaled_df = scaler.fit_transform(df_real)
        # scaled_df = pandas.DataFrame(scaled_df)

        scaler = preprocessing.Normalizer()
        scaled_df = scaler.fit_transform(df_real)
        scaled_df = pandas.DataFrame(scaled_df)
        # print(scaled_df)

        features = [data.values.tolist() for ind, data in df_real.iterrows()]
        # print(features)

        loo = LeaveOneOut()
        LeaveOneOut()
        find = 0
        for train_index, test_index in loo.split(np.array(df_real)):
                X_train = np.array(features)[train_index]
                X_test = np.array(features)[test_index]
                Y_train = np.array(labels)[train_index]
                Y_test = np.array(labels)[test_index]

                pca = PCA(n_components=80)
                principalComponents = pca.fit_transform(df_real)
                principalDF = pandas.DataFrame(data=principalComponents)
                # print(principalDF)

                if choice == "1":
                        model = GaussianNB()
                elif choice == "2":
                        model = SVC(kernel='linear', C=1)
                elif choice == "3":
                        model = SVC(kernel='sigmoid', C=1)     
                elif choice == "4":
                        model = SVC(kernel='poly', C=1)     
                elif choice == "5":
                        model = SVC(kernel='rbf', C=1, gamma=0.01)       
                else:
                        model = KNeighborsClassifier(n_neighbors=10)
                        
                ClassifierFit = model.fit(X_train, Y_train)
                find += ClassifierFit.score(X_test, Y_test)

        print('\nAccuracy:' + "{0:.2f}".format(find / len(df_real) * 100) + '\n')


def main():

        ans = True

        while ans:

                print("---------------------- MAIN MENU -----------------------")
                print("------------- SCHIZOPHRENIA CLASSIFICATION -------------")
                print("Please select the appropriate classification algorithm.")
                choice = input("""
1: Gaussian
2: SVM_Linear
3: SVM_Sigmoid
4: SVM_Poly
5: SVM_Rbf
6: KNeighbors
7: Quit/Log Out
Please enter your choice: """)

                if choice == "1":
                        print("You selected Gaussian Naive Bayes Classifier.")
                        readFile(choice)
                elif choice == "2":
                        print("You selected SVM Linear Classifier.")
                        readFile(choice)
                elif choice == "3":
                        print("You selected SVM Sigmoid Classifier.")
                        readFile(choice)
                elif choice == "4":
                        print("You selected SVM Poly Classifier.")
                        readFile(choice)
                elif choice == "5":
                        print("You selected SVM Non Linear Classifier.")
                        readFile(choice)
                elif choice == "6":
                        print("You selected KNeighbors.")
                        readFile(choice)
                elif choice == "7":  
                        ans = False
                        print("Bye Bye!")
                        sys.exit
                else:
                        print("You must only select either 1,2,3,4,6 or 7.")
                        print("Please try again")
                        main()   
               

if __name__ == "__main__":

    main()