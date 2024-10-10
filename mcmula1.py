# Author: Alex McMullen, based off of support_vector_machine.py  and Assignment_2-starter.py created by Swati Mishra

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split, KFold 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn import svm
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



class my_svm():
    # __init__() function should initialize all your variables
    def __init__(self,learning_rate,epoch,C_value, year_range):
        #initialize the variables
        self.learning_rate =learning_rate
        self.epoch = epoch
        self.C = C_value
        self.year_range = year_range

    # preprocess() function:
    #  1) normalizes the data, 
    #  2) removes missing values
    #  3) assign labels to target 
    def preprocess(self,):

        # import data based on year range
        goes_data= np.load("data-" + self.year_range + "/goes_data.npy", allow_pickle=True)
        self.data_order= np.load("data-" + self.year_range + "/data_order.npy")
        pos_class= np.load("data-" + self.year_range + "/pos_class.npy", allow_pickle=True)
        pos_features_historical = np.load("data-" + self.year_range + "/pos_features_historical.npy")
        pos_features_main_timechange = np.load("data-" + self.year_range + "/pos_features_main_timechange.npy")
        pos_features_maxmin = np.load("data-" + self.year_range + "/pos_features_maxmin.npy")
        neg_class = np.load("data-" + self.year_range + "/neg_class.npy", allow_pickle=True)
        neg_features_historical = np.load("data-" + self.year_range + "/neg_features_historical.npy")
        neg_features_main_timechange = np.load("data-" + self.year_range + "/neg_features_main_timechange.npy")
        neg_features_maxmin = np.load("data-" + self.year_range + "/neg_features_maxmin.npy")


        #make labels for target
        POSY = np.ones(len(pos_class))
        NEGY = -np.ones(len(neg_class))
        self.posy = (np.column_stack(POSY)).T
        self.negy = (np.column_stack(NEGY)).T

        #set up feature sets
        feature_set_1 = np.concatenate((pos_features_main_timechange[:,:18], neg_features_main_timechange[:,:18]), axis=0)
        feature_set_2 = np.concatenate((pos_features_main_timechange[:,18:], neg_features_main_timechange[:,18:]), axis=0)
        feature_set_3 = np.concatenate((pos_features_historical, neg_features_historical), axis=0)
        feature_set_4 = np.concatenate((pos_features_maxmin[:,:18], neg_features_maxmin[:,:18]), axis=0)

        #combine feature sets into one to normalize
        FEA = np.concatenate((feature_set_1, feature_set_2, feature_set_3, feature_set_4), axis=1)

        #normalize features
        scalar = StandardScaler().fit(FEA)
        normalized_features = scalar.transform(FEA)

        #look for missing values
        #make array of values true false, true => missing value
        nan_array = np.isnan(normalized_features)
        #loop through this array and remove row from normalized fatures if it has a missing value
        for i, j in np.ndindex(nan_array.shape):
            if (nan_array[i,j] == True):
                nan_array = np.delete(nan_array, i, 0)
                normalized_features = np.delete(normalized_features, i, 0)
                print('Missing value found')


        #split into 4 normalized feature sets
        norm_fs1 = normalized_features[:,:18]
        norm_fs2 = normalized_features[:,18:90]
        norm_fs3 = normalized_features[:,90:91]
        norm_fs4 = normalized_features[:,91:]
        
        self.feature_sets = [norm_fs1, norm_fs2, norm_fs3, norm_fs4]
    
    # feature_creation() function takes as input the features set label (e.g. FS-I, FS-II, FS-III, FS-IV)
    # and creates 2 D array of corresponding features 
    # for both positive and negative observations.
    # this array will be input to the svm model
    # For instance, if the input is FS-I, the output is a 2-d array with features corresponding to 
    # FS-I for both negative and positive class observations
    def feature_creation(self, fs_value):

        #create feature set depending on input
        if(fs_value == 'FS-I'):
            index = 0
        elif(fs_value == 'FS-II'):
            index = 1
        elif(fs_value == 'FS-III'):
            index = 2
        elif(fs_value == 'FS-IV'):
            index = 3
        else: 
            #make sure valid input has been given
            print('Error, please give a valid feature set')

        #select feature set
        FEATURES = self.feature_sets[index]
        #add bias
        bias = np.ones(len(FEATURES))
        #combine bias
        FEATURES = np.column_stack((FEATURES, bias))

        return FEATURES
    

    # the function return gradient for 1 instance -
    # stochastic gradient decent
    def compute_gradient(self,X,Y):
        # organize the array as vector
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y* np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))
        # hinge loss is not defined at 0
        # is distance equal to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y[0] * X_[0])

        return total_distance
    
    def feature_shuffle(self, X, Y):
        #create two matrices the same size as X and Y
        X_ = np.full_like(X, 0)
        Y_ = np.full_like(Y, 0)
        for i, order in enumerate(self.data_order):
            #set the order based on data_order.npy
            X_[i] = X[order]
            Y_[i] = Y[order]
            

        return X_, Y_
    

    # cross_validation() function splits the data into train and test splits,
    # Use k-fold with k=10
    # the svm is trained on trainining set and tested on test set
    # the output is the average accuracy across all train test splits.
    def cross_validation(self,X,Y):

        #shuffle data using data_order.npy
        X, Y = self.feature_shuffle(X, Y)

        #initialize the weight matrix based on number of features 
        self.weights = np.zeros(X.shape[1])
        #initialize lists
        self.predictions = []
        self.y_tests = []
        self.tss_list = []
        accuracy_list = []


        #KFold cv with k = 10
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            
            #comment this out to use the custom svm
            #--------------------------------------
            # Fit the SVM model with sklearn
            model = svm.SVC(kernel='linear')
            model.fit(X_train, y_train.ravel())
            # Predict using the SVM model
            predictions = model.predict(X_test)
            #--------------------------------------

            #uncomment this part to use the custom svm
            #--------------------------------------
            # call training function
            #weights = self.training(X_train,y_train)
            #make prediction
            #predictions = self.predict(X_test, weights)
            #--------------------------------------
            
            #add each prediction to list
            self.predictions.append(predictions)
            self.y_tests.append(y_test.ravel())

            # call tss function
            tss = self.tss(predictions, y_test)

            #add each tss to list
            self.tss_list.append(tss)

            #compute accuracy and add to list of accuracies for each fold
            accuracy_list.append(accuracy_score(y_test, predictions))

        #find best fold index for confusion matrix using TSS
        self.best_fold_index = self.tss_list.index(max(self.tss_list))
        #calculate mean, std of accuracy list
        mean = sum(accuracy_list) / len(accuracy_list)
        variance = sum((x - mean) ** 2 for x in accuracy_list) / len(accuracy_list)
        std_dev = variance ** 0.5
        #calculate average TSS
        avg_tss = sum(self.tss_list) / len(self.tss_list)
        
        #return average tss, mean, standard deviation across the 10 folds
        return avg_tss, mean, std_dev
    
    #training() function trains a SVM classification model on input features and corresponding target
    def training(self,X,Y):

        # execute the stochastic gradient descent function for defined epochs
        for epoch in range(self.epoch):

            features = X
            output = Y

            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)

        return self.weights
    
    def predict(self,X_test, weights):

        #compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], weights)) for i in range(X_test.shape[0])]
        return predicted_values
    
    # tss() function computes the accuracy of predicted outputs (i.e target prediction on test set)
    # using the TSS measure given in the document
    def tss(self, y_predicted, y_test):
        #initialize variables
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        y_predicted = (np.column_stack(np.array(y_predicted))).T
        #loop through predicted values and compare to test values
        #classify each result as True Negative, False Negative, True Positive, False Positive
        for i, j in np.ndindex(y_predicted.shape):
            if (y_predicted[i,j] == -1 and y_test[i,j] == -1):
                tn = tn + 1
            if (y_predicted[i,j] == -1 and y_test[i,j] == 1):
                fn = fn + 1
            if (y_predicted[i,j] == 1 and y_test[i,j] == 1):
                tp = tp + 1
            if (y_predicted[i,j] == 1 and y_test[i,j] == -1):
                fp = fp + 1

    
        #return TSS measure using formula from document
        return (tp / (tp + fn)) - (fp / (fp + tn))

#this function returns all possible combinations of list elements
def power_set(fs_list):
    count = len(fs_list)
    power_set = []
    for i in range(1, 1 << count):
        power_set.append([fs_list[j] for j in range(count) if (i & (1 << j))])
    return power_set
    

# feature_experiment() function executes experiments with all 4 feature sets.
# svm is trained (and tested) on 2010 dataset with all 4 feature set combinations
# the output of this function is : 
#  1) TSS average scores (mean std) for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document 
#  3) A chart showing TSS scores for all folds of CV. 
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e 10)
#
# Above 3 charts are produced for all TSS combinations
#  4) The function prints the best feature set combination
def feature_experiment():
    #initialize possible feature sets and best accuracy
    #we choose -1 for initial accuracy because this is the worst possible TSS
    possible_feature_sets = power_set(['FS-I', 'FS-II', 'FS-III', 'FS-IV'])
    best_accuracy = -1

    #initialize SVM
    svm1 = my_svm(learning_rate=0.01,epoch=500,C_value=0.1, year_range = '2010-15')
    svm1.preprocess()
    POSY = svm1.posy
    NEGY = svm1.negy
    #crete target
    target = np.concatenate((POSY, NEGY), axis=0)

    #loop through each possible feature set
    for feature_set in possible_feature_sets:
        #initialize TSS list
        tss_final_list = []



        #create feature set using feature creation, depending upon the number of combined feature sets
        #Initialize feature set as the first set in the combination
        features = svm1.feature_creation(feature_set[0])
        #if there is more than one feature set in this combination, append each additional feature set recursively
        if(len(feature_set) > 1):
            for set in feature_set[1:]: 
                features = np.concatenate((features, svm1.feature_creation(set)), axis = 1)

        #compute average TSS, mean, std dev for this feature set combination
        avg_tss1, mean1, std_dev1 = svm1.cross_validation(features, target)

        #store the TSS of each fold in a list
        tss_final_list.append(svm1.tss_list)

        #print the average TSS for the feature set
        print('Average TSS for ' + str(feature_set) + ': ' + str(avg_tss1))
        #print the mean accuracy for the feature set
        print('Mean Accuracy for ' + str(feature_set) + ': ' + str(mean1))
        #print the standard deviation for the feature set
        print('Standard Deviation for ' + str(feature_set) + ': ' + str(std_dev1))
        print('--------------------------------------------------------------')
        
        #find the feature set with the highest TSS
        if(avg_tss1 > best_accuracy):
            best_accuracy = avg_tss1
            best_fs = feature_set

        #create chart 2)
        #use the prection and test ys from the best fold as input for confusion matrix
        best_test_ys = svm1.y_tests[svm1.best_fold_index]
        best_prediction = svm1.predictions[svm1.best_fold_index]

        #create confusion matrix
        cm = confusion_matrix(best_test_ys, best_prediction, labels=[-1,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=[-1,1])
        disp.plot()
        plt.title('Best Fold Confusion Matrix for ' + str(feature_set))
        plt.show()
        
        #create chart 3)
        fig, ax = plt.subplots()
        fig.set_size_inches((15,8))

        #plot the TSS from each fold
        for i, ys in enumerate(tss_final_list):
            xs = [x for x in range(len(ys))]
            ax.plot(xs,ys, color='r', alpha = 0.6)

        #set titles
        ax.set_title('SVM TSS Scores by Fold using ' + str(feature_set))
        ax.set_xlabel('Fold')
        ax.set_ylabel('TSS Score')
        ax.set_ylim([-1,1])
        plt.show()


        
    #output the best feature set combination
    print('Best Feature Set Combination: ' + str(best_fs))
    return best_fs 

# data_experiment() function executes 2 experiments with 2 data sets.
# svm is trained (and tested) on both 2010 data and 2020 data
# the output of this function is : 
#  1) TSS average scores for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document 
#  3) A chart showing TSS scores for all folds of CV. 
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e. 10)
# above 3 charts are produced for both datasets
# feature set for this experiment should be the 
# best performing feature set combination from feature_experiment()
#
def data_experiment(best_fs):
    #set both year ranges
    year_ranges = ['2010-15', '2020-24']

    #loop through each yearrange
    for year in year_ranges:

        #initialize TSS list
        tss_final_list = []

        #initialize SVM
        svm1 = my_svm(learning_rate=0.01,epoch=500,C_value=0.1, year_range = year)
        svm1.preprocess()
        POSY = svm1.posy
        NEGY = svm1.negy
        #create target
        target = np.concatenate((POSY, NEGY), axis=0)

        #create feature set uing feature creation, depending upon the number of combined feature sets
        #Initialize feature set as the first set in the combination
        features = svm1.feature_creation(best_fs[0])
        #if there is more than one feature set in this combination, append each additional feature set recursively
        if(len(best_fs) > 1):
            for set in best_fs[1:]: 
                features = np.concatenate((features, svm1.feature_creation(set)), axis = 1)

        #compute average TSS, mean and standard deviation for this feature set combination
        avg_tss1, mean1, std_dev1 = svm1.cross_validation(features, target)

        #store the TSS of each fold in a list
        tss_final_list.append(svm1.tss_list)

        #print the average TSS for the feature set
        print('Average TSS for ' + str(best_fs) + ' and ' + year + ' dataset: ' + str(avg_tss1))
        #print the mean accuracy for the feature set
        print('Mean accuracy for ' + str(best_fs) + ' and ' + year + ' dataset: ' + str(mean1))
        #print the standard deviation for the feature set
        print('Standard Deviation for ' + str(best_fs) + ' and ' + year + ' dataset: ' + str(std_dev1))
        print('--------------------------------------------------------------')

        
        #create chart 2)
        #use the prection and test ys from the best fold as input for confusion matrix
        best_test_ys = svm1.y_tests[svm1.best_fold_index]
        best_prediction = svm1.predictions[svm1.best_fold_index]

        #create confusion matrix
        cm = confusion_matrix(best_test_ys, best_prediction, labels=[-1,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=[-1,1])
        disp.plot()
        plt.title('Best Fold Confusion Matrix for ' + str(best_fs) + ' and ' + year)
        plt.show()

        #create chart 3)
        fig, ax = plt.subplots()
        fig.set_size_inches((15,8))

        #plot the TSS from each fold
        for i, ys in enumerate(tss_final_list):
            xs = [x for x in range(len(ys))]
            ax.plot(xs,ys, color='b', alpha = 0.6)

        #set title
        ax.set_title('SVM TSS Scores by Fold using ' + str(best_fs) + ' and ' + year + ' dataset')
        ax.set_xlabel('Fold')
        ax.set_ylabel('TSS Score')
        ax.set_ylim([-1,1])
        plt.show()



    return 0

# below should be your code to call the above classes and functions
# with various combinations of feature sets
# and both datasets

best_fs = feature_experiment()
data_experiment(best_fs)







        