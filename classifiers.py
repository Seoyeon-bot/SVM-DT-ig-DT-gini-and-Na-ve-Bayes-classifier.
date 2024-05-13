print(__doc__)

###############################     INSTALLATION/PREP     ################
# This is a DEMO to demonstrate the classifiers we learned about
# in CSI 431/531 @ UAlbany
#
# Might need to install the latest scikit-learn
# On linux or Mac: sudo pip install -U scikit-learn
#
# Codebase with more classifiers here: 
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

###############################     IMPORTS     ##########################

# numeric python and plotting
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# utility to help you split training data
from sklearn.model_selection import train_test_split
# utility to standardize data http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler
# some dataset generation utilities. for example: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
from sklearn.datasets import make_moons, make_circles, make_classification

# Scoring for classifiers
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, confusion_matrix

# Classifiers from scikit-learn
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Bayesian
from sklearn.naive_bayes import GaussianNB
# kNN
from sklearn.neighbors import KNeighborsClassifier
# DT
from sklearn.tree import DecisionTreeClassifier
# SVM: linear and a kernel-SVM (you can read more about it in the SVM chapter)
from sklearn.svm import SVC
# AdaBoost classifier (we talked about it today)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score


import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=FutureWarning)


###############################     CLASSIFIERS     #######################

# Put the names of the classifiers in an array to plot them
names = ["LDA",
         "Naive Bayes",
         "Nearest Neighbors", 
         "Linear SVM", 
         "RBF SVM",
         "Decision Tree",
         "AdaBoost"
         ]

# Create the classifiers with respective parameters
# LDA, NB: No parameters
# kNN:     k=3 for kNN (i.e. 3NN)
# SVM:     One linear and with C=0.025 and one RBF kernel-SVM with C=1
# DT :     Limit depth to 5 (i.e. at most 5 consecutive splits in each decision rule)
classifiers = [
    LinearDiscriminantAnalysis(),
    GaussianNB(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.25), # SVM function , C : regularization parameter. 
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=2),
    AdaBoostClassifier()
    ]


###############################     DATASETS     ##########################

# prepare a linearly separable dataset http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
# add some noise to points
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
# call this our linearly separable dataset
linearly_separable = (X, y)

# put our datasets in an array
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]


###############################  TRAIN AND PLOT  ##########################

figure = plt.figure(figsize=(27, 9))
i = 1
# Iterate over datasets and train and plot each classifier
for ds_cnt, ds in enumerate(datasets):
    # Preprocess dataset, split into training and test part
    X, y = ds
    # Standardize - make all feature to be on the same scale. -> improve model performance 
    X = StandardScaler().fit_transform(X)  # StandardScalar() : standardize features by removing mean. and scalling to unit variance. 
    # Splits our dataset in training and testing
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42) # shuffle data , train_test_split() split dataset into training and testing dataset (80% , 20%) 
    
    # take the min and max for both dimensions to help us with plotting
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
        
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        # Prepare the plot for this classifier
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        # Train the classifier (all classifiers in Scikit implement this functions)
        clf.fit(X_train, y_train)  # fit classifier to the training data using .fit()
        # Predict
        y_pred = clf.predict(X_test) # predidct output for X test data using .predict 
        acc = accuracy_score(y_test, y_pred) # evaluate accuracy 
        ap = average_precision_score(y_test, y_pred) # evaluate average precision 
        rec = recall_score(y_test, y_pred, average='weighted') 
        f1 = f1_score(y_test, y_pred, average='weighted')  # f1_score() calcualtes f1 scores, measure models' performance -> useful for evaluating classifiers. 
        
        #score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() - .32, 'A=' + ('%.2f' % acc).lstrip('0') + ' P=' + ('%.2f' % ap).lstrip('0') + ' R=' + ('%.2f' % rec).lstrip('0') + ' F=' + ('%.2f' % f1).lstrip('0') ,
                size=11, horizontalalignment='right')
        i += 1
plt.tight_layout()
plt.show()


# for pat3 - (a)
"""_load the dataset from cancer-data-train.csv 
    perform 10-fold cross-validation for 
    for linear SVMs with values of C = {0.01, 0.1, 1, 10, 100}.
    (different values of the regularization parameter C )
    in a linear SVM classifier.
    x axis : valuess of C 
    y axis : corresponding avg F measure. 
    _
"""
# load dataset from ccancer data trian.csv  
data = pd.read_csv('cancer-data-train.csv') # use panda library to read data from .csv 

values_Of_C = [0.01, 0.1, 1, 10, 100]  # given values_Of_C

# iterate over list of C and get average F for each of corresponding C
def SVM_cross_validation( values_Of_C, X,y ) : 
    list_of_avg_f1 = [] # Initialize a list to store the average F1-measure for each value of C

    for c in values_Of_C :  # iterate over list of C's 
         linear_svm_classifier = SVC(kernel='linear', C=c) # create linear SVM calssifier for each C 
         # Perform 10-fold cross-validation and obtain F-measure for each fold
         f1_scores = cross_val_score(linear_svm_classifier, X, y, cv=10, scoring='f1_macro')
        # Calculate average F-measure across all folds
         avg_f1 = np.mean(f1_scores)
         list_of_avg_f1.append(avg_f1);  # append to list 
    return list_of_avg_f1

# for part3 - c to pcik best c value from linear svm
best_c_value = None; 
best_avg_f1_measure = 0 ; 

# Iterate through the datasets

for  X, y in enumerate(data): # ds_index, 
    # seperate 30 features and let 31th last colum as y label. 
    X = data.iloc[:, :-1]  # All columns except the last one (assuming last column is labels)
    y = data.iloc[:, -1]   # Last column (labels)

    # call functon to get average f1 score for each c in list of given C 
    list_of_avg_f1 = SVM_cross_validation(values_Of_C, X,y)
    
    for i in range(len(values_Of_C)):  # iterate over length of list c 
        c = values_Of_C[i] # get ith c in values of c list 
        avg_f1 = list_of_avg_f1[i] # get ith avg_F1 in list of avg f1 
        print("c : " , c , " average f1 measure value : " , avg_f1)
        
        # find best C value and correspondingaverage F1 measure -> use this value in part 3 - c 
        if avg_f1 > best_avg_f1_measure: 
            best_avg_f1_measure = avg_f1; 
            best_c_value = c; 
    print("best c value from linear SVM part(a) : " , best_c_value)
    print("corresponding average f measure value : " , best_avg_f1_measure )   
        
        
# Plot the each c values in x axis, corresponding avg-f1 measures in y axis. 
plt.figure()
plt.plot(values_Of_C, list_of_avg_f1, marker='o')
plt.xscale('log')
plt.xlabel('c parameter ')
plt.ylabel('Average f1 measures ')
plt.title('Average f1 measures for each c parameter value in Linear SVM')
plt.show()

print("\nStart part3 - b task\n")   
# part 3 - b 
data = pd.read_csv('cancer-data-train.csv')

# Separate features and labels
X_train = data.iloc[:, :-1]  # attributes are going to be X_train 
y_train = data.iloc[:, -1]   # classifier is going to be y_train 

# Define maximum leaf node k
k_values = [2, 5, 10, 20]

# Initialize lists to store average F-measures for DT-gini and DT-ig
average_f1_for_gini = []
average_f1_for_IGain = []

# for part3 - c to get best side of tree for DT ig  and DT gini
def train_dt_gini_ig_tree(X_train, y_train, k_values): 
    best_size_k_ig = None
    best_size_k_gini = None
    best_avg_f1_measure_ig = 0
    best_avg_f1_measure_gini = 0
    
    for k in k_values: # iterate over given k values 2, 5, 10, 20 
        # get Train DT-gini
        DT_gini = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=k)
        predicted_labels_gini = cross_val_predict(DT_gini, X_train, y_train, cv=10)  # cv = 10 meaning we are doing 10 old cross validation on DT-gini as function of k 
        avg_f1_gini = f1_score(y_train, predicted_labels_gini, average='macro')
        average_f1_for_gini.append(avg_f1_gini)  # Append average F1 for DT-gini
       
        # get Train DT-ig
        DT_ig = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=k)
        predicted_labels_ig = cross_val_predict(DT_ig, X_train, y_train, cv=10)
        avg_f1_ig = f1_score(y_train, predicted_labels_ig, average='macro')
        average_f1_for_IGain.append(avg_f1_ig)  # Append average F1 for DT-ig 
        
         # Calculate average F1 measure for the current k
        avg_f1_gini = np.mean(average_f1_for_gini)
        avg_f1_ig = np.mean(average_f1_for_IGain)
        
     # Compare with previous best and update if necessary
        if avg_f1_gini > best_avg_f1_measure_gini:
            best_avg_f1_measure_gini = avg_f1_gini
            best_size_k_gini = k
        
        if avg_f1_ig > best_avg_f1_measure_ig:
            best_avg_f1_measure_ig = avg_f1_ig
            best_size_k_ig = k
    
     # Print the best sizes of tree for DT-ig and DT-gini and their corresponding average f1 measures
    print("Best size of tree for DT-ig:", best_size_k_ig)
    print("Corresponding average f1 measure for DT-ig:", best_avg_f1_measure_ig)
    print("Best size of tree for DT-gini:", best_size_k_gini)
    print("Corresponding average f1 measure for DT-gini:", best_avg_f1_measure_gini)

        

# call train_dt_gini_ig_tree function 
train_dt_gini_ig_tree(X_train, y_train, k_values)

# Plotting
plt.plot(k_values, average_f1_for_gini, color='black', label='DT-gini')
plt.plot(k_values, average_f1_for_IGain, color='red', label='DT-ig')
plt.xlabel('Provided maximum leaf node k = { 2, 5, 10, 20 }')
plt.ylabel('corresponding F measures ')
plt.title('For DT-gini and DT-ig trees')
plt.legend()
plt.show()



# part 3 - c 
# use best_c_value from part (a) , use best_size_k_ig,  best_size_k_gini from part(b)

# use train.sv for training svm, DTig, DTgini, naive bayes 
print("\nStart part3 - c task\n")

train_data = pd.read_csv('cancer-data-train.csv')
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# extract training sets from test.csv 
testData = pd.read_csv('cancer-data-test.csv')
X_test = testData.iloc[:, :-1]
y_test = testData.iloc[:, -1]

print("Size of X_train:", X_train.shape)
print("Size of y_train:", y_train.shape)
print("Size of X_test:", X_test.shape)
print("Size of y_test:", y_test.shape)

X_test = X_test.values  # convert X_test to array type 

# train SVM classifier and get predicted value 
# since i know best_c_value from part(a) is 0.01, I will use c = 0.01 
svm_clf = SVC(kernel='linear', C=0.01 ); # best_c_value)
svm_pred = svm_clf.fit(X_train, y_train).predict(X_test) # predict based on X_test value 
print("got svm_pred : ", svm_pred); 


# train DT gini classifier and get predicted value 
# I can use  max_leaf_nodes= best_size_k_gini since I know best_size_k_gini = 20, I will use 20. 
DT_gini_clf = DecisionTreeClassifier(criterion='gini', max_leaf_nodes= 20 ) # best_size_k_gini ) # get k from part(b)
DT_gini_pred = DT_gini_clf.fit(X_train, y_train).predict(X_test)
print("got DT_gini_pred : ", DT_gini_pred); 

# train DTig classifier 
# I can use  max_leaf_nodes= best_size_k_ig since I know best_size_k_ig = 20, I will use 20. 
DT_ig_clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=20); # best_size_k_ig) # gte k from part(b)
DT_ig_pred = DT_ig_clf.fit(X_train, y_train).predict(X_test)
print("got DT_ig_pred : ", DT_ig_pred) ; 

# train naive bayes classifier and get predicted value 
naiveBayes_clf = GaussianNB(); 
naiveBayes_pred = naiveBayes_clf.fit(X_train, y_train).predict(X_test); 
print("got naiveBayes_pred : ", naiveBayes_pred ) ; 


# compute average class recision, average calass recall, average class F measure 
# list to hold precision, recall f1 
precision_list = [];  recall_list = []; f1_list =[]; 

# Compute confusion matrices citation 
svm_conf_matrix = confusion_matrix(y_test, svm_pred)
DT_gini_conf_matrix = confusion_matrix(y_test, DT_gini_pred)
DT_ig_conf_matrix = confusion_matrix(y_test, DT_ig_pred)
naiveBayes_conf_matrix = confusion_matrix(y_test, naiveBayes_pred)

print("svm_confusin matrix : " , svm_conf_matrix); 
print("DT_gini_conf_matrix : " , DT_gini_conf_matrix); 
print("DT_ig_conf_matrix : " , DT_ig_conf_matrix); 
print("naiveBayes_conf_matrix: " , naiveBayes_conf_matrix); 


# compute metrics for SVM classifier 
svm_precision = precision_score(y_test , svm_pred, average='macro')
svm_recall = recall_score(y_test, svm_pred, average='weighted')  # weighted or macro 
svm_f1 = f1_score(y_test, svm_pred, average='weighted') # weighted or macro depends on dataset
precision_list.append(svm_precision); 
recall_list.append(svm_recall); 
f1_list.append(svm_f1); 

# Compute metrics for DT-gini
DT_gini_precision = precision_score(y_test, DT_gini_pred, average='macro')
DT_gini_recall = recall_score(y_test, DT_gini_pred, average='weighted')
DT_gini_f1 = f1_score(y_test, DT_gini_pred, average='weighted')
precision_list.append(DT_gini_precision); 
recall_list.append(DT_gini_recall); 
f1_list.append(DT_gini_f1); 

# Compute metrics for DT-ig
DT_ig_precision = precision_score(y_test, DT_ig_pred, average='macro')
DT_ig_recall = recall_score(y_test, DT_ig_pred, average='weighted')
DT_ig_f1 = f1_score(y_test, DT_ig_pred, average='weighted')
precision_list.append(DT_ig_precision); 
recall_list.append(DT_ig_recall); 
f1_list.append(DT_ig_f1); 

# Compute metrics for Naive Bayes
naiveBayes_precision = precision_score(y_test, naiveBayes_pred, average='macro')
naiveBayes_recall = recall_score(y_test, naiveBayes_pred, average='weighted')
naiveBayes_f1 = f1_score(y_test, naiveBayes_pred, average='weighted')
precision_list.append(naiveBayes_precision); 
recall_list.append(naiveBayes_recall); 
f1_list.append(naiveBayes_f1); 

# plot 3 bar chart for each of 4 classifier 
bar_width = 0.25
clf_list = ['SVM', 'DT_ig', 'DT_gini', 'Naive-Bayes']; 

# calculate position 
bar_pos1 = np.arange(len(clf_list))  # for preision 
bar_pos2  = bar_pos1 + bar_width  # for recall 
bar_pos3 = bar_pos2 + bar_width   # for f measure 

# Create bars
plt.bar(bar_pos1, precision_list, width=bar_width, label='Precision', color='red')
plt.bar(bar_pos2, recall_list, width=bar_width, label='Recall', color='blue')
plt.bar(bar_pos3, f1_list, width=bar_width, label='F1-measure', color='orange')

# Add labels to axes
plt.xlabel('Classifiers')
plt.ylabel('Average Metrics')
plt.xticks(bar_pos2, clf_list)

# Add legend
plt.legend()

# Show plot
plt.show()