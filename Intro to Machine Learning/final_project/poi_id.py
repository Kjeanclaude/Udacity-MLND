#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

###################################
### Task 1: Select what features you'll use.
###################################
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_payments','bonus','long_term_incentive','exercised_stock_options',
                 'from_this_person_to_poi','from_poi_to_this_person','from_messages','to_messages',
                 'fraction_from_poi','fraction_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    print("Number of people : ", len(data_dict))
##    print(data_dict)
################## CODE TO ADD TEN (10) POIs with a NaN value for 'total_payments' #################
names=["KK JC1", "KK JC2", "KK JC3", "KK JC4", "KK JC5", "KK JC6", "KK JC7", "KK JC8", "KK JC9", "KK JC10"]
add = {'salary': 274975, 'to_messages': 873, 'deferral_payments': 'NaN', 'total_payments': 'NaN',
         'exercised_stock_options': 384728, 'bonus': 600000, 'restricted_stock': 393818, 'shared_receipt_with_poi': 874,
         'restricted_stock_deferred': 'NaN', 'total_stock_value': 778546, 'expenses': 125978, 'loan_advances': 'NaN',
         'from_messages': 16, 'other': 200308, 'from_this_person_to_poi': 6, 'poi': True, 'director_fees': 'NaN',
         'deferred_income': 'NaN', 'long_term_incentive': 71023, 'email_address': 'ben.glisan@enron.com',
         'from_poi_to_this_person': 52}
### I defined two ways to do it
### 1- Editing the project file : final_project_dataset.pkl
### 'sg7' corresponds in the final_project_dataset.pkl file to the
### value of 'total_payments' feature. And 'g6' to 'NaN' value
### If we put 'g6' under 'sg7' in this file, its value will be 'NaN'
### A way to add the 10 poi with 'NaN' value for the 'total_payments' feature.
### 'sg21' corresponds 'poi' and 'I00' to a 'false' value
### 2- Or do it simply using the name as key of the dictionary
for elt in names:
    data_dict[elt]=add

#print(data_dict)
print("New Number of people : ", len(data_dict))
print " *************************** \n"
################  END CODE TO ADD TEN (10) POI with a NaN value for 'total_payments' #################



# ******************* NEW FEATURES ADDING FUNCTION, for Task 3 ******************
# As I created two new features in the task 3, I added them in the features list (features_list)
# in fact to take them into account now in task 2 (fraction_from_poi and fraction_to_poi)
# See task 3 for more details

# The fraction function
def computeFraction( poi_messages, all_messages ):
    fraction = 0
    if poi_messages=='NaN' or all_messages=='NaN':
        fraction = 0
    else:
        fraction = float(poi_messages)/float(all_messages)
    
    return fraction

for name in data_dict:
    data_point = data_dict[name]
    # Create and add the fraction_from_poi_to_this_person feature for this data point
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi

    # Create and add the fraction_from_this_person_to_poi feature for this data point
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    data_point["fraction_to_poi"] = fraction_to_poi


# ******************* END NEW FEATURES ADDING Task 3 ******************





###################################
### Task 2: Remove outliers
###################################
# outliers related to the higher value of bonus and salary
# I can see them
# 1- via plot
print "features_list : ",features_list
data = featureFormat(data_dict, features_list)
for point in data:
    salary = point[1]
    bonus = point[3]
    matplotlib.pyplot.scatter( salary, bonus )

print "\nPlease close the plot to continue ... \n"
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



# 2- via the corresponding key (name in data_dict)
# The names associated with those points bonuses of at least
# 5 million dollars, and a salary of over 1 million dollars  according to the plot (visualization)
print "\n #==> ********* OUTLIERS ************ "
dic={}
#value="bonus"
bonus = 0
Skeyvalue = ""
for value in data_dict.keys():
    Skeyvalue = value
    dic=data_dict[value]
    val=dic["bonus"]
    val2=dic["salary"]
    if val!="NaN":
        if int(val)>= 5000000 and int(val2)>= 1000000 :
            print("Associeted name : {} | Bonus : {} | Salary : {} |".format(Skeyvalue, val, val2))      
    dic={}
print " ************* END OUTLIERS ************** "
##### The code above provide these outliers #########
##Associeted name : LAY KENNETH L | Bonus : 7000000 | Salary : 1072321 |
##Associeted name : SKILLING JEFFREY K | Bonus : 5600000 | Salary : 1111258 |
##Associeted name : TOTAL | Bonus : 97343619 | Salary : 26704229 |

##So I will remove TOTAL which is a spreadsheet quirk
##And keep SKILLING JEFFREY K and LAY KENNETH L as valid datapoint because both are
## Eron's biggest bosses and are definitely POIs
del_value = data_dict.pop( "TOTAL" )



###################################
### Task 3: Create new feature(s)
###################################
# I will create here two features.
## *** fraction_from_poi_to_this_person (fraction_from_poi) *** : The fraction of all messages to this person that come from POIs
## This feature will help me to evaluate how related is this person from POIs. If he receives a great number of email from
## POIs, may be he's also a POI or work directly with POIs receiving instructions.

## *** fraction_from_this_person_to_poi (fraction_to_poi) *** : The fraction of all messages from this person that are sent to POIs
## This feature will help me to evaluate how related is this person to POIs. A great fraction of email to POIs means that
## he's directly and highly involved with POIs giving instructions and may be a POI.
# The fraction function (in commentary) below is exactly the same I used in the task 1 after adding these new features in the list
# features_list, in fact to be able to plot the outliers with a dictionary which includes them.
"""
# The fraction function
def computeFraction( poi_messages, all_messages ):
    fraction = 0
    if poi_messages=='NaN' or all_messages=='NaN':
        fraction = 0
    else:
        fraction = float(poi_messages)/float(all_messages)
    
    return fraction

for name in data_dict:
    data_point = data_dict[name]
    # Create and add the fraction_from_poi_to_this_person feature for this data point
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi

    # Create and add the fraction_from_this_person_to_poi feature for this data point
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    data_point["fraction_to_poi"] = fraction_to_poi
"""
    
# Uncomment this to see an example of datapoint with the new features
"""
fraction_test1 = data_dict["LAY KENNETH L"]
print "\n********* fraction_test1 ********* : ", fraction_test1
# 
fraction_test2 = data_dict["SKILLING JEFFREY K"]
print "********* fraction_test2 ********* : ", fraction_test2
"""

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



###################################
### Task 4: Try a varity of classifiers
###################################
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn import GaussianNB
from sklearn.naive_bayes import GaussianNB
clf_A = GaussianNB()

# from skearln import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
clf_B = DecisionTreeClassifier(max_depth=2, random_state=42)

# from sklearn import SVM
from sklearn import svm
# clf_C = svm.SVC(C=1000, degree=7, max_iter=1000, gamma=0.1, random_state=42)
clf_C = svm.SVC(C=1, gamma=0.1, random_state=42)
#clf_C = svm.SVC(random_state=42)

# To do
# from sklearn import PCA
from sklearn import svm
# Add the PCA as another Algorithm to explore



###################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
###################################
### Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Here, Choose one of the defined classifier
# *************************************************
# *************************************************
clf = clf_C
# *************************************************
# *************************************************

# Example starting point. Try investigating other evaluation techniques!
# Metrics importation 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Train - predict function, for training and metrics score's computing
def train_predict(clf, features_train, labels_train, features_test, labels_test):
    clf.fit(features_train, labels_train)
    y_pred = clf.predict(features_test)
    
    # pos_label=1, default value as we are looking for POI, poi=True
    #acc = clf.score(features_test, labels_test)
    acc = accuracy_score(labels_test, y_pred) #0.826086956522
    prec = precision_score(labels_test, y_pred, pos_label=1)
    rec = recall_score(labels_test, y_pred, pos_label=1)
    f1 = f1_score(labels_test, y_pred, pos_label=1)
    #f1_2 = fbeta_score(labels_test, y_pred, beta=1) #f1_score calculated with fbeta_score, it is exactly the same
    f2 = fbeta_score(labels_test, y_pred, beta=2) # Fix beta=2 for the F2_score
    print "\n#==> ************ LOCAL TESTS ***************"
    print "Classifier : {} \n\nAccuracy : {} \nPrecision : {} \nRecall : {} \nF1_Score : {} \nF2_Score : {} ".format(clf, acc, prec, rec, f1, f2)
    print "*************** LOCAL TESTS END ***************\n"    


def scorer_fnc(clf, features_test, labels_test):
    y_pred = clf.predict(features_test)
    f1 = f1_score(labels_test, y_pred, pos_label=1)

    return f1


# Most important feature
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
selector = SelectKBest(f_classif, k=11)
selector.fit_transform(features_train, labels_train)
print "\n#==> ******** FEATURES SCORES (Features importance) ********* "
#print "FEATURES SCORES (Features importance) :"
features_2=[]
for elt in features_list:
    features_2.append(elt)

features_2.remove('poi')

print "Features : \n",features_2
print "\nFeatures scores_ : \n",selector.scores_
print "***************** END Features importance *************** \n"

# First train-predict test without parameter's tunning
train_predict(clf, features_train, labels_train, features_test, labels_test)

# ********* START TUNING ***********
# Tuning with GridSearchCV, cross-validation KFolds
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer
if clf==clf_B:
    params = {"max_depth":range(1, 11)}
    #print "n_features_ : {} \nfeature_importances_ : \n {}".format(clf.n_features_,clf.feature_importances_)
elif clf==clf_C:
    #params = { 'kernel' : [ 'rbf', 'linear', 'poly', 'sigmoid' ], 'C' : [ 10, 100, 1000 ], 'gamma' : [ 0.1, 10, 100 ], 'degree' :range(1, 11)}
    params = {'C' : [  1, 10, 100, 1000 ], 'gamma' : [ 0.1, 10, 100 ]}

cv_sets = StratifiedShuffleSplit(labels_train, n_iter=100, test_size=0.1,random_state=80)
scoring_fnc = None # We choose here the defined classifier's score

# Choose the f1_score as scorer should give the better identifier
# which can identify POI's reliably and accurately
"""
#scoring_fnc = make_scorer(f1_score, pos_label=True)
scorer_fnc_res = scorer_fnc(clf, features_test, labels_test)
print "scorer_fnc_res : ", scorer_fnc_res
scoring_fnc = make_scorer(scorer_fnc)
"""

grid = GridSearchCV(estimator=clf, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
grid = grid.fit(features_train, labels_train)
print "\n#==> ******** TUNING WITH GridSearchCV ********* "
#print "grid_scores_, after tuning : \n", grid.grid_scores_
print "best_score_, after tuning : \n", grid.best_score_
print "\nbest_estimator_, after tuning : \n", grid.best_estimator_
print "\nbest_params_, after tuning : \n", grid.best_params_
print "\nscorer_, after tuning : \n", grid.scorer_
print "******************* END GridSearchCV ************************ \n"

# ********* END TUNING ***********




# Test with the test_classifier function from the tester.py script
from tester import test_classifier

print "\n#==> ********** TEST WITH the tester.py SCRIPT *************"
test_classifier(clf, my_dataset, features_list)
print "**************** END test with the tester.py script ************* \n"





print "\n\n++++ Personal complementary tests ++++ "
print "\n#==> ******* TEST OF THE POI IDENTIFIER with the ALGORITHM **********"
print "++++ This gives an idea of how the ALGORITHM should work on unseen data to make prediction ++++"
print "++++ You could change the ALGORITHM to see the difference, in Task 5 section ++++ \n"
#print "features_train[1] : \n", my_dataset[1:3]

# Function to return the real name of the person, from his position in the dataset
def person_name(id):
    dic={}
    person = id
    i = 0
    Skeyvalue = ""
    for value in data_dict.keys():
        if i==person:
            Skeyvalue = value
        i = i+1
    return Skeyvalue
    #print("Associeted name : {} |".format(Skeyvalue))

"""
# Test to return the first four person of the dataset
dic={}
i=0
while i<4:
    for value in data_dict.keys():
        if i<4:
            dic[value]=data_dict[value]
            i=i+1

print "dic \n",dic
"""
perso = person_name(0)

# I print only the 20 first (features_train[:20]). Modify the length for more.

for i, person in enumerate(clf.predict(features_train[:20])):
    if person==0.0:
        person='no'
    if person==1.0:
        person='YES'
    print "Is {} a POI ? : ===============> {}".format(person_name(i), person)
print " ... "

print "\n\n"
# Predicted POIs count in the whole dataset
j=0
poi_ids = []
for i, person in enumerate(clf.predict(features_train)):
    if person==1.0:
        j=j+1
        poi_ids.append(i)

print "Number of predicted POIs by the ALGORITHM for the whole dataset : ",j
print "\nPredicted POIs list : \n"
i=1
for elt in poi_ids:
    print "{}- {} \n".format(i, person_name(elt))
    i=i+1
print "**************** END TEST OF THE POI IDENTIFIER ************* \n"

###################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results.
###################################
### You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)









