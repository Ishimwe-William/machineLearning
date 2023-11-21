# Learning and Predicting
# Find the most accurate Model
# create a persisting Model
# use the persisting Model


#     Part 1    ##
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
#
# # Import Data from Excel file
# Grade_data = pd.read_excel('Data.xlsx')
#
# # Learning with Decision Classifier
# X = Grade_data.drop(columns=['SNAMES', 'Total Marks', 'Marks /20', 'Grading '])
# y = Grade_data['Grading ']
# model = DecisionTreeClassifier()
# model.fit(X.values, y)
#
# # Predict with the model
# predictions = model.predict([[10, 10, 28, 36]])
# print(predictions)


# #      Part 2      # #
# # Imports all the needed Libraries
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
#
# # Import the Dataset
# Grade_data = pd.read_excel('Data.xlsx')
# # Split dataset into training set and test set
# x = Grade_data.drop(columns=['SNAMES', 'Total Marks', 'Marks /20', 'Grading '])
# y = Grade_data['Grading ']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#
# # Create a Decision Tree, Logistic Regression, Support Vector Machine  and Random Forest Classifiers
# Decision_tree_model = DecisionTreeClassifier()
# Logistic_regression_Model = LogisticRegression(solver='lbfgs', max_iter=10000)
# SVM_model = svm.SVC(kernel='linear')
# RF_model = RandomForestClassifier(n_estimators=100)
#
# # Train the models using the training sets
# Decision_tree_model.fit(x_train, y_train)
# Logistic_regression_Model.fit(x_train, y_train)
# SVM_model.fit(x_train, y_train)
# RF_model.fit(x_train, y_train)
#
# # Predict the Model
# DT_Prediction = Decision_tree_model.predict(x_test)
# LR_Prediction = Logistic_regression_Model.predict(x_test)
# SVM_Prediction = SVM_model.predict(x_test)
# RF_Prediction = RF_model.predict(x_test)
#
# # Calculation of Model Accuracy
# DT_score = accuracy_score(y_test, DT_Prediction)
# lR_score = accuracy_score(y_test, LR_Prediction)
# SVM_score = accuracy_score(y_test, SVM_Prediction)
# RF_score = accuracy_score(y_test, RF_Prediction)
#
# # Display Accuracy
# print("Decision Tree accuracy =", DT_score*100, "%")
# print("Logistic Regression accuracy =", lR_score*100, "%")
# print("Support Vector Machine accuracy =", SVM_score*100, "%")
# print("Random Forest accuracy =", RF_score*100, "%")


# #      Part 3      # #
# # Import Libraries
# import pandas as pd
# from sklearn import svm
# import joblib
#
# # Import the Dataset
# Grade_data = pd.read_excel('Data.xlsx')
#
# # Learning with the Model
# X = Grade_data.drop (columns=['SNAMES', 'Total Marks', 'Marks /20', 'Grading '])
# y = Grade_data['Grading ']
# model = svm.SVC(kernel='linear')
# model.fit(X.values, y)
#
# # Create a Persisting Model
# joblib.dump(model, 'grade-recommender.joblib')

# #      Part 4      # #

# Import Libraries
import joblib

# User Inputs
Quiz = int(input("Enter Quiz Marks :"))
Assgn = input("Enter Assignment Marks: ")
Mid = int(input("Enter Mid Exam Marks Marks :"))
Final = input("Enter Final Exam Marks: ")

# Predict from the created model
model = joblib.load('grade-recommender.joblib')
predictions = model.predict([[Quiz, Assgn, Mid, Final]])
print("The Grade you will obtain is:", predictions)
