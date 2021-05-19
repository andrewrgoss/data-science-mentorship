# !/usr/bin/env python3
__author__ = 'agoss'

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import statsmodels.api as sm
import sys

"""
Objective: predict the score on the final exam. You can consider some or all of the available data.
"""


def main():
    # read data from csv into dataframe
    df = pd.read_csv('philosophy_101_grades.csv')

    # # histogram of grades
    # plt.hist(df['Final'], bins=45)
    # plt.xlabel('Final Grade')
    # plt.ylabel('Count')
    # plt.title('Distribution of Final Grades')
    #
    # # make a density plot for each gender
    # sns.kdeplot(df.loc[df['Gender'] == 0, 'Final'],
    #             label='Male', shade=True)
    # sns.kdeplot(df.loc[df['Gender'] == 1, 'Final'],
    #             label='Female', shade=True)
    # plt.legend()
    # plt.xlabel('Final Grade')
    # plt.ylabel('Density')
    # plt.title('Density Plot of Final Grades by Gender')

    # find correlation coefficients and sort - determines most useful variables for predicting a final grade
    print(df.corr()['Final'].sort_values())

    # style.use("ggplot")
    #
    # # select the value we want to predict
    # predict = "Final"
    #
    # # list the variables we want to use for our predictions in this model
    # data = df[["Student", "Homework 1", "Homework 2", "Homework 3", "Midterm", "Final", "Gender"]]
    # data = shuffle(data)
    #
    # x = np.array(data.drop([predict], 1))
    # y = np.array(data[predict])
    #
    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    #
    # # Train model multiple times to find the highest accuracy
    # best = 0
    # for _ in range(200):
    #     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    #
    #     linear = linear_model.LinearRegression()
    #
    #     linear.fit(x_train, y_train)
    #     acc = linear.score(x_test, y_test)
    #     print("Accuracy: " + str(acc))
    #
    #     # Save the highest accuracy
    #     if acc > best:
    #         best = acc
    #         with open("studentgrades.pickle", "wb") as f:
    #             pickle.dump(linear, f)
    # print("Highest Accuracy:", best)
    #
    # # Load model
    # pickle_in = open("studentgrades.pickle", "rb")
    # linear = pickle.load(pickle_in)
    #
    # print("-------------------------")
    # print('Coefficient: \n', linear.coef_)
    # print('Intercept: \n', linear.intercept_)
    # print("-------------------------")
    #
    # predictions = linear.predict(x_test)
    #
    # # Print the predictions, the variables we used and the actual final grade
    # for x in range(len(predictions)):
    #     print("Predicted final grade for student ID#{" + str(x_test[x][0]) + "}:", predictions[x],
    #           "|| Actual final grade:", y_test[x], "|| Data:", x_test[x])
    #
    # # Create visualisation of the model
    # plot = "Homework 2"
    # plt.scatter(data[plot], data["Final"])
    # plt.legend(loc=4)
    # plt.xlabel(plot)
    # plt.ylabel("Final Grade")
    # plt.show()

    # defining the variables
    x = df[['Homework 2', 'Homework 3', 'Midterm', 'Gender']]  # x = independent variable
    y = df['Final']  # y = dependent variable

    # adding the constant term
    x = sm.add_constant(x)

    # performing the regression
    # and fitting the model
    result = sm.OLS(y, x).fit()

    # printing the summary table
    print(result.summary())

    """
    ----Variables summary----
    * 'Homework 2' == [R-squared: 0.732, Adj. R-squared: 0.729, Homework 2 coef: 0.8375, Homework 2 prob: 0.000]
    
    * 'Homework 2', 'Homework 3' == [R-squared: 0.817, Adj. R-squared: 0.813, 
       Homework 2 coef: 0.5835, Homework 3 coef: 0.3276, Homework 2 prob: 0.000, Homework 3 prob: 0.000]
       
    * 'Homework 2', 'Homework 3', 'Midterm' == [R-squared: 0.825, Adj. R-squared: 0.820, 
       Homework 2 coef: 0.5299, Homework 3 coef: 0.3173, Midterm coef: 0.1057, 
       Homework 2 prob: 0.000, Homework 3 prob: 0.000, Midterm prob: 0.037]
       
    * 'Homework 2', 'Homework 3', 'Midterm', 'Gender' == [R-squared: 0.827, Adj. R-squared: 0.820, 
       Homework 2 coef: 0.5248, Homework 3 coef: 0.3174, Midterm coef: 0.1066, Gender coef: -1.3255,
       Homework 2 prob: 0.000, Homework 3 prob: 0.000, Midterm prob: 0.035, Gender prob: 0.297]
    """


try:
    main()
except Exception as err:
    print(err)
    sys.exit(1)
