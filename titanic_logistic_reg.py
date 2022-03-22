import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%% Load and check data

df_train = pd.read_csv("C:/Users/7alta/OneDrive/Masaüstü/Spyder_Python/train.csv")
print(df_train.info())

#%% Edit data
# Find missing data
df_train.columns[df_train.isnull().any()]

df_train.isnull().sum()


# Fill missing data

# We are going to fill just values of "Age". Because in order to predict values, we need just numeric data.
df_train[df_train["Age"].isnull()]

#now, we are going to compare "Age" data with other related datas to fill missing "Age" values

sns.factorplot(x = "Sex", y = "Age", data = df_train, kind = "box")
plt.show() # Sex is not informative for age prediction, age distribution seems to be same.

sns.factorplot(x = "Sex", y = "Age", hue = "Pclass", data = df_train, kind = "box")
plt.show() # 1st class passengers are older than 2nd, and 2nd is older than 3rd class.

sns.factorplot(x = "Parch", y = "Age", data = df_train, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data = df_train, kind = "box")
plt.show()

# in order to examine "Sex" data on heatmap, we return "Sex" values to int.
df_train.Sex = [1 if each == "male" else 0 for each in df_train.Sex]

sns.heatmap(df_train[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot = True)
plt.show() # Age is not correlated with sex but it is correlated with parch, sibsp and pclass.


index_nan_age = list(df_train["Age"][df_train["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = df_train["Age"][((df_train["SibSp"] == df_train.iloc[i]["SibSp"]) & (df_train["Parch"] == df_train.iloc[i]["Parch"]) & (df_train["Pclass"] == df_train.iloc[i]["Pclass"]))].median()
    age_med = df_train["Age"].median()
    if not np.isnan(age_pred):
        df_train["Age"].iloc[i] = age_pred
    else:
        df_train["Age"].iloc[i] = age_med

df_train[df_train["Age"].isnull()]
        

#%% Remove Unnecessary Columns
df_train.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis = 1, inplace = True)

# change type of all values from int to float
df_train.Pclass = [float(each) for each in df_train.Pclass]
df_train.Sex = [float(each) for each in df_train.Sex]
df_train.SibSp = [float(each) for each in df_train.SibSp]
df_train.Parch = [float(each) for each in df_train.Parch]

df_train.info()

y = df_train.Survived.values
x_data = df_train.drop(["Survived"], axis = 1)

#%% Normalization (make the values between 0-1)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values


#%% Train Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


#%% Parameter Initialize and Sigmoid Function

# There are some techniques that in artificial neural network but for this time initial weights are 0.01.
# Also bias is 0
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1), 0.01)
    b = 0.0
    
    return w,b


def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z)) #formula of sigmoid function
    return y_head


#%% Forward and Backward Propagation

def forward_backward_propagation(w,b, x_train, y_train):
    #forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * (np.log(1 - y_head))
    cost = (np.sum(loss)) / x_train.shape[1] # x_train.shape[1] is for scaling
    
    #backward propagation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients


#%% Updating(Learning) Parameters

def update(w,b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost and gradients
        cost, gradients = forward_backward_propagation(w,b, x_train, y_train)
        cost_list.append(cost)
        
        # let's update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
    
    # we update(learn) parameters weights and bias
    
    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation = "vertical")
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()    
    return parameters, gradients, cost_list

#%% Prediction

def predict(w,b, x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T, x_test) + b)
    y_prediction = np.zeros((1, x_test.shape[1]))
    
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0)
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    
    return y_prediction


#%% Logistic Regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    # initialize
    dimension = x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)
    
    parameters, gradients, cost_list = update(w,b, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    
    # print test errors
    print("Test Accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 1, num_iterations = 300)


