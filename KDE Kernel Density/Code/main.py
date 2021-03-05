
import numpy as np
from kde import *
import matplotlib.pyplot as plt
# Define training set
c = np.array([[1,1],[1,4],[3,2.5],[4,2.5]])
n = c.shape[0]

# ## Define test set
# Create a regular set of points which cover the plane $[0, 5] \times [0, 5]$  and stores them in testSet
# Define test set
min_X, max_X = 0, 5
min_Y, max_Y = 0, 5
intLength = 30
x = np.linspace(min_X, max_X, num=intLength)
y = np.linspace(min_Y, max_Y, num=intLength)
testSet = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
X,Y = np.meshgrid(x,y)


# smoothing parameters
h = np.array([0.3,0.4,0.5])

# dimensionality of the test set
d = testSet.shape[1]

## Ex1:

#plot conditional density function

for j in range(len(h)):
    for i in range(len(c)):
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        ax.set_title("Kernel Function at point :  c = "+str(c[i])+ ". h : "+str(h[j]))
        ax.plot_surface(X,Y,densityEstimation(testSet,c[i],h[j],d).reshape(30,30))

plt.show()

#plot density function
for i in range(len(h)):
    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    ax.set_title("Density Function p(x), h:"+str(h[i]))
    ax.plot_surface(X, Y, densityEstimation2(testSet, c, h[i], d).reshape(30, 30))

plt.show()

## Ex2

# # Load the Iris data.
from data_utils import load_IRIS

data_X, data_y = load_IRIS(test=False)
X=np.asarray(data_X)

unique_y = np.unique (data_y)
points_by_class = [[x for x, t in zip (data_X, data_y) if t == c] for c in unique_y]
points_by_class = np.asarray(points_by_class)

h = [0.1]

# EX 2 A)

# Declaration of my trainings sets
sepalLenghtClass0 = points_by_class[0][:,0]
sepalLenghtClass1 = points_by_class[1][:,0]
sepalLenghtClass2 = points_by_class[2][:,0]
sepalLenghtClass = [sepalLenghtClass0,sepalLenghtClass1,sepalLenghtClass2]

sepalWidthClass0 = points_by_class[0][:,1]
sepalWidthClass1 = points_by_class[1][:,1]
sepalWidthClass2 = points_by_class[2][:,1]
sepalWidthClass = [sepalWidthClass0,sepalWidthClass1,sepalWidthClass2]

petalLengthClass0 = points_by_class[0][:,2]
petalLengthClass1 = points_by_class[1][:,2]
petalLengthClass2 = points_by_class[2][:,2]
petalLengthClass = [petalLengthClass0,petalLengthClass1,petalLengthClass2]

petalWidthClass0 = points_by_class[0][:,3]
petalWidthClass1 = points_by_class[1][:,3]
petalWidthClass2 = points_by_class[2][:,3]
petalWidthClass = [petalWidthClass0,petalWidthClass1,petalWidthClass2]

# Putting together my trainings sets, titles and x domain
attributes= [sepalLenghtClass,sepalWidthClass,petalLengthClass,petalWidthClass]

title = ["sepal Lenght","sepal Width","petal Length", "petal Width"]

x1 = np.asarray(np.linspace(4, 8, num=50))
x2 = np.asarray(np.linspace(2, 5, num=50))
x3 = np.asarray(np.linspace(1, 7, num=50))
x4 = np.asarray(np.linspace(0, 3, num=50))
x = np.asarray([x1,x2,x3,x4])

# Plotting the density function for each attribute
for i in range(len(x)):
    fig = plt.figure()
    plt.title(title[i])
    plt.plot(x[i],densityEstimation2(x[i],attributes[i][0],h[0],1))
    plt.plot(x[i],densityEstimation2(x[i],attributes[i][1],h[0],1))
    plt.plot(x[i],densityEstimation2(x[i],attributes[i][2],h[0],1))

plt.show()


# Ex 2 B)

# Declaration of my trainings sets

# sepal.length/witdh
sepalLenghtClass0 = points_by_class[0][:,0]
sepalWidthClass0 = points_by_class[0][:,1]
lenghtWidthClassO = np.asarray(list(zip(sepalLenghtClass0,sepalWidthClass0)))

sepalLenghtClass1 = points_by_class[1][:,0]
sepalWidthClass1 = points_by_class[1][:,1]
lenghtWidthClass1 = np.asarray(list(zip(sepalLenghtClass1,sepalWidthClass1)))

sepalLenghtClass2 = points_by_class[2][:,0]
sepalWidthClass2 = points_by_class[2][:,1]
lenghtWidthClass2 = np.asarray(list(zip(sepalLenghtClass2,sepalWidthClass2)))

# sepal.length_petal.length
sepalLenghtClass0 = points_by_class[0][:,0]
petalLengthClass0 = points_by_class[0][:,2]
lenghtSepalPetalClass0 = np.asarray(list(zip(sepalLenghtClass0,petalLengthClass0)))

sepalLenghtClass1 = points_by_class[1][:,0]
petalLengthClass1 = points_by_class[1][:,2]
lenghtSepalPetalClass1 = np.asarray(list(zip(sepalLenghtClass1,petalLengthClass1)))

sepalLenghtClass2 = points_by_class[2][:,0]
petalLengthClass2 = points_by_class[2][:,2]
lenghtSepalPetalClass2 = np.asarray(list(zip(sepalLenghtClass2,petalLengthClass2)))

# sepal.length_petal.width
sepalLenghtClass0 = points_by_class[0][:,0]
petalWidthClass0 = points_by_class[0][:,3]
lenghtSepalPetalWidthClass0 = np.asarray(list(zip(sepalLenghtClass0,petalWidthClass0)))

sepalLenghtClass1 = points_by_class[1][:,0]
petalWidthClass1 = points_by_class[1][:,3]
lenghtSepalPetalWidthClass1 = np.asarray(list(zip(sepalLenghtClass1,petalWidthClass1)))

sepalLenghtClass2 = points_by_class[2][:,0]
petalWidthClass2 = points_by_class[2][:,3]
lenghtSepalPetalWidthClass2 = np.asarray(list(zip(sepalLenghtClass2,petalWidthClass2)))

# Declaration of my titles
title1 = ["sepal.length_sepal.width Iris Setosa", "sepal.length_sepal.width Iris Versicolor","sepal.length_sepal.width Iris Virginica"]
title2 = ["sepal.length_petal.length Iris Setosa", "sepal.length_petal.length Iris Versicolor","sepal.length_petal.length Iris Virginica"]
title3 = ["sepal.length_petal.width Iris Setosa", "sepal.length_petal.width Iris Versicolor","sepal.length_petal.width Iris Virginica"]

title = [title1,title2,title3]

# Putting together my trainings sets
c1 = [lenghtWidthClassO,lenghtWidthClass1,lenghtWidthClass2]
c2 = [lenghtSepalPetalClass0,lenghtSepalPetalClass1,lenghtSepalPetalClass2]
c3 = [lenghtSepalPetalWidthClass0,lenghtSepalPetalWidthClass1,lenghtSepalPetalWidthClass2]

c = [c1,c2,c3]

# Definition of my testSet and axes

# I take a biger interval that's why the plot may not look as yours but if i take a shorter interval for pairs of attribute 1 and 2 its seems to be the same (Except for the value on my Z axe c.f report )
min_X, max_X = 4, 8
min_Y, max_Y = 0, 6
intLength = 30
x = np.linspace(min_X, max_X, num=intLength)
y = np.linspace(min_Y, max_Y, num=intLength)
testSet = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
X,Y = np.meshgrid(x,y)

#Numbers of pairs of attribute
pairsAttribute = 3

#Plotting for every class every pairs of attributes
for j in range(pairsAttribute):
    for i in range(len(unique_y)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title[j][i])
        ax.plot_surface(X, Y, densityEstimation2(testSet, c[j][i], h[0], 2).reshape(30, 30))
plt.show()


## Ex3. Iris dataset: Naive Bayes
h = [0.3,0.4,0.5]

X_train, y_train, X_test, y_test = load_IRIS(test=True)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

# Declaration of my points by class to define my trainings sets
unique_y = np.unique(y_train)
points_by_class = [[x for x, t in zip(X_train, y_train) if t == c] for c in unique_y]
points_by_class_array = np.asarray(points_by_class)

# Declaration of my trainings sets
c = [points_by_class[0], points_by_class[1], points_by_class[2]]

# Declaration of my testing sets and training sets
testSet = X_test
trainSet = X_train
# Declaration of the dimension of my data (Number of attributs
d = len(testSet[0])


# Compute of my precision for trainSet
prior = train(X_train, y_train)

print("TRAIN ACCURACY")
print(" ")
for j in range(len(h)):
    resultat1=densityEstimation2(trainSet,c[0],h[j],d) * prior[0]
    resultat2=densityEstimation2(trainSet,c[1],h[j],d) * prior[1]
    resultat3=densityEstimation2(trainSet,c[2],h[j],d) * prior[2]

    posterior = list(zip(resultat1,resultat2,resultat3))
    y_pred = np.argmax(posterior, axis=1)

    num_correct_train = np.unique(y_pred==y_train, return_counts=True)[1][1]
    accuracy_train = num_correct_train/len(y_train)
    print('Train: with h = '+str(h[j])+': Got %d / %d correct => accuracy: %f' % (num_correct_train, len(y_train), accuracy_train))


# Compute of my precision for testSet
print("TEST ACCURACY")
print(" ")
for j in range(len(h)):
    resultat1=densityEstimation2(testSet,c[0],h[j],d) * prior[0] # I have a better precision when i dont multiplicate by prior why?
    resultat2=densityEstimation2(testSet,c[1],h[j],d) * prior[1]
    resultat3=densityEstimation2(testSet,c[2],h[j],d) * prior[2]

    posterior = list(zip(resultat1,resultat2,resultat3))
    y_pred = np.argmax(posterior, axis=1)


    num_correct_test = np.unique(y_pred==y_test, return_counts=True)[1][1]
    accuracy_test = num_correct_test/len(y_test)
    print('Test: with h = '+str(h[j])+': Got %d / %d correct => accuracy: %f' % (num_correct_test, len(y_test), accuracy_test))


