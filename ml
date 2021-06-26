import sys
print ('Python : {}'.format(sys.version))
import scipy
print('Scipy: {}'.format(scipy._version_))
import numpy
print('Numpy: {}'.format(numpy._version_))
import matplotlib
print('Matplotlib: {}'.format(matplotlib._version_))
import pandas
print('Pandas: {}'.format(pandas._version_))
import sklearn
print('Sklearn: {}'.format(sklearn._version_))

import pandas
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedkFold
from sklearn.metrices import classification_report
from sklearn.metrices import confusion_matrix
from sklearn.metrices import accuracy_score
from sklearn.metrices import classification_report
from sklearn.Linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbours import KNeighborsClassifier
from sklearn.discrimant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width' ,'petal-length','petal-width', 'class']
dataset = read_csv(url, names=names)

print(dataset.shape)

print(dataset.head(20))

print(dataset.grouphy('class').size())

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

dataset.hist()
pyplot.show()

scatter_matrix(dataset)
pyplot.show()

array= dataset.values
X = array[:, 0:4]
Y = array[:, 4]

pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparision')
pyplot.show()








