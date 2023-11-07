# Decision-Tree-Assignment
# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Loading the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Creating a decision tree classifier
clf = DecisionTreeClassifier()

# Training the classifier using the training data
clf = clf.fit(X_train, y_train)

# Making predictions on the test data
y_pred = clf.predict(X_test)

# Evaluating the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# You can also visualize the decision tree if you have graphviz installed
# The following lines can be added after training the classifier:
# from sklearn.tree import export_graphviz
# from six import StringIO
# from IPython.display import Image
# import pydotplus
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=iris.feature_names, class_names=iris.target_names)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
