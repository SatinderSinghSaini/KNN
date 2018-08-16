from sklearn.datasets import load_iris
iris = load_iris()

x=iris.data
y=iris.target

print("Features:",x.shape)
print("Target:",y.shape)

print(x[:5])
print(y[56:62])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))

sample = [[3.1, 5, 1.4, 0.2], [2.3, 3.3, 5.4, 4.5],[1.1, 1, 1.1, 1.2]]
preds = knn.predict(sample)
pred_species = [iris.target_names[p] for p in preds]
print("Predictions:", pred_species)