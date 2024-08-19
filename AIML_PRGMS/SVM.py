import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()

x=iris.data[:,:2]
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,  test_size=0.3,random_state=42)
 

model=SVC(kernel='linear',C=1.0)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"ACCURACY: {accuracy*100:.2f}%")
 
def plot_decision_boundaries(X,y,model):
    h=.02            
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    z=model.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    plt.contourf(xx,yy,z,alpha=0.8)
    plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k',marker='o')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title('svm decision boundaries')
    plt.show()
plot_decision_boundaries(x,y,model)