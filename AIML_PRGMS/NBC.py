import csv
import math
import random
import statistics
def cal_probability(x,mean,stdev):
    exponent=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent
dataset=[]
dataset_size=0
with open("p8.csv")as csvfile:
    lines=csv.reader(csvfile)
    for row in lines:
        dataset.append([float(attr)for attr in row])
dataset_size=len(dataset)
print("size of dataset is:",dataset_size)
train_size=int(0.7*dataset_size)
print(train_size)
x_train=[]
x_test=dataset.copy()
training_indexes=random.sample(range(dataset_size),train_size)
for i in training_indexes:
    x_train.append(dataset[i])
    x_test.remove(dataset[i])
classes={}
for samples in x_train:
    last=int(samples[-1])
    if last not in classes:
        classes[last]=[]
    classes[last].append(samples)
print(classes)
summaries={}
for classvalue,training_data in classes.items():
    summary=[(statistics.mean(attributes),statistics.stdev(attributes))for attributes in zip(*training_data)]
    del summary[-1]
    summaries[classvalue]=summary
print(summaries)
x_prediction=[]
for i in x_test:
    probabilities={}
    for classvalue,classsummary in summaries.items():
        probabilities[classvalue]=1
        for index,attr in enumerate(classsummary):
            probabilities[classvalue]*=cal_probability(i[index],attr[0],attr[1])
    best_label,best_prob=None,-1
    for classvalue,probability in probabilities.items():
        if best_label is None or probability>best_prob:
            best_prob=probability
            best_label=classvalue
    x_prediction.append(best_label)
correct=0
for index,key in enumerate(x_test):
    if x_test[index][-1]==x_prediction[index]:
        correct+=1
print("Accuracy",correct/(float(len(x_test)))*100)