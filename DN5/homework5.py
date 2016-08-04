import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import os

def parseTrainData(fileName):
    print("Parsing train data...")
    rawData = np.genfromtxt(fileName,dtype="S8,S10,S8,i4,i2,S2,i2,f5,f5,S8,i2,S8,i2,S8,i2",delimiter=";",skip_header=1)
    pm = np.array([b'BPRG', b'KGRG', b'BPPL', b'NN', b'BPLS', b'PAYPALVC', b'RG', b'CBA', b'KKE', b'VORAUS'])
    filee = open("data/trainSubmission.txt","w")

    for line in rawData:

        try:
            x = str(float(line[5])) == 'NA'
        except ValueError:
            filee.write(str(0) + "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t"+str(0)+ "\t"+str(0) +"\n")
            continue

        filee.write(str(float(line[0][1:]))+"\t")
        filee.write(str(float(str(line[1]).replace("-","")[2:-1]))+"\t")
        filee.write(str(float(line[2][1:]))+"\t")
        filee.write(str(float(line[3]))+"\t")
        filee.write(str(float(line[4]))+"\t")
        filee.write(str(float(line[5]))+"\t")
        filee.write(str(float(line[6]))+"\t")
        filee.write(str("{0:.2f}".format(float(line[7]),2))+"\t")
        filee.write(str("{0:.2f}".format(float(line[8]),2))+"\t")
        if len(line[9]) == 1 or line[9] == b'NA':
            filee.write(str(0)+"\t")
        else:
            filee.write(str(float(line[9][1:]))+"\t")

        filee.write(str(float(line[10]))+"\t")
        filee.write(str(float(line[11][1:]))+"\t")
        filee.write(str(float(line[12]))+"\t")
        filee.write(str(float(np.where(line[13]==pm)[0]))+"\t")
        filee.write(str(float(line[14]))+"\n")
def parseTestData(fileName):
    print("Parsing test data...")
    rawData = np.genfromtxt(fileName,dtype="S8,S10,S8,i4,i2,S2,i2,f5,f5,S8,i2,S8,i2,S8",delimiter=";",skip_header=1)
    pm = np.array([b'BPRG', b'KGRG', b'BPPL', b'NN', b'BPLS', b'PAYPALVC', b'RG', b'CBA', b'KKE', b'VORAUS'])
    filee = open("data/testSubmission.txt","w")

    for line in rawData:

        try:
            x = str(float(line[5])) == 'NA'
        except ValueError:
            filee.write(str(0) + "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t" + str(0)+ "\t"+str(0)+ "\n")
            continue

        filee.write(str(float(line[0][1:]))+"\t")
        filee.write(str(float(str(line[1]).replace("-","")[2:-1]))+"\t")
        filee.write(str(float(line[2][1:]))+"\t")
        filee.write(str(float(line[3]))+"\t")
        filee.write(str(float(line[4]))+"\t")
        filee.write(str(float(line[5]))+"\t")
        filee.write(str(float(line[6]))+"\t")
        filee.write(str("{0:.2f}".format(float(line[7]),2))+"\t")
        filee.write(str("{0:.2f}".format(float(line[8]),2))+"\t")
        if len(line[9]) == 1 or line[9] == b'NA':
            filee.write(str(0)+"\t")
        else:
            filee.write(str(float(line[9][1:]))+"\t")

        filee.write(str(float(line[10]))+"\t")
        filee.write(str(float(line[11][1:]))+"\t")
        filee.write(str(float(line[12]))+"\t")
        filee.write(str(float(np.where(line[13]==pm)[0]))+"\n")
def getProbabilities(data):
    print("Constructing new feature...")
    costumer_return = {}
    for line in data:
        if line[11] in costumer_return:
            costumer_return[line[11]] += line[-1]
        else:
            costumer_return[line[11]] = line[-1]

    unique_costumers, countsCos = np.unique(data[:,11], return_counts=True)
    costumers = np.asarray((unique_costumers, countsCos)).T

    probs = []
    for key, value in costumer_return.items():
        x = costumers[costumers[:,0] == key]
        num_orders = x[0][1]
        prob = value/num_orders
        if prob > 1.0:
            prob = 1
        probs.append([key,prob])

    np.savetxt("data/probsToReturnSubmission.txt",np.array(probs),fmt="%i %f",delimiter="\t")
def addColumnProbsRemoveOthers(dataTrain, dataTest):
    costumers = np.loadtxt("data/probsToReturnSubmission.txt",delimiter=" ",dtype="i",usecols=(0,))
    probs = np.loadtxt("data/probsToReturnSubmission.txt",dtype="f",delimiter=" ",usecols=(1,))
    combined = {}
    for i, cost in enumerate(costumers):
        combined[cost] = probs[i]

    prob_column_train = []
    prob_column_test = []
    for line in dataTrain:
        prob_column_train.append(combined[line[11]])

    avg_prob = np.mean(probs)
    for line in dataTest:
        if line[11] in combined:
            prob_column_test.append(combined[line[11]])
        else:
            prob_column_test.append(avg_prob);

    prob_column_train = np.array(prob_column_train)
    prob_column_test = np.array(prob_column_test)


    return (np.column_stack((prob_column_train,dataTrain[:,-1])), prob_column_test)
def randomForest(dataTrain,dataTest):
    print("Running RandomForest...")
    trainX = dataTrain[:,0:-1]
    trainY = dataTrain[:,-1]
    print(dataTest.shape)

    rf = RandomForestClassifier(n_estimators=100,max_depth=10,criterion="entropy")
    start1 = time.clock()

    a = np.random.choice(np.arange(0, len(trainY)), replace=False, size=5000) #izberemo 5000 naključnih primerov za učenje iz učne množice, v resnici je ta številko okrog 30% vseh učnih podatkov

    rf.fit(trainX[a], trainY[a])
    end1 = time.clock()

    print("Learning Time:"+str(end1-start1))

    start2 = time.clock()

    testY = rf.predict(dataTest)
    end2 = time.clock()

    print("Prediciting Time:"+str(end2-start2))

    np.savetxt("data/results.txt",testY,fmt='%d')



##########################################################

        #INPUT PARAMETERS

#########################################################
#Predpostavim, da so učni podatki v podmapi data/train.txt
#Predpostavim, da so testni podatki v podmapi data/test.txt
#Predpostavim, da train.txt in test.txt vsebujeta header (prvo vrstico torej ignoriram, če ni headerja, pride do zamika in napačnih rezultatov)
#Vmes se generirajo pomožne datoteke, ki jih na koncu program avtomatsko pobriše

parseTrainData("data/train.txt")
parseTestData("data/test.txt")

dataTrain = np.genfromtxt("data/trainSubmission.txt",delimiter="\t")
dataTest = np.genfromtxt("data/testSubmission.txt",delimiter="\t")
getProbabilities(dataTrain)
train,test = addColumnProbsRemoveOthers(dataTrain,dataTest)
randomForest(np.column_stack((dataTrain[:,[5,7,8,11]],train)),np.column_stack((dataTest[:,[5,7,8,11]],test)))

#Odstranimo pomožne datoteke
os.remove("data/probsToReturnSubmission.txt")
os.remove("data/trainSubmission.txt")
os.remove("data/testSubmission.txt")

