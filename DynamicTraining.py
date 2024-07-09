"""
Created on Mon June  18 15:06 2024

@author: Baptiste Guilleminot

Final file to code the way I want to train the GNS.
"""

from SetOfSimulations import *

#simus1 = setOfSimulations(nbSimuTraining=[5,10], nbSimuTest=1, nbSimuValid=4)
#simus2 = setOfSimulations(nbSimuTraining=[5], nbSimuTest=1, nbSimuValid=4)


def Data(simus) :
    #create dataset
    parameters = []
    for i in range(10) :
        for l in range(4) :
            dico = {}
            dico["nu"] = 0.3
            dico["rho"] = 2500
            dico["young"] = 10**(5 + i*0.2)
            dico["friction angle"] = l * 5 + 20 
            parameters.append(dico)
    simus.createSuperDataset(parameters, randomnessTrain=0.01, randomnessValid=0.001)
    print("Data created and backup done")

def dynamicTraining1(simus) :
    # Train GNS
    lossFctSteps = []
    stdLossFctSteps = []
    nbSteps = []
    for i in range(100) :
        lossFctSteps.append(simus.trainGNScycle())
        stdLossFctSteps.append(simus.stdLoss())
        nbSteps.append(simus.nbTrainingSteps)
        print("training " + str(i) + " done")
    simus.saveSetOfSimulations("halfPlane" + str(nbSteps[-1]) + "StepsVersion1")
    #print quelques rendues
    nbRendue = 1
    pace = len(simus.setOfSimulations) // nbRendue
    for i in range(nbRendue) :
        simus.setOfSimulations[i*pace].rolloutGNS(simus.nbTrainingSteps, typeOutput='gif')
    return lossFctSteps, stdLossFctSteps, nbSteps

def dynamicTraining2(simus) :
    simus.saveSetOfSimulations("halfPlaneStepsVersion1")
    simusBis = setOfSimulations(fromJsonFile="backupSimus/halfPlaneStepsVersion1.json")
    #change the model and state file compared to others
    simus.nbStepsPerParametersPerCycle = 1
    simus.trainGNScycle()
    simus.nbStepsPerParametersPerCycle = 100

    # Train GNS
    lossFctSteps = []
    stdLossFctSteps = []
    nbSteps = []
    for i in range(100) :
        lossFctSteps.append(simus.trainGNScycle())
        simusBis.nbTrainingSteps  = simus.nbTrainingSteps
        simusBis.setOfSimulations = simus.setOfSimulations[len(simus.setOfSimulations)//2:]
        simusBis.trainGNScycle()
        simus.nbTrainingSteps += 100
        stdLossFctSteps.append(simus.stdLoss())
        nbSteps.append(simus.nbTrainingSteps)
        print("training " + str(i) + " done")
    simus.saveSetOfSimulations("halfPlane" + str(nbSteps[-1]) + "StepsVersion2")
    #print quelques rendues
    nbRendue = 3
    pace = len(simus.setOfSimulations) // nbRendue
    for i in range(nbRendue) :
        simus.setOfSimulations[i*pace].rolloutGNS(simus.nbTrainingSteps, typeOutput = 'gif')
    return lossFctSteps, stdLossFctSteps, nbSteps



def studyTraining(simus, lossFctSteps, stdLossFctSteps, nbSteps, versionTraining) :
    #Study GNS during and after training
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Study of the evolution of the loss')
    axs[0].plot(nbSteps, lossFctSteps, c='b')
    axs[0].set_yscale('log')
    axs[0].set_xlabel("Number of training steps")
    axs[0].set_ylabel("Average loss")

    axs[1].plot(nbSteps, stdLossFctSteps, color = 'orange')
    axs[1].set_yscale('log')
    axs[1].set_xlabel("Number of training steps")
    axs[1].set_ylabel("Standard deviation of the loss")
    fig.savefig("dataTraining/" + versionTraining + ".png")

    simus.graphLossParameter("friction angle", "dataTraining/loss_" + str(nbSteps[-1]) + versionTraining)
    simus.graphLossParameter("young", "dataTraining/loss_" + str(nbSteps[-1]) + versionTraining)
    

def allTraining1() : 
    simus1 = setOfSimulations(nbSimuTraining=[5,10], nbSimuTest=1, nbSimuValid=4)
    Data(simus1)
    lossFctSteps, stdLossFctSteps, nbSteps = dynamicTraining1(simus1)
    with open('loss.txt', 'w') as f :
        line = ' '.join(map(str, lossFctSteps)) + '\n'
        f.write(line)
        line = ' '.join(map(str, stdLossFctSteps)) + '\n'
        f.write(line)
        line = ' '.join(map(str, nbSteps)) + '\n'
        f.write(line)
    studyTraining(simus1, lossFctSteps, stdLossFctSteps, nbSteps, "dynamic_training_version_1")

def allTraining2() :
    simus2 = setOfSimulations(nbSimuTraining=[5], nbSimuTest=1, nbSimuValid=4)
    Data(simus2)
    lossFctSteps, stdLossFctSteps, nbSteps = dynamicTraining2(simus2)
    with open('loss.txt', 'w') as f :
        line = ' '.join(map(str, lossFctSteps)) + '\n'
        f.write(line)
        line = ' '.join(map(str, stdLossFctSteps)) + '\n'
        f.write(line)
        line = ' '.join(map(str, nbSteps)) + '\n'
        f.write(line)
    studyTraining(simus2, lossFctSteps, stdLossFctSteps, nbSteps, "dynamic_training_version_2")

def allTrainingClassique() : 
    simus3 = setOfSimulations(nbSimuTraining=[8], nbSimuTest=1, nbSimuValid=4)
    Data(simus3)
    simus3.nbStepsPerParametersPerCycle = 2
    simus3.trainGNScycle()
    simus3.nbStepsPerParametersPerCycle = 100
    lossFctSteps, stdLossFctSteps, nbSteps = dynamicTraining1(simus3)
    print(lossFctSteps)
    print(stdLossFctSteps)
    print(nbSteps)
    studyTraining(simus3, lossFctSteps, stdLossFctSteps, nbSteps, "classic_training")

