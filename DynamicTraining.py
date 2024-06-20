"""
Created on Mon June  18 15:06 2024

@author: Baptiste Guilleminot

Final file to code the way I want to train the GNS.
"""

from SetOfSimulations import *

simus = setOfSimulations(nbSimuTraining=[5,10], nbSimuTest=1, nbSimuValid=4)

def dynamicTraining(simus) :
    #create dataset
    parameters = []
    for i in range(3) :
        for j in range(3) :
            for k in range(3) :
                for l in range(3) :
                    dico = {}
                    dico["young"] = (i + 1) * 1e5
                    dico["nu"] = (j + 1) * 0.1
                    dico["rho"] = (k + 1) * 10000
                    dico["friction angle"] = (1 + l) * 10
                    parameters.append(dico)
    simus.createSuperDataset(parameters, randomnessTrain=0.1, randomnessValid=0.01)
    # Train GNS
    lossFctSteps = []
    stdLossFctSteps = []
    nbSteps = []
    for i in range(100) :
        lossFctSteps.append(simus.trainGNScycle())
        stdLossFctSteps.append(simus.stdLoss())
        nbSteps.append(simus.nbTrainingSteps)
        print("training " + str(i) + " done")
    
    #print quelques rendues
    nbRendue = 3
    pace = len(simus.setOfSimulations) // nbRendue
    for i in range(nbRendue) :
        simus.setOfSimulations[i*pace].rolloutGNS()
        shutil.copyfile('/home/user/Documents/Baptiste/surrogate_modelling/gns/examples/granular_collapse/rollouts/rollout_ex0.gif', "/home/user/Documents/Baptiste/surrogate_modelling/gns/dynamicTraining/DynamicTraining.py/rollout" + simus.setOfSimulations[i*pace].signature() + ".gif")
    
    #Study GNS during and after training
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Study of the evolution of the loss')
    axs[0].plot(nbSteps, lossFctSteps, c='b')
    axs[0].xlabel("Number of training steps")
    axs[0].ylabel("Average loss")

    axs[1].plot(nbSteps, stdLossFctSteps, c='o')
    axs[1].xlabel("Number of training steps")
    axs[1].ylabel("Standard deviation of the loss")

    simus.graphLossParameter("rho")
    simus.graphLossParameter("nu")
    simus.graphLossParameter("friction angle")
    simus.graphLossParameter("young")
    plt.show()




