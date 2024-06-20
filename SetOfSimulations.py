"""
Created on Mon June  10 17:29:22 2024

@author: Baptiste Guilleminot

This file contains the class with to train a GNS on different set of parameters.
"""

import matplotlib.pyplot as plt
from Simulations import *

class setOfSimulations :

    def __init__(self, setOfSimulations = [], nbTrainingSteps = 0, nbStepsPerParametersPerCycle = 100, nbSimuTraining = [1], nbSimuTest = 1, nbSimuValid = 1, nbFrame = 81, pathBgeo = '/media/user/Volume/granular_collapse_GNS_dyn/', pathNpz = '/home/user/Documents/Baptiste/surrogate_modelling/gns/examples/granular_collapse/datasets/', pathExe = '/home/user/Documents/Baptiste/surrogate_modelling/gns/dynamicTraining/', pathGNS = '/home/user/Documents/Baptiste/surrogate_modelling/gns', pathMPM = '/home/user/Documents/myJixie/jixie_lars/Projects/mpm/mpm', pathLua = '/home/user/Documents/Baptiste/surrogate_modelling/gns/dynamicTraining/granular_collapse_gns.lua') :
        # Define useful quantities
        self.pathBgeo = pathBgeo
        self.pathExe = pathExe
        self.pathNpz = pathNpz
        self.setOfSimulations = setOfSimulations
        self.nbTrainingSteps = nbTrainingSteps
        self.nbStepsPerParametersPerCycle = nbStepsPerParametersPerCycle
        self.nbSimuTraining = nbSimuTraining
        self.nbSimuTest = nbSimuTest
        self.nbSimuValid = nbSimuValid
        self.nbFrame = nbFrame
        self.pathLua = pathLua
        # Modify bash file to go where the GNS algo is
        with open(pathExe + "exeMPM.sh", 'w') as file :
            file.write("#!/bin/bash" + '\n')
            file.write(pathMPM + " " + pathLua + '\n')
        
        with open(pathExe + "runGNSretrain.sh", 'r') as file:
            lines = file.readlines()
        with open(pathExe + "runGNSretrain.sh", 'w') as file:
            for line in lines:
                if line.strip()[:3] == "cd ":
                    file.write("cd " + pathGNS + '\n')
                else:
                    file.write(line)
        
        with open(pathExe + "runGNStrain.sh", 'r') as file:
            lines = file.readlines()
        with open(pathExe + "runGNStrain.sh", 'w') as file:
            for line in lines:
                if line.strip()[:3] == "cd ":
                    file.write("cd " + pathGNS + '\n')
                else:
                    file.write(line)
        
        with open(pathExe + "runGNSrollout.sh", 'r') as file:
            lines = file.readlines()
        with open(pathExe + "runGNSrollout.sh", 'w') as file:
            for line in lines:
                if line.strip()[:3] == "cd ":
                    file.write("cd " + pathGNS + '\n')
                else:
                    file.write(line)
        
        with open(pathExe + "runGNStest.sh", 'r') as file:
            lines = file.readlines()
        with open(pathExe + "runGNStest.sh", 'w') as file:
            for line in lines:
                if line.strip()[:3] == "cd ":
                    file.write("cd " + pathGNS + '\n')
                else:
                    file.write(line)
        
        with open(pathExe + "runGNSvalid.sh", 'r') as file:
            lines = file.readlines()
        with open(pathExe + "runGNSvalid.sh", 'w') as file:
            for line in lines:
                if line.strip()[:3] == "cd ":
                    file.write("cd " + pathGNS + '\n')
                else:
                    file.write(line)
        
        with open(pathLua, 'r') as file:
            lines = file.readlines()
        with open(pathLua, 'w') as file:
            for line in lines:
                if line.strip()[:12] == "end_frame = ":
                    file.write("end_frame = " + str(nbFrame-1) + '\n')
                else:
                    file.write(line)
        
        with open(pathNpz + "metadata.json", 'r') as file:
            lines = file.readlines()
        with open(pathNpz + "metadata.json", 'w') as file:
            for line in lines:
                if line.strip()[:12] == "\"sequence_length\": ":
                    file.write("\"sequence_length\": " + str(nbFrame) + '\n')
                else:
                    file.write(line)

    def createSuperDataset(self, parameters = [{"young" : 3e5, "nu" : 0.3, "rho" : 25000, "friction angle" : 23}], nbPointsVolume = 1000, frictionVolume = 0.31, randomnessTrain = 0, randomnessValid = 0, adressObject = ['/home/user/Documents/Baptiste/surrogate_modelling/gns/dynamicTraining/halfPlane.dat']) :
        for path in adressObject :
            for setOfParameter in parameters :
                simu = simulations(young = setOfParameter["young"], nu = setOfParameter["nu"], rho = setOfParameter["rho"], frictionAngle = setOfParameter["friction angle"], nbFrame = self.nbFrame, frictionVolume = frictionVolume, adressObject=path)
                simu.createDataset(adressBgeo = self.pathBgeo, adressNpz = self.pathNpz, adressLua = self.pathLua, nbPointsVolume = nbPointsVolume, nbSimuTrain=self.nbSimuTraining, nbSimuTest=self.nbSimuTest, nbSimuValid=self.nbSimuValid, exeMPM = self.pathExe + "exeMPM.sh", randomnessTrain=randomnessTrain, randomnessValid=randomnessValid)
                self.setOfSimulations.append(simu)
    
    def orderByLoss(self) :
        """
        Go through the list and exchange pairs if they are not ordered. Goes through all the list without exchanging when ordered.
        """
        i = 0
        ordered = False
        while (i < len(self.setOfSimulations)**2 and not(ordered)) :
            ordered = True
            for i in range(len(self.setOfSimulations)-1) :
                if self.setOfSimulations[i].loss > self.setOfSimulations[i+1].loss: 
                    self.setOfSimulations[i], self.setOfSimulations[i+1] = self.setOfSimulations[i+1], self.setOfSimulations[i]
                    ordered = False
    
    def adaptNbOfSimu(self) :
        quotient = len(self.setOfSimulations) // len(self.nbSimuTraining)
        quotient += 1
        for i in range(len(self.setOfSimulations)) :
            self.setOfSimulations[i].AdressTrain = i // quotient

    def trainGNScycle(self) :
        for simu in self.setOfSimulations :
            simu.trainGNS(self.nbTrainingSteps,self.nbStepsPerParametersPerCycle, exeGNStrain=self.pathExe+"runGNStrain.sh", exeGNSretrain=self.pathExe+"runGNSretrain.sh", adressNpz=self.pathNpz)
            self.nbTrainingSteps += self.nbStepsPerParametersPerCycle
        for simu in self.setOfSimulations :
            simu.validGNS(self.nbTrainingSteps, exeGNSvalid=self.pathExe+"runGNSvalid.sh", adressNpz=self.pathNpz)
        self.orderByLoss()
        self.adaptNbOfSimu()
        return(self.averageLoss)
    
    def averageLoss(self) :
        moy = 0
        for simu in self.setOfSimulations :
            moy += simu.loss
        moy /= len(self.setOfSimulations)
        return(moy)

    def stdLoss(self) : 
        std = 0
        for simu in self.setOfSimulations :
            std += simu.loss**2
        std /= len(self.setOfSimulations)
        std -= self.averageLoss()**2
        return(std)
  
    def graphLossParameter(self, parameter = "rho") :
        '''
        Input : the parameter on which we want the graph made
        Show the graph using test values.
        Output : list of values, loss on which the graph was build
        '''
        dico = {}
        for simu in self.setOfSimulations :
            b = 0
            if parameter == "rho" :
                b = simu.rho
            elif parameter == "nu" :
                b = simu.nu
            elif parameter == "young" :
                b = simu.young
            elif parameter == "friction angle" :
                b = simu.frictionAngle
            else : 
                return("wrong name as parameter")
            loss = simu.testGNS(self.nbTrainingSteps, exeGNStest = self.pathExe + "runGNStest.sh", adressNpz = self.pathNpz)
            if b in dico :
                dico[b] += np.array([loss, 1])
            else :
                dico[b] = np.array([loss, 1])
        listOfValue = []
        listOfLoss = []
        for valuePar, loss in dico.items() :
            listOfValue.append(valuePar)
            listOfLoss.append(loss[0] / loss[1])
        if parameter == "rho" :
            par = "Volumic mass"
            b = "kg/m^3"
        elif parameter == "nu" :
            par = "Viscosity"
            b = "Pa.s"
        elif parameter == "young" :
            par = "Young modulus"
            b = "MPa"
        elif parameter == "friction angle" :
            par = "Friction angle"
            b = "degrés"
        plt.figure("Loss as a function of " + par)
        plt.clf()
        plt.scatter(listOfValue, listOfLoss)
        plt.ylabel("Loss")
        plt.xlabel(par + " [" + b + "]")
        plt.title("Average loss for each value of " + par)
        return(listOfValue, listOfLoss)

                




def testClassSetOfSimulations() :
    simu = setOfSimulations(nbSimuTraining=[1])
    simu.createSuperDataset(parameters = [{"young" : 3e5, "nu" : 0.3, "rho" : 25000, "friction angle" : 23}, {"young" : 4e7, "nu" : 0.3, "rho" : 25000, "friction angle" : 23}])
    for i in range(1) :
        simu.trainGNScycle()
        print("training " + str(i) + " done")
    simu.setOfSimulations[0].rolloutGNS(simu.nbTrainingSteps)
    print("rollout done")
    simu.graphLossParameter("young")