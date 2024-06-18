"""
Created on Mon June  10 17:29:22 2024

@author: Baptiste Guilleminot

This file contains the class with to train a GNS on a specific set of parameters.
"""

import hou
import numpy as np
import subprocess
import shutil

class simulations :
    def __init__(self, young = 3e5, nu = 0.3, rho = 25000, frictionAngle = 23, frictionVolume = 0.31, nbFrame = 81,
                 AdressBgeoTrain = 0, AdressBgeoTrainReserve = [], AdressBgeoTest = [], AdressBgeoValid = [], adressBgeoObject = "", AdressNpzTrainReserve = [], adressNpzTest = "", adressNpzValid = "",
                 loss = 0) :
        self.young = young
        self.nu = nu
        self.rho = rho 
        self.frictionAngle = frictionAngle
        self.frictionVolume = frictionVolume
        self.nbFrame = nbFrame
        #Contains the adress of the current simu used to trained among the reserve
        self.AdressTrain = AdressBgeoTrain
        #Contains the adress of all the simulations
        self.AdressBgeoTrainReserve = AdressBgeoTrainReserve
        self.AdressBgeoTest = AdressBgeoTest
        self.AdressBgeoValid = AdressBgeoValid
        self.adressBgeoObject = adressBgeoObject
        self.AdressNpzTrainReserve = AdressNpzTrainReserve
        self.AdressNpzTest = adressNpzTest
        self.AdressNpzValid = adressNpzValid
        self.loss = loss
    

    def launchSimulation(self,  adressSimu, exeMPM, adressLua, nbSimu, typeSimu, randomness = 0) :
        """
        Input : desired number of simulation and adress used plus in which category we want it
        Launch all the simulations and place them at the desired place, the adress is saved.
        Output : Confirmation simulation done
        """
        AdressBgeo = []
        for i in range(nbSimu) :
            adressCurrentSimu = adressSimu + typeSimu + "/"
            adressCurrentSimu += str(int(self.young)) + "_" + str(int(self.nu * 100)) + "_" + str(int(self.rho)) + "_" + str(int(self.frictionAngle)) + "_" + str(i) + "-" + str(nbSimu)
            # The different parameters which need to be adjusted
            linesToReplace = {"Youn" : "Youngs = " + str(self.young),
                              "nu =" : "nu = " + str(self.nu),
                              "rho " : "rho = " + str(self.rho),
                              "fric" : "friction_angle =" + str(self.frictionAngle),
                              "outp" : "output = \"" + adressCurrentSimu + "\"",
                              "volF" : "volFriction = " + str(self.frictionVolume), 
                              "rand" : "randomness = " + str(randomness)}
            # Read the contents of the file
            with open(adressLua, 'r') as file:
                lines = file.readlines()
            # Modify the specific lines
            with open(adressLua, 'w') as file:
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line[:4] in linesToReplace:
                        file.write(linesToReplace[stripped_line[:4]] + '\n')
                    else:
                        file.write(line)

            # Run the simulation
            resultMPM = subprocess.run([exeMPM], capture_output=True, text=True)
            AdressBgeo.append(adressCurrentSimu)
        
        #Put the adress in the correct list
        if (typeSimu == "test") :
            self.AdressBgeoValid = AdressBgeo
        elif (typeSimu == "valid") :
            self.AdressBgeoTest = AdressBgeo
        else :
            self.AdressBgeoTrainReserve.append(AdressBgeo)
        print("simulation " + typeSimu + " " + str(self.young) + "_" + str(self.nu) + "_" + str(self.rho) + "_" + str(self.frictionAngle) + "_" + str(i+1) + "-" + str(nbSimu) + " done")
    
    def bgeoToDataPoints(self, adressBgeo) :
        positions = []
        for i in range(self.nbFrame) :
            filePath = adressBgeo + "/partio_" + str(i) + ".bgeo"
            
            #Create a Geometry object
            geo = hou.Geometry()
            geo.loadFromFile(filePath)
            
            #create list positions for the points for each increment of time
            positions_tps_fixe = []
            for point in geo.points() :
                position = list(point.position())
                positions_tps_fixe.append(position)
            positions.append(positions_tps_fixe)
        
        #create the array for particule type and gives a value (0 for points or 5?)
        particles_type = [6 for x in positions[0]]
        particle_parameters = [[self.young, self.nu, self.rho, self.frictionAngle] for x in positions[0]]
        return(positions, particles_type, particle_parameters) 
    
    def bgeoToDataVolume(self, nbPointsVolume) :
        #Input : nb of points we want for the volume and folder path to the simulation
        #Output : points to represent the volume and the rigt type for an obstacle
        filePath = self.AdressBgeoObject
        
        #Create a Geometry object
        geo = hou.Geometry()
        geo.loadFromFile(filePath)
            
        # Get the bounding box of the geometry
        bbox = geo.boundingBox() 
        min_pos = bbox.minvec()
        max_pos = bbox.maxvec()
        
        #Create an array of points from a volume
        points = []
        max_try = 1000000  #secu if no volume
        nb_try = 0
        while (len(points) < nbPointsVolume and nb_try < max_try) :
            randPoint = hou.Vector3(np.random.uniform(min_pos[0], max_pos[0]),
                                    np.random.uniform(min_pos[1], max_pos[1]),
                                    np.random.uniform(min_pos[2], max_pos[2]))
            #Look if the point is inside the volume or not
            sop = hou.node("/obj/geo1")  # Adjust this to your geometry node path
            test_geo = sop.createNode("testgeometry")
            test_geo.setPosition(randPoint)
            result = geo.intersect(randPoint)
            test_geo.destroy()
            result != hou.Vector3()
            if result:
                points.append(list(randPoint.position()))
            nb_try += 1
        #create the array for particule type and gives a value 4
        particle_type_vol = [0 for x in points]
        particle_parameters = [self.frictionVolume for x in points[0]]
        return(points, particle_type_vol, particle_parameters) 

    def bgeoToNpz(self, nbPointsVolume, adressNpz) :
        """
        Input : nb points in the volume and adress where npz stored
        Put all the information contained in the different bgeo file in right npz file
        Output : Confirmation transfer done
        """
        if (self.adressBgeoObject != "") :
            positionsVol, particleTypeVol = self.bgeoToDataVolume(nbPointsVolume)
        else :
            positionsVol, particleTypeVol, particleParVol = [], [], []
        # Transform bgeo file for training in npz
        for setSimuTrain in self.AdressBgeoTrainReserve :
            i = 0
            dataset = {}
            for simuTrain in setSimuTrain :
                positionsPoints, particleTypePoint, particlePar = self.bgeoToDataPoints(simuTrain)
                positions = positionsVol + positionsPoints 
                particleType = particleTypeVol + particleTypePoint
                particlePar += particleParVol
                data = (np.array(positions), np.array(particleType), np.array(particlePar))
                name = "simulation_trajectory_" + str(i)
                i += 1
                dataset[name] = data
            folderPathNpz = adressNpz + "train" + str(self.young) + "_" + str(self.nu) + "_" + str(self.rho) + "_" + str(self.frictionAngle) + str(len(setSimuTrain))
            np.savez_compressed(folderPathNpz, **dataset)
            folderPathNpz += ".npz"
            self.AdressNpzTrainReserve.append(folderPathNpz)

        # Transform bgeo file for validation in npz
        i = 0
        dataset = {}
        for simuValid in self.AdressBgeoValid :
            positionsPoints, particleTypePoint, particlePar = self.bgeoToDataPoints(simuValid)
            positions = positionsVol + positionsPoints 
            particleType = particleTypeVol + particleTypePoint
            particlePar += particleParVol
            data = (np.array(positions), np.array(particleType), np.array(particlePar))
            name = "simulation_trajectory_" + str(i)
            i += 1
            dataset[name] = data
        folderPathNpz = adressNpz + "valid" + str(self.young) + "_" + str(self.nu) + "_" + str(self.rho) + "_" + str(self.frictionAngle) + str(len(self.AdressBgeoValid))
        np.savez_compressed(folderPathNpz, **dataset)
        folderPathNpz += ".npz"
        self.AdressNpzValid = folderPathNpz

        # Transform bgeo file for ttest in npz
        i = 0
        dataset = {}
        for simuTest in self.AdressBgeoTest :
            positionsPoints, particleTypePoint, particlePar = self.bgeoToDataPoints(simuTest)
            positions = positionsVol + positionsPoints 
            particleType = particleTypeVol + particleTypePoint
            particlePar += particleParVol
            data = (np.array(positions), np.array(particleType), np.array(particlePar))
            name = "simulation_trajectory_" + str(i)
            i += 1
            dataset[name] = data
        folderPathNpz = adressNpz + "test" + str(self.young) + "_" + str(self.nu) + "_" + str(self.rho) + "_" + str(self.frictionAngle) + "_" + str(len(self.AdressBgeoTest))
        np.savez_compressed(folderPathNpz, **dataset)
        folderPathNpz += ".npz"
        self.AdressNpzTest = folderPathNpz
        print("Transformation bgeo to npz done")

    def createDataset(self, adressBgeo = '/media/user/Volume/granular_collapse_GNS_dyn/', adressNpz = '/home/user/Documents/Baptiste/surrogate_modelling/gns/examples/granular_collapse/datasets/', adressLua = '/home/user/Documents/Baptiste/surrogate_modelling/gns/myCode/granular_collapse_gns.lua', 
                       nbPointsVolume = 1000, nbSimuTrain = [10, 20], nbSimuTest = 4, nbSimuValid = 4, randomnessTrain = 0, randomnessValid = 0 ,
                       exeMPM = '/home/user/Documents/Baptiste/surrogate_modelling/gns/myCode/exeMPM.sh') :
        self.launchSimulation(adressBgeo, exeMPM, adressLua, nbSimuTest, "test")
        self.launchSimulation(adressBgeo, exeMPM, adressLua, nbSimuValid, "valid",randomnessValid)
        for nbSimu in nbSimuTrain :
            self.launchSimulation(adressBgeo, exeMPM, adressLua, nbSimu, "train", randomnessTrain)
        self.bgeoToNpz(nbPointsVolume, adressNpz)
    
    def trainGNS(self, nbTrainingSteps, nbTrainingStepsToAdd, exeGNStrain = '/home/user/Documents/Baptiste/surrogate_modelling/gns/myCode/runGNStrain.sh', exeGNSretrain = '/home/user/Documents/Baptiste/surrogate_modelling/gns/myCode/runGNSretrain.sh', adressNpz = '/home/user/Documents/Baptiste/surrogate_modelling/gns/examples/granular_collapse/datasets/') :
        #Copy the right training set
        shutil.copy(self.AdressNpzTrainReserve[self.AdressTrain], adressNpz + "train.npz")
        #Modify the bash file to have the right one 
        if nbTrainingSteps != 0 :
            linesToReplace = {"MODEL_FILE=" : "MODEL_FILE=\"model-" + str(nbTrainingSteps) + ".pt\"",
                              "TRAIN_STATE": "TRAIN_STATE_FILE=\"train_state-" + str(nbTrainingSteps) + ".pt\"", 
                              "NTRAINING_S" : "NTRAINING_STEPS=" + str((nbTrainingSteps+nbTrainingStepsToAdd))}
            # Read the contents of the file
            with open(exeGNSretrain, 'r') as file:
                lines = file.readlines()
            # Modify the specific lines
            with open(exeGNSretrain, 'w') as file:
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line[:11] in linesToReplace:
                        file.write(linesToReplace[stripped_line[:11]] + '\n')
                    else:
                        file.write(line)
            command = f"conda run -n GPU_pytorch1 bash {exeGNSretrain}"
        else : 
            with open(exeGNStrain, 'r') as file:
                lines = file.readlines()

            # Modify the line to use the right model
            with open(exeGNStrain, 'w') as file:
                for line in lines:
                    if line.strip()[:11] == "NTRAINING_S":
                        file.write("NTRAINING_STEPS=" + str(nbTrainingStepsToAdd) + '\n')
                    else:
                        file.write(line)
            command = f"conda run -n GPU_pytorch1 bash {exeGNStrain}"
        #Run the actual trtaining cycle
        resultGNS = subprocess.run(command, capture_output=True, text=True, shell=True)
            
    def validGNS(self, nbTrainingSteps, exeGNSvalid = '/home/user/Documents/Baptiste/surrogate_modelling/gns/myCode/runGNSvalid.sh', adressNpz = '/home/user/Documents/Baptiste/surrogate_modelling/gns/examples/granular_collapse/datasets/') :
        """
        Input : the nbTraining cycle to know where we are in term of training, the rest is already known
        Output : modif of loss value of the object
        """
        shutil.copy(self.AdressNpzValid, adressNpz + "valid.npz")
        with open(exeGNSvalid, 'r') as file:
            lines = file.readlines()

        # Modify the line to use the right model
        with open(exeGNSvalid, 'w') as file:
            for line in lines:
                if line.strip()[:11] == "MODEL_FILE=":
                    file.write("MODEL_FILE=\"model-" + str(nbTrainingSteps) + ".pt\"" + '\n')
                else:
                    file.write(line)
        command = f"conda run -n GPU_pytorch1 bash {exeGNSvalid}"
        resultGNS = subprocess.run(command, capture_output=True, text=True, shell=True)
        #Find the loss in the log
        loss = resultGNS.stdout[-5:]
        i = 0
        while loss[:2] != ": " and i < 100 :
            i += 1
            loss = resultGNS.stdout[-(5+i):]
        if i == 100 :
            print("No value of loss found during validation")
        loss = float(loss[1:])
        self.loss = loss
    
    def testGNS(self, nbTrainingSteps, exeGNStest = '/home/user/Documents/Baptiste/surrogate_modelling/gns/myCode/runGNStest.sh', adressNpz = '/home/user/Documents/Baptiste/surrogate_modelling/gns/examples/granular_collapse/datasets/') :
        """
        Input : the nbTraining cycle to know where we are in term of training, use same bash as valid becvause same thing informatiquement
        Output : loss on the test set
        """
        shutil.copy(self.AdressNpzTest, adressNpz + "test.npz")
        with open(exeGNStest, 'r') as file:
            lines = file.readlines()

        # Modify the line to use the right model
        with open(exeGNStest, 'w') as file:
            for line in lines:
                if line.strip()[:11] == "MODEL_FILE=":
                    file.write("MODEL_FILE=\"model-" + str(nbTrainingSteps) + ".pt\"" + '\n')
                else:
                    file.write(line)
        command = f"conda run -n GPU_pytorch1 bash {exeGNStest}"
        resultGNS = subprocess.run(command, capture_output=True, text=True, shell=True)
        #Find the loss in the log
        loss = resultGNS.stdout[-5:]
        i = 0
        while loss[:2] != ": " and i < 100 :
            i += 1
            loss = resultGNS.stdout[-(5+i):]
        if i == 100 :
            print("No value of loss found during validation")
        loss = float(loss[1:])
        return(loss)
    
    def rolloutGNS(self, nbTrainingSteps, exeGNSrollout = '/home/user/Documents/Baptiste/surrogate_modelling/gns/myCode/runGNSrollout.sh') :
        """
        Input : the nbTraining cycle to know where we are in term of training, use same bash as valid becvause same thing informatiquement
        Output : create one rollout as a gif
        """
        with open(exeGNSrollout, 'r') as file:
            lines = file.readlines()

        # Modify the line to use the right model
        with open(exeGNSrollout, 'w') as file:
            for line in lines:
                if line.strip()[:11] == "MODEL_FILE=":
                    file.write("MODEL_FILE=\"model-" + str(nbTrainingSteps) + ".pt\"" + '\n')
                else:
                    file.write(line)
        command = f"conda run -n GPU_pytorch1 bash {exeGNSrollout}"
        resultGNS = subprocess.run(command, capture_output=True, text=True, shell=True)
    
    def signature(self) :
        return(str(int(self.young)) + "_" + str(int(self.nu * 100)) + "_" + str(int(self.rho)) + "_" + str(int(self.frictionAngle)))

def testClassSimulations() : 
    simu = simulations()
    simu.createDataset(nbSimuTest=1, nbSimuValid=1, nbSimuTrain=[1])
    simu.trainGNS(0,1)
    print("training done")
    simu.validGNS(1)
    print("loss lors de la validation : " + str(simu.loss))
    simu.rolloutGNS(1)
    print("rollout done")
    print("loss lors du test : " + str(simu.testGNS(1)))





