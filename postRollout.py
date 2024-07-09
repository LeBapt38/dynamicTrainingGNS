import vtk
import pandas as pd
import matplotlib.pyplot as plt

def read_vtu(file_path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints().GetData()
    num_points = points.GetNumberOfTuples()

    points_list = []
    for i in range(num_points):
        points_list.append(points.GetTuple2(i))

    df = pd.DataFrame(points_list, columns=["x", "y"])
    return df


#%%
with open('dataTraining/lossTraining1.txt', 'r') as file :
    lines = file.readlines()
LL = []
for line in lines : 
    a = line.split()
    L = []
    for b in a :
        L.append(float(b))
    LL.append(L)

loss1 = LL[0]
stdLoss1 = LL[1]
nbSteps1 = LL[2]

with open('dataTraining/lossTraining2.txt', 'r') as file :
    lines = file.readlines()
LL = []
for line in lines : 
    a = line.split()
    L = []
    for b in a :
        L.append(float(b))
    LL.append(L)

loss2 = LL[0]
stdLoss2 = LL[1]
nbSteps2 = LL[2]

with open('dataTraining/lossTrainingClassique.txt', 'r') as file :
    lines = file.readlines()
LL = []
for line in lines : 
    a = line.split()
    L = []
    for b in a :
        L.append(float(b))
    LL.append(L)

lossClassic = LL[0]
stdLossClassic = LL[1]
nbStepsClassic = LL[2]

fig, axs = plt.subplots(1,2)
fig.suptitle('Study of the evolution of the loss')
axs[0].plot(nbSteps1, loss1, c='b', label='Dynamic training 1')
axs[0].plot(nbSteps2, loss2, c='r', label='Dynamic training 2')
axs[0].plot(nbStepsClassic, lossClassic, c='orange', label='Classic training')
axs[0].set_yscale('log')
axs[0].set_xlabel("Number of training steps")
axs[0].set_ylabel("Average loss")
axs[0].legend(loc='upper right')

axs[1].plot(nbSteps1, stdLoss1, color = 'b', label='Dynamic training 1')
axs[1].plot(nbSteps2, stdLoss2, color = 'r', label='Dynamic training 2')
axs[1].plot(nbStepsClassic, stdLossClassic, color = 'orange', label='Classic training')
axs[1].set_yscale('log')
axs[1].set_xlabel("Number of training steps")
axs[1].set_ylabel("Standard deviation of the loss")
axs[0].legend(loc='upper right')

plt.tight_layout()

fig.savefig("dataTraining/studyLoss.png")

