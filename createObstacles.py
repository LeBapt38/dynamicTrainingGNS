"""
Created on Mon June  19 18:02 2024

@author: Baptiste Guilleminot

Short file to define the obstacles used.
"""
import numpy as np
#import pyopenvdb as vdb

def createHalfPlane(nbPointsPerDirection : list, Origin, Basis, Size) :
    """
    Args : nb points representing the plane on each elements of the basis, the origin ("bottom left" of the plan), the basis of the plane (2 3D vectors repented by lists), size allong each component of the basis

    Output : file with the list of points to represent surface
    """
    Basis = np.array(Basis)
    dx = Size[0] * (1-0.00001) / nbPointsPerDirection[0]
    dy = Size[1] * (1-0.00001) / nbPointsPerDirection[1]
    points = []
    for i in range(nbPointsPerDirection[0]) :
        for j in range(nbPointsPerDirection[1]) :
            point = np.array(Origin)
            point += float(i) * dx * Basis[0]
            point += float(j) * dy * Basis[1]
            points.append(point)
    return points

def saveObject(object, filePath) : 
    with open(filePath, 'w') as file:
        for point in object:
            line = ' '.join(map(str, point)) + '\n'
            file.write(line)

"""
def datToVdb(pathToDat, nameObject, path = '/home/user/Documents/myJixie/jixie_lars/Data/LevelSets/Baptiste/') :
    #Charge the data
    points = []
    with open(pathToDat, 'r') as file : 
        lines = file.readlines()
        for line in lines[1:] :
            position = line.split()
            positionFloat = []
            for val in position :
                positionFloat.append(float(val))
            points.append(positionFloat)
    points = np.array(points)
    
    #Get the right dl
    dx = np.array([0,0])
    dy = np.array([0,0])
    dz = np.array([0,0])
    for i in range(len(points)-1) :
        point1, point2 = points[i], points[i+1]
        dr = np.absolute(point1 - point2)
        if dr[0] != 0 :
            dx += np.array([dr[0], 1])
        if dr[1] != 0 :
            dy += np.array([dr[1], 1])
        if dr[2] != 0 :
            dz += np.array([dr[2], 1])
    vecToScale = []
    dim = 0
    if dx[1] == 0 : 
        vecToScale.append([1,0,0])
        dx = 0
    else : 
        dx = dx[0]/dx[1]
        dim += 1
    if dy[1] == 0 : 
        vecToScale.append([0,1,0])
        dy = 0
    else : 
        dy = dy[0]/dy[1]
        dim += 1
    if dz[1] == 0 : 
        vecToScale.append([0,0,1])
        dz = 0
    else : 
        dz = dz[0]/dz[1]
        dim += 1
    dl = (dx+dy+dz)/dim
    vecToScale = np.array(vecToScale)
    points = list(points)

    # Scale up the object so it is a 3D object big enough to be read by MPM
    for point in points :
        for vec in vecToScale :
            for i in range(10) :
                numpPoint = np.array(point) 
                points.append(numpPoint + i*vec)

    # Create a vdb file from a set of points
    grid = vdb.FloatGrid()
    grid.transform = vdb.createLinearTransform(voxelSize=dl*2)
    grid.background = 0.0
    for point in points:
        voxel = grid.transform.worldToIndex(point)
        grid.setValueOn(voxel, 1.0)
    grid.prune()
    vdb.write(path + nameObject + ".vdb", grids=[grid])


    relativePathVdb = "LevelSets/Baptiste/" + nameObject + ".vdb"
    return relativePathVdb
"""