import vtk
import pandas as pd

def read_vtu(file_path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints().GetData()
    num_points = points.GetNumberOfTuples()

    points_list = []
    for i in range(num_points):
        points_list.append(points.GetTuple3(i))

    df = pd.DataFrame(points_list, columns=["x", "y", "z"])
    return df


