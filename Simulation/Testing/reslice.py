import slicer
import time
import numpy as np

scene = slicer.mrmlScene

# Assign Raw data volume

print("1. Assign raw MR and CT data")
MR = slicer.util.getNode("MR")
CT = slicer.util.getNode("CT")


# Create new node for output volume

print("2. Create empty output volumes")
nodes = [slicer.vtkMRMLScalarVolumeNode() for idx in range(5)]

if (slicer.util.getNode("MR_axial") == None):
    scene.AddNode(nodes[0]).SetName("MR_axial")

if (slicer.util.getNode("CT_axial") == None):
    scene.AddNode(nodes[1]).SetName("CT_axial")

if (slicer.util.getNode("CT_regi") == None):
    scene.AddNode(nodes[2]).SetName("CT_regi")

# Assign new node for output volume
MR_axial = slicer.util.getNode("MR_axial")
CT_axial = slicer.util.getNode("CT_axial")
CT_regi = slicer.util.getNode("CT_regi")


# Make parameters for orientation
# Re-slice as Axial direction
print("3. Start reslice")
params_MR = {'inputVolume1' : MR.GetID(), 'outputVolume' : MR_axial.GetID(), 'orientation' : 'Axial'}


# Start re-slice (MR)
conv_MR = slicer.cli.run(slicer.modules.orientscalarvolume, None, params_MR)


# Make parameters for orientation
# Re-slice as Axial direction
params_CT = {'inputVolume1' : CT.GetID(), 'outputVolume' : CT_axial.GetID(), 'orientation' : 'Axial'}


# Start re-slice (CT)
conv_CT = slicer.cli.run(slicer.modules.orientscalarvolume, None,params_CT)

