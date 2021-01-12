import slicer
import Elastix
import time 
import numpy as np

scene = slicer.mrmlScene


MR_axial = slicer.util.getNode("MR_axial")
CT_axial = slicer.util.getNode("CT_axial")
CT_regi = slicer.util.getNode("CT_regi")



elastixLogic = Elastix.ElastixLogic()
parameterFilenames = elastixLogic.getRegistrationPresets()[1][Elastix.RegistrationPresets_ParameterFilenames]
print("4. Start registration using Elastix")
regi =  elastixLogic.registerVolumes(MR_axial, CT_axial, parameterFilenames = parameterFilenames ,outputVolumeNode = CT_regi)


print("5. Registration Done")

