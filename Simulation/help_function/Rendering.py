import numpy as np
import slicer
import qt
import vtk
import SimpleITK as sitk


l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

class helpfunction:
    pass

def onNodeAdded(caller, event, calldata):
    node = calldata
    if isinstance(node, slicer.vtkMRMLVolumeNode):
        # Call showVolumeRendering using a timer instead of calling it directly
        # to allow the volume loading to fully complete.
        qt.QTimer.singleShot(0, lambda: showVolumeRendering(node))

def showVolumeRenderingCT(volumeNode):
    volRenLogic = slicer.modules.volumerendering.logic()
    displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
    displayNode.SetVisibility(True)
    displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName('CT-Chest-Contrast-Enhanced'))


def showVolumeRendering(volumeNode):
    #slicer.util.loadNodeFromFile("C:/Users/FUS/Desktop/Simulation/Colormap.vp", "TransferFunctionFile", returnNode=False)
    #preset = slicer.mrmlScene.GetNodeByID('vtkMRMLVolumePropertyNode1')
    volRenLogic = slicer.modules.volumerendering.logic()
    displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
    propertyNode = displayNode.GetVolumePropertyNode()

    opacityTransfer = vtk.vtkPiecewiseFunction()
    opacityTransfer.AddPoint(0,0)
    opacityTransfer.AddPoint(0.15,0.03)
    opacityTransfer.AddPoint(0.3,0.15)
    opacityTransfer.AddPoint(0.5,0.3)
    opacityTransfer.AddPoint(0.8,0.8)
    opacityTransfer.AddPoint(0.9,0.9)


    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(0.17, 0.1,0.1,1.0)
    ctf.AddRGBPoint(0.3, 0.2,1.0,0.2)
    ctf.AddRGBPoint(0.5, 1.0,0.5,0.0)
    ctf.AddRGBPoint(0.8, 1.0,0.0,0.0)

    propertyNode.SetColor(ctf)
    propertyNode.SetScalarOpacity(opacityTransfer)

    slicer.mrmlScene.AddNode(propertyNode)
    displayNode.UnRegister(volRenLogic)
    #volRenLogic.UpdateDisplayNodeFromVolumeNodes(propertyNode, volumeNode)
    # displayNode.SetVisibility(True)
    # displayNode.GetVolumePropertyNode()
    # displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName('CT-Chest-Contrast-Enhanced'))


def saveITK(path, itkImage):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(itkImage)