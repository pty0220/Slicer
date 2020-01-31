import os
import unittest
import helpfunction as hlp
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

#
# Simulation
#


def createHLayout(elements):
  rowLayout = qt.QHBoxLayout()
  for element in elements:
    rowLayout.addWidget(element)
  return rowLayout


class Simulation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Simulation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["TY Park (KIST)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an Acoustic simulation module for the transcranial focused ultrasound. This module use the K-Wave background.  
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by TY Park, KIST (Korea Institution of Science and Technology.
And it was funded by grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# SimulationWidget
#

class SimulationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...
    medicalimageButton = ctk.ctkCollapsibleButton()
    medicalimageButton.text = "Input CT image"
    self.layout.addWidget(medicalimageButton)

    medicalimageFormLayout = qt.QFormLayout(medicalimageButton)


    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Transducer Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    medicalimageFormLayout.addRow("Input CT Volume: ", self.inputSelector)

    #
    # output volume selector
    #
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = True
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene( slicer.mrmlScene )
    self.outputSelector.setToolTip( "Pick the output to the algorithm." )
    #parametersFormLayout.addRow("Output Volume: ", self.outputSelector)

    #
    # threshold value
    #
    self.ROC = ctk.ctkSliderWidget()
    self.ROC.singleStep = 0.1
    self.ROC.minimum = 0
    self.ROC.maximum = 100
    self.ROC.value = 71
    parametersFormLayout.addRow("ROC", self.ROC)


    self.width = ctk.ctkSliderWidget()
    self.width.singleStep = 0.1
    self.width.minimum = 0
    self.width.maximum = 100
    self.width.value = 71
    parametersFormLayout.addRow("Width", self.width)


    self.freq = ctk.ctkSliderWidget()
    self.freq.singleStep = 1
    self.freq.minimum = 20
    self.freq.maximum = 10000
    self.freq.value = 200
    parametersFormLayout.addRow("kHz", self.freq)

    self.test = ctk.ctkDoubleSpinBox()
    self.test.value = 1.0

    self.test2 = ctk.ctkDoubleSpinBox()
    self.test2.value = 2.0

    self.test3 = ctk.ctkDoubleSpinBox()
    self.test3.value = 3.0

    self.label1 =  qt.QLabel("Test1")
    self.label1.setFixedWidth(50)

    rowLayout = qt.QHBoxLayout()
    rowLayout.addWidget(self.label1)
    rowLayout.addWidget(self.test)
    rowLayout.addWidget(self.test2)

    parametersFormLayout.addRow(rowLayout)

    #parametersFormLayout.insertRow(2, self.test)


    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Run simulation")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = True
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    #self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode()

  def onApplyButton(self):
    print("Transducer parameters >> /n")
    print("ROC:"+str(self.ROC.value))
    print("Width:"+str(self.width.value))
    print("ROC:"+str(self.freq.value))

    skull, skull_actor = hlp.read_skull_vtk("/Users/home/Desktop/Simulation/Simulation/SMA_matlab_skull_transform0.vtk", 0.5, [0.8, 0.8, 0.8])

    model = slicer.vtkMRMLModelNode()
    model.SetAndObservePolyData(skull)

    ## set model node properties
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetColor(0.8, 0.8, 0.8)
    modelDisplay.BackfaceCullingOff()
    modelDisplay.SetOpacity(0.5)
    modelDisplay.SetPointSize(3)

    ## mode node display
    modelDisplay.SetSliceIntersectionVisibility(True)
    modelDisplay.SetVisibility(True)
    slicer.mrmlScene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())

    ## model node set name
    slicer.mrmlScene.AddNode(model).SetName("Skull")

    ## read vtk object convert to vtkimage
    result_image = hlp.vtk_grid2image("/Users/home/Desktop/Simulation/Simulation/SMA_full_0.vtk")
    result_volume = slicer.vtkMRMLScalarVolumeNode()
    result_volume.SetAndObserveImageData(result_image)

    ## image to 2D scene
    defaultVolumeDisplayNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeDisplayNode")
    defaultVolumeDisplayNode.AutoWindowLevelOn()
    defaultVolumeDisplayNode.SetVisibility(True)
    defaultVolumeDisplayNode.AddWindowLevelPresetFromString("ColdToHotRainbow")
    slicer.mrmlScene.AddDefaultNode(defaultVolumeDisplayNode)
    result_volume.SetAndObserveDisplayNodeID(defaultVolumeDisplayNode.GetID())
    defaultVolumeDisplayNode.SetInputImageDataConnection(result_volume.GetImageDataConnection())

    ## volume node set name
    slicer.mrmlScene.AddNode(result_volume).SetName("simulation_result")

    ## volume rendering
    hlp.showVolumeRendering(result_volume)
    slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, hlp.onNodeAdded)

#
# SimulationLogic
#

class SimulationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def run(self, inputVolume, outputVolume, imageThreshold, enableScreenshots=0):
    """
    Run the actual algorithm
    """

    if not self.isValidInputOutputData(inputVolume, outputVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

    # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    cliParams = {'InputVolume': inputVolume.GetID(), 'OutputVolume': outputVolume.GetID(), 'ThresholdValue' : imageThreshold, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('SimulationTest-Start','MyScreenshot',-1)

    logging.info('Processing completed')

    return True


class SimulationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_Simulation1()

  def test_Simulation1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = SimulationLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')


def setSlicerViews(backgroundID, foregroundID):
    mainWindow = slicer.util.mainWindow()
    if mainWindow is not None:
        layoutManager = slicer.app.layoutManager()
        if layoutManager is not None:
            makeSlicerLinkedCompositeNodes()

            slicer.util.setSliceViewerLayers(background=backgroundID, foreground=foregroundID, foregroundOpacity=0.5)

            layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
            slicer.util.resetSliceViews()

def makeSlicerLinkedCompositeNodes():
    # Set linked slice views  in all existing slice composite nodes and in the default node
    sliceCompositeNodes = slicer.util.getNodesByClass('vtkMRMLSliceCompositeNode')
    defaultSliceCompositeNode = slicer.mrmlScene.GetDefaultNodeByClass('vtkMRMLSliceCompositeNode')
    if not defaultSliceCompositeNode:
        defaultSliceCompositeNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLSliceCompositeNode')
        slicer.mrmlScene.AddDefaultNode(defaultSliceCompositeNode)
    for sliceCompositeNode in sliceCompositeNodes:
        sliceCompositeNode.SetLinkedControl(True)
    defaultSliceCompositeNode.SetLinkedControl(True)