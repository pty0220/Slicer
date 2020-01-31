import os
import unittest
import helpfunction as hlp
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
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


    ##########################################################################################
    # Input medical image
    ##########################################################################################
    medicalimageButton = ctk.ctkCollapsibleButton()
    medicalimageButton.text = "Input CT image"
    self.layout.addWidget(medicalimageButton)
    medicalimageFormLayout = qt.QFormLayout(medicalimageButton)

    # input volume selector
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


    # output volume selector
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
    medicalimageFormLayout.addRow("Output Volume: ", self.outputSelector)


    ##########################################################################################
    # Set Location Area
    ##########################################################################################
    locationCollapsibleButton = ctk.ctkCollapsibleButton()
    locationCollapsibleButton.text = "Set Locations"
    self.layout.addWidget(locationCollapsibleButton)
    locationFormLayout = qt.QFormLayout(locationCollapsibleButton)

    # Set Entry location
    self.entry = qt.QLabel("Entry Position")
    self.entry.setFixedWidth(100)
    locationFormLayout.addRow(self.entry)

    self.eXLabel =  qt.QLabel("X")
    self.eXLabel.setFixedWidth(10)
    self.eX = qt.QDoubleSpinBox()
    self.eX.setRange(-300,300)
    self.eX.value = -100

    self.eYLabel =  qt.QLabel("Y")
    self.eYLabel.setFixedWidth(10)
    self.eY = qt.QDoubleSpinBox()
    self.eY.setRange(-300,300)
    self.eY.value = 30

    self.eZLabel =  qt.QLabel("Z")
    self.eZLabel.setFixedWidth(10)
    self.eZ =  qt.QDoubleSpinBox()
    self.eZ.setRange(-300,300)
    self.eZ.value = 71

    entryLayout = qt.QHBoxLayout()
    entryLayout.addWidget(self.eXLabel)
    entryLayout.addWidget(self.eX)
    entryLayout.addWidget(self.eYLabel)
    entryLayout.addWidget(self.eY)
    entryLayout.addWidget(self.eZLabel)
    entryLayout.addWidget(self.eZ)
    locationFormLayout.addRow(entryLayout)

    # Set Target location
    self.target = qt.QLabel("Target Position")
    self.target.setFixedWidth(100)
    locationFormLayout.addRow(self.target)

    self.pXLabel = qt.QLabel("X")
    self.pXLabel.setFixedWidth(10)
    self.pX = qt.QDoubleSpinBox()
    self.pX.setRange(-300,300)
    self.pX.value = 100.0

    self.pYLabel = qt.QLabel("Y")
    self.pYLabel.setFixedWidth(10)
    self.pY = qt.QDoubleSpinBox()
    self.pY.setRange(-300,300)
    self.pY.value = -200

    self.pZLabel = qt.QLabel("Z")
    self.pZLabel.setFixedWidth(10)
    self.pZ = qt.QDoubleSpinBox()
    self.pZ.setRange(-300,300)
    self.pZ.value = 250

    targetLayout = qt.QHBoxLayout()
    targetLayout.addWidget(self.pXLabel)
    targetLayout.addWidget(self.pX)
    targetLayout.addWidget(self.pYLabel)
    targetLayout.addWidget(self.pY)
    targetLayout.addWidget(self.pZLabel)
    targetLayout.addWidget(self.pZ)
    locationFormLayout.addRow(targetLayout)

    self.setLocation = qt.QPushButton("Set Location")
    self.setLocation.toolTip = "Set entry and target location"
    self.setLocation.enabled = True
    locationFormLayout.addRow(self.setLocation)

    ##########################################################################################
    # Transducer parameters
    ##########################################################################################
    transducerCollapsibleButton = ctk.ctkCollapsibleButton()
    transducerCollapsibleButton.text = "Transducer Parameters"
    self.layout.addWidget(transducerCollapsibleButton)
    transducerFormLayout = qt.QFormLayout(transducerCollapsibleButton)

    self.ROC = qt.QDoubleSpinBox()
    self.ROC.setRange(-300, 300)
    self.ROC.value = 71

    self.width = qt.QDoubleSpinBox()
    self.width.setRange(-300, 300)
    self.width.value = 65

    self.freq = qt.QDoubleSpinBox()
    self.freq.setRange(-300, 300)
    self.freq.value = 0.2

    transducerFormLayout.addRow("ROC", self.ROC)
    transducerFormLayout.addRow("Width", self.width)
    transducerFormLayout.addRow("MHz", self.freq)

    self.simulationButton = qt.QPushButton("Run simulation")
    self.simulationButton.toolTip = "Run the algorithm."
    self.simulationButton.enabled = True
    transducerFormLayout.addRow(self.simulationButton)

    # connections
    self.simulationButton.connect('clicked(bool)', self.runSimulation)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    #self.onSelect()

  def cleanup(self):
    pass


  def runSimulation(self):
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
    result_image, pressure, extent = hlp.vtk_grid2image("/Users/home/Desktop/Simulation/Simulation/SMA_full_0.vtk")
    dimension = result_image.GetDimensions()
    reshapePressure = np.reshape(pressure, (dimension[2]-1,dimension[0]-1, dimension[1]-1))

    result_volume = slicer.vtkMRMLScalarVolumeNode()
    result_volume.SetAndObserveImageData(result_image)
    slicer.util.updateVolumeFromArray(result_volume, reshapePressure)
    result_volume.SetName("simulation_result")
    slicer.mrmlScene.AddNode(result_volume)
    result_volume.CreateDefaultDisplayNodes()
    result_volume.SetExtent(extent)
    slicer.util.setSliceViewerLayers(background=result_volume)

    ## image to 2D scene
    displaynode = result_volume.GetDisplayNode()
    displaynode.AutoWindowLevelOff()
    displaynode.SetWindowLevelMinMax(10,900)
    displaynode.AddWindowLevelPresetFromString("ColdToHotRainbow")
    # displaynode.SetInputImageDataConnection(result_volume.GetImageDataConnection())
    displaynode.SetVisibility(True)
    slicer.mrmlScene.AddNode(displaynode)
    #result_volume.SetAndObserveDisplayNodeID(displaynode.GetID())

    # volume rendering
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
