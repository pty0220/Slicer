import sys
import os
import qt, ctk, slicer
import logging
import SimpleITK as sitk
import sitkUtils
import datetime

from slicer.ScriptedLoadableModule import *
from simulation_function import makeSimulation


current_path = os.path.dirname(__file__)
sys.path.append(current_path+'/help_function')
import Rendering as ren
from position_info import *

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
    self.parent.categories = ["FUS"]
    self.parent.dependencies = []
    self.parent.contributors = ["TY Park (KIST)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an acoustic simulation module for the transcranial focused ultrasound using K-Wave background.  
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by TY Park, KIST (Korea Institution of Science and Technology.
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
    # Set model name
    ##########################################################################################
    modelNameButton = ctk.ctkCollapsibleButton()
    modelNameButton.text = "Simulation name"
    self.layout.addWidget(modelNameButton)
    modelNameButtonLayout = qt.QFormLayout(modelNameButton)
    self.nameBox = qt.QLineEdit()
    now = datetime.datetime.now()
    self.nameBox.text = "TEST_"+str(now.minute)+"_"+str(now.second)
    modelNameButtonLayout.addRow("Model_name", self.nameBox)


    ##########################################################################################
    # Input medical image
    ##########################################################################################
    medicalimageButton = ctk.ctkCollapsibleButton()
    medicalimageButton.text = "Input CT image"
    self.layout.addWidget(medicalimageButton)
    medicalimageFormLayout = qt.QFormLayout(medicalimageButton)

    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input CT image." )
    medicalimageFormLayout.addRow("Input CT Volume: ", self.inputSelector)


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
    self.eX.value = 87.275

    self.eYLabel =  qt.QLabel("Y")
    self.eYLabel.setFixedWidth(10)
    self.eY = qt.QDoubleSpinBox()
    self.eY.setRange(-300,300)
    self.eY.value = -45.256

    self.eZLabel =  qt.QLabel("Z")
    self.eZLabel.setFixedWidth(10)
    self.eZ =  qt.QDoubleSpinBox()
    self.eZ.setRange(-300,300)
    self.eZ.value = 83.913

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
    self.pX.value = 49.694

    self.pYLabel = qt.QLabel("Y")
    self.pYLabel.setFixedWidth(10)
    self.pY = qt.QDoubleSpinBox()
    self.pY.setRange(-300,300)
    self.pY.value = -39.790

    self.pZLabel = qt.QLabel("Z")
    self.pZLabel.setFixedWidth(10)
    self.pZ = qt.QDoubleSpinBox()
    self.pZ.setRange(-300,300)
    self.pZ.value = 43.826

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
    self.setLocation.connect('clicked(bool)', self.setLocationPrint)



    ##########################################################################################
    # Transducer parameters
    ##########################################################################################
    transducerCollapsibleButton = ctk.ctkCollapsibleButton()
    transducerCollapsibleButton.text = "Transducer & voxelization Parameters"
    self.layout.addWidget(transducerCollapsibleButton)
    transducerFormLayout = qt.QFormLayout(transducerCollapsibleButton)

    self.ROC = qt.QDoubleSpinBox()
    self.ROC.setRange(0, 300)
    self.ROC.value = 71

    self.width = qt.QDoubleSpinBox()
    self.width.setRange(0, 300)
    self.width.value = 65

    self.freq = qt.QDoubleSpinBox()
    self.freq.setRange(0, 2)
    self.freq.value = 0.25

    self.PPW = qt.QDoubleSpinBox()
    self.PPW.setRange(2, 20)
    self.PPW.value = 10

    self.sizeSmallRadioButton = qt.QRadioButton()
    self.sizeSmallRadioButton.text = 'Small'
    self.sizeSmallRadioButton.checked = True

    self.sizeLargeRadioButton = qt.QRadioButton()
    self.sizeLargeRadioButton.text = 'Large'
    #self.sizeBigRadioButton.setchecked(False)

    boundaryLayout = qt.QHBoxLayout()
    boundaryLayout.addWidget(self.sizeSmallRadioButton)
    boundaryLayout.addWidget(self.sizeLargeRadioButton)

    transducerFormLayout.addRow("ROC", self.ROC)
    transducerFormLayout.addRow("Width", self.width)
    transducerFormLayout.addRow("MHz", self.freq)
    transducerFormLayout.addRow("Point per wavelength", self.PPW)
    transducerFormLayout.addRow("Boundary size", boundaryLayout)

    self.transButton = qt.QPushButton("Transducer and Voxelization")
    self.transButton.toolTip = "Make transducer and voxelization for simulation."
    self.transButton.enabled = True
    transducerFormLayout.addRow(self.transButton)

    self.transButton.connect('clicked(bool)', self.voxelization)

    ##########################################################################################
    # Simulation parameters
    ##########################################################################################
    simulationCollapsibleButton = ctk.ctkCollapsibleButton()
    simulationCollapsibleButton.text = "Simulation Parameters"
    self.layout.addWidget(simulationCollapsibleButton)
    simulationFormLayout = qt.QFormLayout(simulationCollapsibleButton)

    self.endTime = qt.QDoubleSpinBox()
    self.endTime.setRange(0, 300)
    self.endTime.value = 150

    self.CFL = qt.QDoubleSpinBox()
    self.CFL.setRange(0, 1)
    self.CFL.value = 0.1

    simulationFormLayout.addRow("End time (μs)", self.endTime)
    simulationFormLayout.addRow("CFL", self.CFL)

    self.simulationButton = qt.QPushButton("Run simulation")
    self.simulationButton.toolTip = "Run the algorithm."
    self.simulationButton.enabled = True

    # Run simulation button
    simulationFormLayout.addRow(self.simulationButton)
    self.simulationButton.connect('clicked(bool)', self.runSimulation)

    # Add vertical spacer
    self.layout.addStretch(1)
    self.simul = makeSimulation()

    # Make model dir
    self.newdir = current_path +'/'+self.nameBox.text
    if not os.path.exists(self.newdir):
      os.makedirs(self.newdir)



    # Refresh Apply button state
    #self.onSelect()

  def cleanup(self):
    pass


  def setLocationPrint(self):

    print("Entry point")
    print(self.eX.value, self.eY.value, self.eZ.value)

    print("Target point")
    print(self.pX.value, self.pY.value, self.pZ.value)

    fiducialNode_entry = slicer.vtkMRMLAnnotationFiducialNode()
    fiducialNode_entry.SetFiducialWorldCoordinates([self.eX.value, self.eY.value, self.eZ.value])
    fiducialNode_entry.SetName('Entry')
    fiducialNode_entry.Initialize(slicer.mrmlScene)

    fiducialNode_target = slicer.vtkMRMLAnnotationFiducialNode()
    fiducialNode_target.SetFiducialWorldCoordinates([self.pX.value, self.pY.value, self.pZ.value])
    fiducialNode_target.SetName('Target')
    fiducialNode_target.Initialize(slicer.mrmlScene)


  def voxelization(self):

    self.simul.source_freq = self.freq.value*1e6
    self.simul.ROC = self.ROC.value
    self.simul.width = self.width.value
    self.simul.points_per_wavelength = self.PPW.value
    if self.sizeLargeRadioButton.checked == True:
      self.simul.boundary = 1

    print("############################")
    print("Transducer parameters")
    print("############################")
    print("ROC   :"+str(self.ROC.value))
    print("Width :"+str(self.width.value))
    print("Freq  :"+str(self.freq.value))
    print("PPW  :"+str(self.PPW.value))
    print("############################")

    tran_pose = (self.eX.value, self.eY.value, self.eZ.value)
    target_pose = (self.pX.value, self.pY.value, self.pZ.value)

    # read current node and covert to simple itk image.
    inputVolumeNode = self.inputSelector.currentNode() # Get node from the selected volume from GUI
    inputVolumeNodeSceneAddress = inputVolumeNode.GetScene().GetAddressAsString("").replace('Addr=', '')
    inputVolumeNodeFullITKAddress = 'slicer:{0}#{1}'.format(inputVolumeNodeSceneAddress, inputVolumeNode.GetID())
    itk_image = sitk.ReadImage(inputVolumeNodeFullITKAddress)

    # start processing
    skullCrop_arr, skullCrop_itk, image_cordi, p0, trans_itk = self.simul.preprocessing(itk_image, tran_pose, target_pose)

    print("Push to node")
    skullNode = sitkUtils.PushVolumeToSlicer(skullCrop_itk, None)
    skullNode.SetName("skull_crop")
    ren.showVolumeRenderingCT(skullNode)

    transNode = sitkUtils.PushVolumeToSlicer(trans_itk, None)
    transNode.SetName("transducer")
    ren.showVolumeRenderingCT(transNode)

    self.skullNode = skullNode
    self.skullCrop_arr = skullCrop_arr
    self.skullCrop_itk = skullCrop_itk
    self.image_cordi = image_cordi
    self.p0 = p0
    self.trans_itk =trans_itk

    ren.saveITK(self.newdir+"/Cropped_skull.nii", self.skullCrop_itk)



  def runSimulation(self):

    self.simul.CFL = self.CFL.value
    self.simul.end_time = self.endTime.value*(1e-6)

    print("############################")
    print("Simulation condition")
    print("############################")
    print("CFL   :"+str(self.CFL.value))
    print("End time :"+str(self.endTime.value))
    print("############################")


    # Preform simulation
    result_itk = self.simul.run_simulation(self.skullCrop_arr, self.image_cordi, self.p0, self.newdir)

    # Render the result
    resultNode = sitkUtils.PushVolumeToSlicer(result_itk, None)
    resultNode.SetName("simulation_result")

    resultDisplayNode = resultNode.GetDisplayNode()
    resultDisplayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt')

    ren.showVolumeRendering(resultNode)
    slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, ren.onNodeAdded)
    slicer.util.setSliceViewerLayers(background=resultNode, foreground=self.skullNode, foregroundOpacity=0.35, fit=True)

    # Save as nii
    ren.saveITK(self.newdir+"/Result.nii", result_itk)
    ren.saveITK(self.newdir+"/Transducer.nii", self.trans_itk)




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
    # import SampleData
    # SampleData.downloadFromURL(
    #   nodeNames='FA',
    #   fileNames='FA.nrrd',
    #   uris='http://slicer.kitware.com/midas3/download?items=5767')
    # self.delayDisplay('Finished with download and loading')
    #
    # volumeNode = slicer.util.getNode(pattern="FA")
    # logic = SimulationLogic()
    # self.assertIsNotNone( logic.hasImageData(volumeNode) )
    # self.delayDisplay('Test passed!')
