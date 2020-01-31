import vtk
import numpy as np
import slicer
import qt
from vtk.util import numpy_support as ns


l2n = lambda l: np.array(l)
n2l = lambda n: list(n)


def setSlicerViews(backgroundID):
    mainWindow = slicer.util.mainWindow()
    if mainWindow is not None:
        layoutManager = slicer.app.layoutManager()
        if layoutManager is not None:
            makeSlicerLinkedCompositeNodes()

            slicer.util.setSliceViewerLayers(background=backgroundID)

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



def onNodeAdded(caller, event, calldata):
    node = calldata
    if isinstance(node, slicer.vtkMRMLVolumeNode):
        # Call showVolumeRendering using a timer instead of calling it directly
        # to allow the volume loading to fully complete.
        qt.QTimer.singleShot(0, lambda: showVolumeRendering(node))



def showVolumeRendering(volumeNode):
    print("Show volume rendering of node " + volumeNode.GetName())
    slicer.util.loadNodeFromFile("/Users/home/Desktop/Simulation/Colormap.vp", "TransferFunctionFile", returnNode=True)
    preset = slicer.mrmlScene.GetNodeByID('vtkMRMLVolumePropertyNode1')

    volRenLogic = slicer.modules.volumerendering.logic()
    displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
    displayNode.SetVisibility(True)

    displayNode.GetVolumePropertyNode().Copy(preset)

   #displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName('CT-Chest-Contrast-Enhanced'))



def vtk_grid2image(vtk_filename):

    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(vtk_filename)
    reader.Update()

    grid = reader.GetOutput()

    celldata = grid.GetCellData()

    Array0 = celldata.GetArray(0)
    Array1 = celldata.GetArray(1)

    Re = ns.vtk_to_numpy(Array0)
    Im = ns.vtk_to_numpy(Array1)

    pressure = np.absolute(Re+1j*Im)
    pressure[np.isnan(pressure)] = 0

    maximum_pressure = max(pressure)
    pressure = pressure/maximum_pressure ### maximum value SMA_K targeting normalize
    pressure = pressure*1000
    pressure_vtk = ns.numpy_to_vtk(pressure,deep=True, array_type=vtk.VTK_FLOAT)

    grid.GetCellData().SetScalars(pressure_vtk)
    bounds = grid.GetBounds()
    dimension  = grid.GetDimensions()
    extent = grid.GetExtent()
    print(extent)
    vtk_coordi = grid.GetXCoordinates()
    xcoordi = ns.vtk_to_numpy(vtk_coordi)
    space = 1000*(xcoordi[1]- xcoordi[0])
    bounds = l2n(bounds)*1000
    print(bounds)
    print(space)
    image = vtk.vtkImageData()
    image.DeepCopy(grid)
    image.SetDimensions(dimension)
    image.SetExtent(extent)
    image.SetSpacing(space,space,space)
    image.SetOrigin(bounds[0], bounds[2], bounds[4])

    return image, pressure, extent


def read_skull_vtk(filename,  opacity, color):

    ren = vtk.vtkRenderer()
    readerstl = vtk.vtkPolyDataReader()
    readerstl.SetFileName(filename)
    readerstl.Update()

    reader = readerstl.GetOutput()


    STLmapper = vtk.vtkPolyDataMapper()
    STLmapper.SetInputData(reader)

    STLactor = vtk.vtkActor()
    STLactor.SetMapper(STLmapper)
    STLactor.GetProperty().SetOpacity(opacity)
    STLactor.GetProperty().SetColor(color)

    return reader, STLactor


def read_skull(filename,  opacity, color):

    ren = vtk.vtkRenderer()
    readerstl = vtk.vtkSTLReader()
    readerstl.SetFileName(filename)
    readerstl.Update()

    reader = readerstl.GetOutput()


    STLmapper = vtk.vtkPolyDataMapper()
    STLmapper.SetInputData(reader)

    STLactor = vtk.vtkActor()
    STLactor.SetMapper(STLmapper)
    STLactor.GetProperty().SetOpacity(opacity)
    STLactor.GetProperty().SetColor(color)

    return reader, STLactor


def addLine(p1, p2, color=[0.0, 0.0, 1.0], opacity=1.0):
    line = vtk.vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty()

    return line, actor


def addPoint(p, color=[0.0,0.0,0.0], radius=0.5):

    point = vtk.vtkSphereSource()
    point.SetCenter(p)
    point.SetRadius(radius)
    point.SetPhiResolution(100)
    point.SetThetaResolution(100)


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(point.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)



    return point, actor


def make_centerline_target(skull, target, centerline_length):
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(skull)
    pointLocator.BuildLocator()
    id = pointLocator.FindClosestPoint(target)
    point_s = skull.GetPoint(id)

    vector = l2n(point_s) - l2n(target)
    centerline_vector = vector / np.linalg.norm(vector)
    centerline_target = n2l(l2n(target) + centerline_length * centerline_vector )
    middle_target =  n2l(l2n(target) - 10 * centerline_vector )
    deep_target = n2l(l2n(target) - 20 * centerline_vector )
    return centerline_target, centerline_vector , point_s, middle_target, deep_target


def make_analysis_rage2(num_pts, radius, range_angle, opacity=0.25,centerline_vector=[0,0,0], Target=[0,0,0]):

    # num_pts     : number of transducer (setting value, int)
    # range_angle : analysis range angle (setting value, degree)
    # radius      : transducer focal size

    # calculate height of analysis range
    # Pythagorean theorem (under three lines)
    h_wid = radius*np.sin(np.deg2rad(range_angle))
    p_height = radius ** 2 - h_wid ** 2
    height_from_center = np.sqrt(p_height)
    # height of analysis range
    height = radius - height_from_center

    # ratio height/radius*2
    rate = height / (radius * 2)

    # make evenly distributed sphere
    indices_theta = np.arange(0, num_pts, dtype=float)
    indices_phi = np.linspace(0, num_pts * rate, num=num_pts)  ## define transdcuer's height as ratio

    phi = np.arccos(1 - 2 * indices_phi / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices_theta

    # x,y,z is coordination of evenly distributed sphere
    # multiply radius(focal size)
    x, y, z = np.cos(theta) * np.sin(phi)*radius, np.sin(theta) * np.sin(phi)*radius, np.cos(phi)*radius;

    coordi  = np.zeros((num_pts, 3))
    dis_min = np.zeros((num_pts,1))
    coordi[:, 0] = x
    coordi[:, 1] = y
    coordi[:, 2] = z


    points = vtk.vtkPoints()

    for i in range(len(x)):
        points.InsertNextPoint(x[i],y[i],z[i])
        dis = np.sqrt(np.sum(np.power((coordi-coordi[i,:]),2), axis =1))
        dis_min[i] = np.min(dis[dis>0])


    dis_average = np.average(dis_min)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    # To create surface of a sphere we need to use Delaunay triangulation
    d3D = vtk.vtkDelaunay3D()
    d3D.SetInputData( poly ) # This generates a 3D mesh

    # We need to extract the surface from the 3D mesh
    dss = vtk.vtkDataSetSurfaceFilter()
    dss.SetInputConnection( d3D.GetOutputPort() )
    dss.Update()

    # Now we have our final polydata
    spherePoly = dss.GetOutput()


    # rotation of analysis range
    center_vector = [1, 0, 0]
    unit_vector = centerline_vector / np.linalg.norm(centerline_vector)
    xy_unit_vector = l2n((unit_vector[0], unit_vector[1], 0))

    if n2l(xy_unit_vector) == [0, 0, 0]:
        xy_angle = 0.0
        z_angle = 90.0
    else:
        xy_angle = np.rad2deg(np.arccos(
            np.dot(center_vector, xy_unit_vector) / (np.linalg.norm(center_vector) * np.linalg.norm(xy_unit_vector))))
        z_angle = np.rad2deg(np.arccos(
            np.dot(xy_unit_vector, unit_vector) / (np.linalg.norm(xy_unit_vector) * np.linalg.norm(unit_vector))))
    if unit_vector[2] < 0:
        z_angle = -z_angle
    if unit_vector[1] < 0:
        xy_angle = -xy_angle

    #### transform (rotation)
    ##### translate first !!!! rotate second !!!!!!!!!!!!!!! important!!!!!

    transform = vtk.vtkTransform()
    transform.Translate(Target)

    transform.RotateWXYZ(90, 0, 1, 0)
    transform.RotateWXYZ(-xy_angle, 1, 0, 0)
    transform.RotateWXYZ(-z_angle, 0, 1, 0)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(spherePoly)
    transformFilter.Update()

    Cutpoly = transformFilter.GetOutput()


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(transformFilter.GetOutput())

    # for debugging dummy file
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(spherePoly)



    actor = vtk.vtkActor()
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor([0,0,1])
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor([0,0,0])
    actor.SetMapper(mapper)

    return Cutpoly, actor, dis_average, dis_min


def make_evencirle(num_pts=1000, ROC=71, width=65, focal_length=55.22, range_vector=[0, 0, 0], Target=[0, 0, 0],
                   opacity=0.7, color=[1, 0, 0]):
    ##################### make transducer function with evely distributed spots

    X = Target[0]
    Y = Target[1]
    Z = Target[2]

    h_wid = width / 2
    p_height = ROC ** 2 - h_wid ** 2
    height_from_center = np.sqrt(p_height)
    height = ROC - height_from_center  ### transducer's height
    rate = height / (ROC * 2)  ## ratio height/ROC*2

    indices_theta = np.arange(0, num_pts, dtype=float)
    indices_phi = np.linspace(0, num_pts * rate, num=num_pts)  ## define transdcuer's height as ratio

    phi = np.arccos(1 - 2 * indices_phi / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices_theta

    phi_deg = np.rad2deg(phi)




    x, y, z = np.cos(theta) * np.sin(phi) * ROC + X, np.sin(theta) * np.sin(phi) * ROC + Y, np.cos(
        phi) * ROC + Z;


    # for mesh distance calculation
    coordi  = np.zeros((num_pts, 3))
    dis_min = np.zeros((num_pts,1))
    coordi[:, 0] = x
    coordi[:, 1] = y
    coordi[:, 2] = z



    # x,y,z is coordination of evenly distributed shpere
    # I will try to make poly data use this x,y,z*radius


    points = vtk.vtkPoints()

    for i in range(len(x)):
        points.InsertNextPoint(x[i], y[i], z[i])
        dis = np.sqrt(np.sum(np.power((coordi-coordi[i,:]),2), axis =1))
        dis_min[i] = np.min(dis[dis>0])

    dis_average = np.average(dis_min)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    # To create surface of a sphere we need to use Delaunay triangulation
    d3D = vtk.vtkDelaunay3D()
    d3D.SetInputData(poly)  # This generates a 3D mesh

    # We need to extract the surface from the 3D mesh
    dss = vtk.vtkDataSetSurfaceFilter()
    dss.SetInputConnection(d3D.GetOutputPort())
    dss.Update()

    # Now we have our final polydata
    spherePoly = dss.GetOutput()

    return spherePoly, dis_average, dis_min

def make_transducer(spherePoly, ROC=71, width=65, focal_length=55.22, range_vector=[0, 0, 0], Target=[0, 0, 0],
                    opacity=1, color=[1, 0, 0]):


    center_vector = [1, 0, 0]
    unit_vector = range_vector / np.linalg.norm(range_vector)
    xy_unit_vector = l2n((unit_vector[0], unit_vector[1], 0))

    if n2l(xy_unit_vector) == [0, 0, 0]:
        xy_angle = 0.0
        z_angle = 90.0
    else:
        xy_angle = np.rad2deg(np.arccos(
            np.dot(center_vector, xy_unit_vector) / (np.linalg.norm(center_vector) * np.linalg.norm(xy_unit_vector))))
        z_angle = np.rad2deg(np.arccos(
            np.dot(xy_unit_vector, unit_vector) / (np.linalg.norm(xy_unit_vector) * np.linalg.norm(unit_vector))))
    if unit_vector[2] < 0:
        z_angle = -z_angle
    if unit_vector[1] < 0:
        xy_angle = -xy_angle

    #### transform (rotation)

    gap = focal_length - ROC
    GAP = n2l(l2n(Target) + l2n(range_vector) * gap)



    transform = vtk.vtkTransform()
    transform.Translate(GAP)    #### move to the gap(trandcuer center to target) point
    transform.RotateWXYZ(90, 0, 1, 0)
    transform.RotateWXYZ(-xy_angle, 1, 0, 0)
    transform.RotateWXYZ(-z_angle, 0, 1, 0)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(spherePoly)
    transformFilter.Update()

    Transducer = transformFilter.GetOutput()


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(transformFilter.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOff()
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)


    return Transducer, actor, xy_angle, z_angle