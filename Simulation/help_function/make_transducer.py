import numpy as np
import SimpleITK as sitk
import os

def makeSphere(shape, radius, position):
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    result = (arr <= 1.0)
    result = result.astype(int)

    return result



def makeTransducer_vertical(image_cordi, step_size, ROC, width, tran_pose, target_pose):

    ## ROC depend on Voxel
    radiusVN = int(np.round(ROC / step_size))

    ## direction vector from target to geometrical focus
    dirVec = (target_pose-tran_pose)/np.linalg.norm(target_pose-tran_pose)
    geometry_center = target_pose+dirVec*np.abs(ROC-np.linalg.norm(target_pose-tran_pose))

    x_arr = image_cordi[0]
    y_arr = image_cordi[1]
    z_arr = image_cordi[2]

    x_idx = np.argmin(np.abs(x_arr-geometry_center[0]))
    y_idx = np.argmin(np.abs(y_arr-geometry_center[1]))
    z_idx = np.argmin(np.abs(z_arr-geometry_center[2]))

    # Make mesh grid IMPORTANT order !!
    my, mx, mz = np.meshgrid(y_arr, x_arr, z_arr)
    shape = mx.shape

    # Make large sphere for transducer
    sphere = makeSphere((shape[0], shape[1], shape[2]), radiusVN, (x_idx, y_idx, z_idx))
    sphere_sub = makeSphere((shape[0], shape[1], shape[2]), radiusVN-1, (x_idx, y_idx, z_idx))
    sphere = sphere - sphere_sub

    # Make plane to cut the sphere and make transducer
    # calculate normal vector of plane
    normal = (tran_pose-target_pose)/np.linalg.norm(tran_pose-target_pose)
    temp_length = np.sqrt(ROC**2 - (width/2)**2)

    # Find one point on the cutting plane
    plane_point = geometry_center + temp_length*normal

    # Plane equation using normal and point
    cont = normal[0]*plane_point[0] + normal[1]*plane_point[1] + normal[2]*plane_point[2]
    plane_cal = mx * normal[0] + my * normal[1] + mz * normal[2] - cont

    plane_cal[plane_cal > 0] = 1
    plane_cal[plane_cal < 0] = 0
    plane_cal = plane_cal.astype(int)

    # Cut sphere using plane
    transducer = np.multiply(sphere, plane_cal)

    sitk_transducer = transducer*1000
    sitk_transducer = sitk_transducer.transpose([2,1,0])
    spacing = np.array((step_size, step_size, step_size))
    origin = np.array((min(x_arr), min(y_arr), min(z_arr)))
    trans_itk = sitk.GetImageFromArray(sitk_transducer, sitk.sitkInt8)
    trans_itk.SetSpacing(spacing)
    trans_itk.SetOrigin(origin)

    return transducer, trans_itk


def makeBowlTransducer(Nx, Ny, Nz, step_size, ROC, width, PML = 10):


    Nx_half = int(Nx/2)
    Ny_half = int(Ny/2)

    cutting_len = np.sqrt(pow(ROC, 2) - pow(width/2, 2))
    cutting_lenVNM = int(np.round(cutting_len/step_size))

    radiusVN = int(np.round(ROC/step_size))
    radiusVNM = radiusVN + PML

    sphere = makeSphere((radiusVNM*4, radiusVNM*4, radiusVNM*2), radiusVN, (radiusVNM*2, radiusVNM*2, radiusVNM))
    sphere_sub  = makeSphere((radiusVNM*4, radiusVNM*4, radiusVNM*2), radiusVN-1, (radiusVNM*2, radiusVNM*2, radiusVNM))
    sphereN = sphere-sphere_sub

    # This function differ to the "makebowl" in k-wave, to match this difference x step margin was given
    function_difference_gap = 0
    radiusVNM2 = radiusVNM * 2
    Transducer = sphereN[radiusVNM2-Nx_half:radiusVNM2+Nx_half, radiusVNM2-Ny_half:radiusVNM2+Ny_half, function_difference_gap:(radiusVNM-cutting_lenVNM)+function_difference_gap]
    sensor_model = np.zeros((Nx,Ny,Nz))
    sensor_model[:,:,PML:(radiusVNM-cutting_lenVNM)+PML] = Transducer

    return sensor_model


def neper2db(alpha, y):

    alphaDB = 20*np.log10(np.exp(1))*alpha*pow((2*np.pi*1e6), y)/100

    return alphaDB

