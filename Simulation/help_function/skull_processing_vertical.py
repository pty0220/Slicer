import SimpleITK as sitk
import numpy as np
import os
l2n = lambda l: np.array(l)
n2l = lambda n: list(n)



def make_ref_image(origin_image, grid_spacing, tran_pos, target_pos, boundary):

    origin = origin_image.GetOrigin()

    dimension = origin_image.GetSize()
    origin_spacing = origin_image.GetSpacing()
    print(origin)
    print(dimension)
    x_end = origin[0] + origin_spacing[0] * (dimension[0] - 1)
    y_end = origin[1] - origin_spacing[1] * (dimension[1] - 1)
    z_end = origin[2] + origin_spacing[2] * (dimension[2] - 1)

    x_arr = np.linspace(origin[0], x_end, dimension[0])
    y_arr = np.linspace(origin[1], y_end, dimension[1])
    z_arr = np.linspace(origin[2], z_end, dimension[2])

    dist = np.linalg.norm(target_pos - tran_pos)
    grid_min_point = target_pos-(dist*1.2)
    grid_max_point = target_pos+(dist*1.2)

    if boundary>0:
        if grid_min_point[0] > np.min(x_arr):
            grid_min_point[0] = np.min(x_arr)

        if grid_min_point[1] > np.min(y_arr):
            grid_min_point[1] = np.min(y_arr)

        grid_min_point[2] = np.min(z_arr)+60

        if grid_max_point[0] < np.max(x_arr):
            grid_max_point[0] = np.max(x_arr)

        if grid_max_point[1] < np.max(y_arr):
            grid_max_point[1] = np.max(y_arr)
            print('Set large boundary')

    print(np.max(x_end))
    print(np.max(y_end))
    print(np.max(z_end))


    print("grid min", grid_min_point)
    print("grid max", grid_max_point)


    grid_bound = np.abs(grid_max_point-grid_min_point)
    grid_number = np.round(grid_bound/grid_spacing)

    reference_image = sitk.Image(int(grid_number[0]), int(grid_number[1]), int(grid_number[2]), sitk.sitkFloat32)
    reference_image.SetSpacing(grid_spacing)
    reference_image.SetOrigin(grid_min_point)
    reference_image[:, :, :] = 0
    return reference_image



def resample(image, reference_image):

    rigid_euler = sitk.Euler3DTransform()
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = -1000.0
    imageT = sitk.Resample(image, reference_image, rigid_euler,interpolator, default_value)

    return imageT



def make_skull_medium_vertical(image, grid_spacing, transducer, target, boundary):
    upper_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

    simulation_grid = make_ref_image(image, grid_spacing, transducer, target, boundary)
    # transform_matix = make_transform(transducer, target)
    crop_image_itk = resample(image, simulation_grid)
    crop_image_arr = sitk.GetArrayFromImage(crop_image_itk)
    crop_image_arr = crop_image_arr.transpose([2,1,0])


    origin = crop_image_itk.GetOrigin()
    dimension = crop_image_arr.shape

    x_end = origin[0] + grid_spacing[0] * (dimension[0] - 1)
    y_end = origin[1] + grid_spacing[1] * (dimension[1] - 1)
    z_end = origin[2] + grid_spacing[2] * (dimension[2] - 1)

    x_arr = np.linspace(origin[0], x_end, dimension[0])
    y_arr = np.linspace(origin[1], y_end, dimension[1])
    z_arr = np.linspace(origin[2], z_end, dimension[2])

    image_cordi = [x_arr, y_arr, z_arr]

    return crop_image_arr, crop_image_itk, image_cordi





