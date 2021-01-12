cmake_minimum_required(VERSION 3.5)

project(Simulation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/Simulation")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Taeyoung Park (KIST.)")
set(EXTENSION_DESCRIPTION "This is an example of Acoustic simulation")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/Simulation.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/Simulation/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Simulation)
add_subdirectory(FUS)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
