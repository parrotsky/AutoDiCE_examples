cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_FLAGS "-fopenmp -fexceptions")
set(CMAKE_C_FLAGS "-fopenmp -fexceptions")
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_PREFIX_PATH /usr/local/lib/)
set(CMAKE_PREFIX_PATH /usr/lib/x86_64-linux-gnu)
# 指定project的名称
project(c)

INCLUDE_DIRECTORIES(
    ./include
)
# ADD_SUBDIRECTORY(src)

option(WITH_ZMQ "zeromq library support" OFF)
option(WITH_MPI "mpi library support" ON)
option(WITH_NCNN "ncnn library support" ON)
option(WITH_TENSORRT "ncnn library support" OFF)
# add the 'tools' subdirectory to the build
add_subdirectory(tools)

if(WITH_ZMQ)
    find_package(ZeroMQ REQUIRED)
endif()

if(WITH_MPI)
    # MPI
    find_package(MPI REQUIRED)
    SET (CMAKE_C_COMPILER  "mpicxx")
    SET (CMAKE_CXX_COMPILER  "mpic++")
else()
	message(WARNING "The compiler does not support MPI. MPI will be OFF.")
endif()

if(WITH_NCNN)
# add ncnn support for compilation.
#    find_package(Protobuf)
    #    set(ncnn_DIR "/home/sky/cloud/x86_ncnn/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
    set (CMAKE_PREFIX_PATH "./ncnn/")
    set(ncnn_DIR "./ncnn/lib/cmake/ncnn/" CACHE PATH "Directory that contains ncnnConfig.cmake")
    find_package(ncnn REQUIRED)
else()
    message("build without ncnn")
endif()


macro(ncnn_add_example name)
    #set(CMAKE_CXX_FLAGS "-Wall  -Ofast -fopenmp -fexceptions")
    #set(CMAKE_C_FLAGS " -Wall -Ofast -fopenmp  -fexceptions")
    #file(GLOB FUNCSRC "./models/*.cpp")
    add_executable(${name} ./models/${name}.cpp)
    #file(GLOB FUNCSRC "./models/*.cpp")
    #add_executable(${name} ./models/${name}.cpp ${FUNCSRC})
    if(WITH_TENSORRT)
        option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
        # include and link dirs of cuda and tensorrt, you need adapt them if yours are different
        # cuda
        include_directories(/usr/local/cuda/include)
        link_directories(/usr/local/cuda/lib64)
        # tensorrt
	#include_directories(/home/sky/TensorRT-8.5.1.7/include)
	#include_directories(/usr/src/tensorrt/samples/common/)
        #link_directories(/home/sky/TensorRT-8.5.1.7/lib)
        # on jetson
        include_directories(/usr/include/x86_64-linux-gnu/)
        include_directories(/usr/src/tensorrt/samples/common/)
        link_directories(/usr/lib/x86_64-linux-gnu/)

        add_definitions(-O2 -pthread)
    endif()

    if(WITH_MPI)
        target_link_libraries(${name} PUBLIC MPI::MPI_CXX MPI::MPI_C)
    endif()
    if(WITH_TENSORRT)
        target_link_libraries(${name} PUBLIC nvinfer nvparsers nvonnxparser cudart)
    endif()

    if(WITH_ZMQ)
	target_link_libraries(${name} PUBLIC zmq zmqpp)
    endif()

    #if(WITH_MPI AND WITH_NCNN AND WITH_TENSORRT)
    #    target_link_libraries(${name} PUBLIC MPI::MPI_CXX MPI::MPI_C ncnn nvinfer nvparsers nvonnxparser cudart)
    #elseif(WITH_MPI AND WITH_NCNN)
    #    target_link_libraries(${name} PUBLIC ncnn MPI::MPI_CXX MPI::MPI_C)
    #elseif(WITH_MPI AND WITH_TENSORRT)
    #    target_link_libraries(${name} PUBLIC  nvinfer nvparsers nvonnxparser cudart MPI::MPI_CXX MPI::MPI_C)
    #elseif(WITH_ZMQ AND WITH_TENSORRT)
    #    target_link_libraries(${name} PUBLIC  nvinfer nvparsers nvonnxparser cudart MPI::MPI_CXX MPI::MPI_C)
    #endif()


    
    # opencv
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIR})
    target_link_libraries(${name} PRIVATE ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES}) 
    if (WITH_NCNN)
        target_link_libraries(${name} PRIVATE ncnn) 
    endif()


    # add test to a virtual project group
    set_property(TARGET ${name} PROPERTY FOLDER "examples")
endmacro()

# Use the correct version of CUDA

# We require CUDA, OpenCV, and TensorRT
#find_package(TensorRT REQUIRED)
find_package(OpenCV QUIET COMPONENTS opencv_world)
# for opencv 2.4 on ubuntu 16.04, there is no opencv_world but OpenCV_FOUND will be TRUE
if("${OpenCV_LIBS}" STREQUAL "")
    set(OpenCV_FOUND FALSE)
endif()
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
endif()


if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found, examples won't be built")
    #elseif(NOT NCNN_PIXEL)
    #message(WARNING "NCNN_PIXEL not enabled, examples won't be built")
else()
    message(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

    if(${OpenCV_VERSION_MAJOR} GREATER 3)
        set(CMAKE_CXX_STANDARD 11)
    endif()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)
    #   include_directories(${CUDAToolkit_INCLUDE_DIRS})

    #ncnn_add_example(hello)
    #ncnn_add_tensorrt(trthello)
    ncnn_add_example(multinode)
    #ncnn_add_example(multinode_bench)
endif()

#### The below involves with onnx operations.
#### You can comment below cmake commands if you just want to compile the code.
# first, get a list of all .onnx files in the current directory
file(GLOB ONNX_FILES "./models/*.onnx")

# then, for each .onnx file, create a custom command to run onnx2ncnn on it
foreach(ONNX_FILE ${ONNX_FILES})
    get_filename_component(FILENAME ${ONNX_FILE} NAME_WE) # Extract the file name without extension
    add_custom_target(
        run_onnx2ncnn_${FILENAME} ALL # create a unique target for each file
        COMMAND ./tools/onnx2ncnn ${ONNX_FILE} # run the command
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running onnx2ncnn on ${ONNX_FILE}..."
    )
endforeach()

