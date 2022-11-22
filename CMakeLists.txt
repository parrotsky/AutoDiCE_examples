cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_FLAGS "-fopenmp -fexceptions")
set(CMAKE_C_FLAGS "-fopenmp -fexceptions")

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# 指定project的名称
project(c)

macro(ncnn_add_example name)
	SET (CMAKE_C_COMPILER  "mpicxx")
	SET (CMAKE_CXX_COMPILER  "mpic++")
    set(CMAKE_CXX_FLAGS "-fopenmp -fexceptions")
    set(CMAKE_C_FLAGS "-fopenmp -fexceptions")
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE ncnn ${OpenCV_LIBS})
    target_link_libraries(${name} PUBLIC MPI::MPI_CXX MPI::MPI_C)
    #target_link_libraries(${name} PUBLIC MPI::MPI_C)

    # add test to a virtual project group
    set_property(TARGET ${name} PROPERTY FOLDER "examples")
endmacro()

macro(ncnn_add_tensorrt name)
	set(CMAKE_CXX_STANDARD 17)
	SET (CMAKE_C_COMPILER  "mpicxx")
	SET (CMAKE_CXX_COMPILER  "mpic++")
    set(TensorRT_DIR /usr/lib/aarch64-linux-gnu/)
    # Use the correct version of CUDA
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
    find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

    set(CMAKE_CXX_FLAGS "-Wall  -Ofast -DNDEBUG -fexceptions")
    set(CMAKE_C_FLAGS " -Wall -Ofast -DNDEBUG -fexceptions")
    #target_link_libraries(${name} PUBLIC MPI::MPI_CXX)
    #target_link_libraries(${name} PUBLIC MPI::MPI_C)

    add_library(tensorrt_cpp_api SHARED engine.cpp)
    #include_directories("/usr/include/aarch64-linux-gnu/")
    include_directories("/usr/src/tensorrt/samples/common")
    target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR} /usr/src/tensorrt/samples/common)
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} CUDA::cudart ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

#target_compile_options(${name} PUBLIC -fno-rtti)

set_source_files_properties(${name}.cpp PROPERTIES COMPILE_FLAGS -fexceptions)

    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${name} PUBLIC tensorrt_cpp_api)
    target_link_libraries(${name} PRIVATE ncnn ${OpenCV_LIBS})
    #target_link_libraries(${name} PUBLIC MPI::MPI_CXX MPI::MPI_C)
    #target_link_libraries(${name} PUBLIC MPI::MPI_C)




    # add test to a virtual project group
    set_property(TARGET ${name} PROPERTY FOLDER "examples")
endmacro()

set(CMAKE_PREFIX_PATH /usr/local/lib/)
set(ncnn_DIR "/home/sky/cloud/arm_ncnn/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)



# MPI
find_package(MPI REQUIRED)
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
#if(NOT OpenCV_FOUND)
#    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
#endif()

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
    include_directories(${CUDAToolkit_INCLUDE_DIRS})



     ncnn_add_example(hello)
     # ncnn_add_tensorrt(trthello)
    # ncnn_add_example(alexnet)
    # ncnn_add_example(resnet)
    # ncnn_add_example(squeezenet)
    # ncnn_add_example(squeezenetxt)
    # ncnn_add_example(squeezenet_c_api)
    # ncnn_add_example(fasterrcnn)
    # ncnn_add_example(rfcn)
    # ncnn_add_example(yolov2)
    # ncnn_add_example(yolov3)
    # ncnn_add_example(yolov4)
    # ncnn_add_example(yolov5)
    # ncnn_add_example(mobilenetv2ssdlite)
    # ncnn_add_example(mobilenetssd)
    # ncnn_add_example(squeezenetssd)
    # ncnn_add_example(shufflenetv2)
    # ncnn_add_example(peleenetssd_seg)
    # ncnn_add_example(simplepose)
    # ncnn_add_example(retinaface)
    # ncnn_add_example(yolact)
    # ncnn_add_example(nanodet)
    # ncnn_add_example(bench)
    ncnn_add_example(multinode)
    ncnn_add_example(multinode_bench)
endif()
