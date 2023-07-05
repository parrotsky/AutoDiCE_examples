cmake_minimum_required(VERSION 3.5)
project(onnxtools)
set(CMAKE_CXX_STANDARD 11)

find_package(Protobuf)

if(PROTOBUF_FOUND)
    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)
    add_executable(onnx2ncnn onnx2ncnn.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
    target_include_directories(onnx2ncnn
        PRIVATE
            ${PROTOBUF_INCLUDE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(onnx2ncnn PRIVATE ${PROTOBUF_LIBRARIES})

    #add_executable(ncnn2mem ncnn2mem.cpp)
    #target_link_libraries(ncnn2mem PRIVATE ncnn)
    ##target_link_libraries(ncnn2mem PRIVATE ${Vulkan_LIBRARY})
    # 
    #add_executable(ncnnoptimize ncnnoptimize.cpp)
    #target_link_libraries(ncnnoptimize PRIVATE ncnn)
    ##if(NCNN_VULKAN)
    ##    target_link_libraries(ncnnoptimize PRIVATE ${Vulkan_LIBRARY})
    ##endif()
    # 
    #add_executable(ncnnmerge ncnnmerge.cpp)

    # add all onnx2ncnn tool to a virtual project group
    set_property(TARGET onnx2ncnn PROPERTY FOLDER "tools/converter")
    #    set_property(TARGET ncnn2mem PROPERTY FOLDER "tools/converter")
    #    set_property(TARGET ncnnoptimize PROPERTY FOLDER "tools/converter")
    #    set_property(TARGET ncnnmerge PROPERTY FOLDER "tools/converter")
else()
    message(WARNING "Protobuf not found, onnx model convert tool won't be built")
endif()
