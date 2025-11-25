cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
endif(CCACHE_FOUND)

project(wenet VERSION 0.1)

# Ensure modern C++ and PIC for all the static libs you build
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Make sure CMake knows which Python to target so pybind11 uses the correct suffix (cpython-39, etc.)
# If the user passed a custom Python executable, forward it as a proper hint.
if(DEFINED PYTHON3_EXECUTABLE)
    set(Python3_EXECUTABLE "${PYTHON3_EXECUTABLE}")
    # Optionally also set a root dir (two levels up from the exe)
    get_filename_component(Python3_ROOT_DIR "${Python3_EXECUTABLE}" DIRECTORY)
    get_filename_component(Python3_ROOT_DIR "${Python3_ROOT_DIR}" DIRECTORY)
endif()

# Ask CMake for both the interpreter and dev artifacts in ONE call.
# Use Development for maximum compatibility; it implies Development.Module. :contentReference[oaicite:3]{index=3}
find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Using Python from: ${Python3_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python3_LIBRARIES}")



add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

option(GRPC "whether to build gRPC" OFF)

include_directories(
${CMAKE_SOURCE_DIR}
${CMAKE_SOURCE_DIR}/kaldi  # for not changing c++ header names in kaldi source files
)
set(CMAKE_VERBOSE_MAKEFILE on)

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

if(NOT MSVC)
  # Keep the same with openfst, -fPIC or -fpic
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
endif()

# third_party: gflags
FetchContent_Declare(gflags
  URL      https://github.com/gflags/gflags/archive/v2.2.1.zip
  URL_HASH SHA256=4e44b69e709c826734dbbbd5208f61888a2faf63f239d73d8ba0011b2dccc97a
)
FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)

# Force glog to use the FetchContent gflags instead of searching system paths
# Set these variables before glog's CMake runs its tests
set(gflags_DIR ${gflags_BINARY_DIR})
set(gflags_ROOT ${gflags_SOURCE_DIR})
# CMAKE_REQUIRED_INCLUDES is used by try_compile tests (like DetermineGflagsNamespace)
set(CMAKE_REQUIRED_INCLUDES ${gflags_BINARY_DIR}/include ${CMAKE_REQUIRED_INCLUDES})
# Also set include path for general compilation
set(CMAKE_INCLUDE_PATH ${gflags_BINARY_DIR}/include ${CMAKE_INCLUDE_PATH})
# Set library path too
set(CMAKE_LIBRARY_PATH ${gflags_BINARY_DIR} ${CMAKE_LIBRARY_PATH})

# third_party: glog
FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)
# Disable gflags in glog - not needed since we use glog through openfst
set(WITH_GFLAGS OFF CACHE BOOL "Use gflags" FORCE)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

# third_party: gtest
FetchContent_Declare(googletest
  URL      https://github.com/google/googletest/archive/release-1.10.0.zip
  URL_HASH SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91
)
FetchContent_MakeAvailable(googletest)

# third_party: boost
FetchContent_Declare(boost
  URL      https://archives.boost.io/release/1.75.0/source/boost_1_75_0.tar.gz
  URL_HASH SHA256=aeb26f80e80945e82ee93e5939baebdca47b9dee80a07d3144be1e1a6a66dd6a
)
FetchContent_MakeAvailable(boost)
include_directories(${boost_SOURCE_DIR})

# third_party: cnpy
FetchContent_Declare(cnpy
  URL      https://github.com/rogersce/cnpy/archive/refs/heads/master.zip
  URL_HASH SHA256=dd58304295ce30e2299fafe8a72cb1acac508139df740c8976bcaab5c17591dc
)
FetchContent_MakeAvailable(cnpy)
include_directories(${cnpy_SOURCE_DIR})

# Redis
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/hiredis)
set(REDIS_PLUS_PLUS_CXX_STANDARD 14)
set(HIREDIS_HEADER ${CMAKE_SOURCE_DIR}/third_party)
set(HIREDIS_LIB hiredis)
set(TEST_HIREDIS_LIB hiredis)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/redis-plus-plus)

# Try to find PyTorch via Python first
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_IMPORT_RESULT
    ERROR_QUIET
)

# If torch is installed via pip, it won't have CMake config files
# We need to download libtorch separately or use the system installation
if(TORCH_IMPORT_RESULT EQUAL 0 AND EXISTS "${TORCH_PYTHON_PATH}")
    set(TORCH_DIR "${TORCH_PYTHON_PATH}/share/cmake/Torch")
    if(NOT EXISTS "${TORCH_DIR}/TorchConfig.cmake")
        # Try parent directory
        get_filename_component(TORCH_DIR "${TORCH_PYTHON_PATH}/.." ABSOLUTE)
        set(TORCH_DIR "${TORCH_DIR}/share/cmake/Torch")
    endif()
    
    # If still not found, we need to download libtorch
    if(NOT EXISTS "${TORCH_DIR}/TorchConfig.cmake")
        message(STATUS "PyTorch found via Python but no CMake config. Trying to find libtorch in Python installation...")
        
        # For Apple Silicon, try to find libtorch in the Python installation
        if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            # Try to find libtorch in site-packages
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
                OUTPUT_VARIABLE TORCH_LIB_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE TORCH_LIB_RESULT
            )
            if(TORCH_LIB_RESULT EQUAL 0 AND EXISTS "${TORCH_LIB_DIR}")
                message(STATUS "Found PyTorch lib directory: ${TORCH_LIB_DIR}")
                # For pip-installed PyTorch on Apple Silicon, we need to manually set up paths
                # The libraries are in torch/lib, but we need to point to a valid TorchConfig location
                # Since there's no CMake config, we'll need to set up the paths manually later
                message(WARNING "PyTorch found but no CMake config. Will attempt to use libraries directly.")
                # Set TORCH_DIR to the lib directory for now
                set(TORCH_DIR "${TORCH_LIB_DIR}")
            else()
                message(FATAL_ERROR "PyTorch found but libtorch not available. For Apple Silicon, please ensure PyTorch is properly installed with: pip install torch")
            endif()
        else()
            # For non-Apple Silicon, download libtorch as before
            message(STATUS "Downloading libtorch...")
            set(PYTORCH_VERSION "1.13.1")
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
            set(URL_HASH "SHA256=d289185802baa1977701b72fc17c8aa6715481703352ce02d6f710cf007dc4e9")
            
            FetchContent_Declare(libtorch
                URL      ${LIBTORCH_URL}
                URL_HASH ${URL_HASH}
            )
            set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
            FetchContent_MakeAvailable(libtorch)
            set(TORCH_DIR ${libtorch_SOURCE_DIR})
        endif()
    endif()
endif()

# Try to find Torch via CMake config
if(EXISTS "${TORCH_DIR}/TorchConfig.cmake")
    find_package(Torch REQUIRED PATHS ${TORCH_DIR} NO_DEFAULT_PATH)
else()
    # For Apple Silicon with pip-installed PyTorch, manually set up Torch
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        message(STATUS "Torch CMake config not found. Setting up Torch manually for Apple Silicon...")
        
        # Get PyTorch installation path
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
            OUTPUT_VARIABLE TORCH_PYTHON_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        # Set up Torch variables manually
        set(TORCH_FOUND TRUE)
        set(TORCH_VERSION "1.13.1")  # Adjust if needed
        
        # Include directories
        set(TORCH_INCLUDE_DIRS 
            "${TORCH_PYTHON_PATH}/include"
            "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include"
        )
        
        # Library directory
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
            OUTPUT_VARIABLE TORCH_LIB_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        # Libraries - find all .so/.dylib files in the lib directory
        file(GLOB TORCH_LIBS "${TORCH_LIB_DIR}/*.dylib" "${TORCH_LIB_DIR}/*.so")
        set(TORCH_LIBRARIES ${TORCH_LIBS})
        
        # CXX flags
        set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
        
        message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
        message(STATUS "Torch lib dir: ${TORCH_LIB_DIR}")
        message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
    else()
        message(FATAL_ERROR "Could not find Torch CMake config and not on Apple Silicon. Please install libtorch.")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")

# Filter out Linux-specific linker flags on macOS
if(APPLE)
    # Extract only the actual library paths from TORCH_LIBRARIES
    # PyTorch's CMake adds -Wl,--no-as-needed and -Wl,--as-needed flags
    # that macOS linker doesn't support
    set(TORCH_LIBRARIES_FILTERED "")
    foreach(item ${TORCH_LIBRARIES})
        # Skip linker flags
        if(item MATCHES "^-Wl,--(no-)?as-needed")
            continue()
        endif()
        # Extract library path from quoted strings like "-Wl,--no-as-needed,\"/path\""
        if(item MATCHES "\"([^\"]+)\"")
            # Extract the quoted path
            string(REGEX REPLACE ".*\"([^\"]+)\".*" "\\1" lib_path "${item}")
            list(APPEND TORCH_LIBRARIES_FILTERED "${lib_path}")
        elseif(item MATCHES "^/.*\\.(so|a)$" OR item MATCHES "^-l")
            # It's already a clean library path or -l flag
            list(APPEND TORCH_LIBRARIES_FILTERED "${item}")
        endif()
    endforeach()
    set(TORCH_LIBRARIES ${TORCH_LIBRARIES_FILTERED})
endif()

# utils
add_library(utils STATIC
  utils/string.cc
  utils/utils.cc
)

# The original openfst uses GNU Build System to run configure and build.
# So, we use "OpenFST port for Windows" to build openfst with cmake in Windows.
# Openfst is compiled with glog/gflags to avoid log and flag conflicts with log and flags in wenet/libtorch.
# To build openfst with gflags and glog, we comment out some vars of {flags, log}.h and flags.cc.
if(NOT MSVC)
  set(openfst_SOURCE_DIR ${fc_base}/openfst-src)
  set(openfst_BINARY_DIR ${fc_base}/openfst-build)
  set(openfst_PREFIX_DIR ${fc_base}/openfst-subbuild/openfst-populate-prefix)
  ExternalProject_Add(openfst
    URL               https://github.com/mjansche/openfst/archive/1.6.5.zip
    URL_HASH          SHA256=b720357a464f42e181d7e33f60867b54044007f50baedc8f4458a3926f4a5a78
    SOURCE_DIR        ${openfst_SOURCE_DIR}
    BINARY_DIR        ${openfst_BINARY_DIR}
    CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure --prefix=${openfst_PREFIX_DIR}
                        "CPPFLAGS=-I${gflags_BINARY_DIR}/include -I${glog_SOURCE_DIR}/src -I${glog_BINARY_DIR} -D_GLIBCXX_USE_CXX11_ABI=0"
                        "LDFLAGS=-L${gflags_BINARY_DIR} -L${glog_BINARY_DIR}"
                        "LIBS=-lgflags_nothreads -lglog -lpthread"
    COMMAND           ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
    BUILD_COMMAND     make -j 4
  )
  add_dependencies(openfst gflags glog)
  add_dependencies(utils openfst)
  link_directories(${openfst_PREFIX_DIR}/lib)
  target_link_libraries(utils PUBLIC fst dl)
else()
  execute_process(COMMAND bootstrap.bat WORKING_DIRECTORY ${boost_SOURCE_DIR})
  execute_process(COMMAND b2.exe WORKING_DIRECTORY ${boost_SOURCE_DIR})
  link_directories(${boost_SOURCE_DIR}/stage/lib)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  file(COPY ${TORCH_DLLS} DESTINATION ${CMAKE_BINARY_DIR})
  FetchContent_Declare(openfst
    URL      https://github.com.cnpmjs.org/kkm000/openfst/archive/refs/tags/win/1.6.5.1.tar.gz
    URL_HASH SHA256=02c49b559c3976a536876063369efc0e41ab374be1035918036474343877046e
  )
  FetchContent_MakeAvailable(openfst)
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR})
  add_dependencies(fst gflags glog)
  target_link_libraries(fst PUBLIC gflags_nothreads_static glog)
  target_link_libraries(utils PUBLIC fst)
endif()
include_directories(${openfst_SOURCE_DIR}/src/include)

# kaldi: wfst based decoder
add_subdirectory(kaldi)

# decoder
add_library(decoder STATIC
  decoder/ctc_prefix_beam_search.cc
  decoder/ctc_wfst_beam_search.cc
  decoder/brain_speech_decoder.cc
)
target_link_libraries(decoder PUBLIC ${TORCH_LIBRARIES} kaldi-decoder utils kaldi-fstext)
# Add Torch include directories so decoder can find torch/script.h
target_include_directories(decoder PUBLIC ${TORCH_INCLUDE_DIRS})

find_package(ZLIB REQUIRED)

# binary
add_executable(brain_speech_decoder_main bin/brain_speech_decoder_main.cc)
target_link_libraries(brain_speech_decoder_main PUBLIC decoder cnpy-static ${ZLIB_LIBRARIES})
#add_executable(redis_server_main bin/redis_server_main.cc)
#target_link_libraries(redis_server_main PUBLIC redis++_static hiredis_static decoder)

add_subdirectory(pybind11)
pybind11_add_module(lm_decoder python/lm_decoder.cc)
target_link_libraries(lm_decoder PUBLIC decoder)
