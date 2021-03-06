# Requirements list
cmake_minimum_required(VERSION 2.8.11)
project(python-cpp CXX)
include(CheckCXXCompilerFlag)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.0")
   message(FATAL_ERROR "ERROR: Insufficient compiler version, need at lest 4.8 to continue")
endif()

check_cxx_compiler_flag(-std=c++11 COMPILER_HAS_CXX11)
if(!COMPILER_HAS_CXX11)
	message(FATAL_ERROR "ERROR: Compiler is lacking support for c++11")
endif()

set(EXTERNAL_INSTALL_DIR external/install)

# set(ENV{PYTHONPATH} "$ENV{PYTHONPATH} ${PROJECT_SOURCE_DIR}/python")
# set(DEFINITIONS "-DHOME=${PROJECT_SOURCE_DIR} -DPYLIBS=${PROJECT_SOURCE_DIR}/external/install")
add_definitions(-DPROJECT_HOME="${PROJECT_SOURCE_DIR}"
                -DPYLIBS="${PROJECT_SOURCE_DIR}/external/install"
                -DPYTHONPATH="${PROJECT_SOURCE_DIR}/python/")

include_directories(include
   # $ENV{HOME}/Downloads/Python-3.5.2/Include
   # $ENV{HOME}/opt/include
   # $ENV{HOME}/Downloads/Python-2.7.12/Include
   # $ENV{HOME}/Downloads/Python-2.7.12
   ${EXTERNAL_INSTALL_DIR}/include/python2.7
   ${EXTERNAL_INSTALL_DIR}/include/
   ${EXTERNAL_INSTALL_DIR}/lib/python2.7/site-packages
   # $ENV{HOME}/work/anaconda2/include/python2.7
   # $ENV{HOME}/work/anaconda2/include
   # $ENV{HOME}/work/anaconda2/lib/python2.7)
   )

link_directories(
   # /usr/lib
   # /usr/lib64 
   # /usr/local/lib
   # $ENV{HOME}/opt/lib
   # $ENV{HOME}/opt/lib/python2.7/config
   ${EXTERNAL_INSTALL_DIR}/lib/
   )
   # $ENV{HOME}/work/anaconda2/lib)


set(CMAKE_BUILD_TYPE Release)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
      set(CMAKE_CXX_FLAGS         "-std=c++11 -fopenmp -Wall -pedantic -Wunused ${DEFINITIONS}")
      set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3")# -fsanitize=address")
      set(CMAKE_CXX_FLAGS_RELEASE "-march=native")
      
      check_cxx_compiler_flag(-Ofast COMPILER_HAS_OFAST)
      if(COMPILER_HAS_OFAST)
         set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
      else()
         set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
      endif()
      
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}") #-fopt-info-vec=report.txt")
      
      set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto")
endif()


#set(LIBRARIES blond gsl gslcblas fftw3 m)
set(LIBRARIES libpython python2.7 util dl)#python2.7)
# set(LIBRARIES "${LIBRARIES} -libpython3.5m.a")

file(GLOB SOURCES src/*.cpp)

set(SOURCE_FILES ${SOURCES})

# SHARED for dynamic LINKING
# STATIC for static LINKING
add_library(libpython STATIC ${SOURCE_FILES})

file(GLOB EXECUTABLE_SOURCES "tests/*.cpp")
set(EXECUTABLE_FILES ${EXECUTABLE_SOURCES})

foreach(src ${EXECUTABLE_FILES})
   file(RELATIVE_PATH testfile ${CMAKE_CURRENT_SOURCE_DIR} ${src})
   string( REPLACE ".cpp" "" testname ${testfile})
   string( REPLACE "tests/" "" testname ${testname})
   add_executable( ${testname} ${src})
   target_link_libraries( ${testname} ${LIBRARIES})
endforeach()
