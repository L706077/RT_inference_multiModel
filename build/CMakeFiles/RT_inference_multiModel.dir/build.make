# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/tensorrt2.1/RT_inference_multiModel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/tensorrt2.1/RT_inference_multiModel/build

# Include any dependencies generated for this target.
include CMakeFiles/RT_inference_multiModel.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/RT_inference_multiModel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RT_inference_multiModel.dir/flags.make

CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o: CMakeFiles/RT_inference_multiModel.dir/flags.make
CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o: ../RT_inference_multiModel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/tensorrt2.1/RT_inference_multiModel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o -c /home/ubuntu/tensorrt2.1/RT_inference_multiModel/RT_inference_multiModel.cpp

CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/tensorrt2.1/RT_inference_multiModel/RT_inference_multiModel.cpp > CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.i

CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/tensorrt2.1/RT_inference_multiModel/RT_inference_multiModel.cpp -o CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.s

CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o.requires:

.PHONY : CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o.requires

CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o.provides: CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o.requires
	$(MAKE) -f CMakeFiles/RT_inference_multiModel.dir/build.make CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o.provides.build
.PHONY : CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o.provides

CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o.provides.build: CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o


# Object files for target RT_inference_multiModel
RT_inference_multiModel_OBJECTS = \
"CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o"

# External object files for target RT_inference_multiModel
RT_inference_multiModel_EXTERNAL_OBJECTS =

RT_inference_multiModel: CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o
RT_inference_multiModel: CMakeFiles/RT_inference_multiModel.dir/build.make
RT_inference_multiModel: /usr/local/lib/libopencv_core.a
RT_inference_multiModel: /usr/local/lib/libopencv_flann.a
RT_inference_multiModel: /usr/local/lib/libopencv_imgproc.a
RT_inference_multiModel: /usr/local/lib/libopencv_highgui.a
RT_inference_multiModel: /usr/local/lib/libopencv_features2d.a
RT_inference_multiModel: /usr/local/lib/libopencv_calib3d.a
RT_inference_multiModel: /usr/local/lib/libopencv_ml.a
RT_inference_multiModel: /usr/local/lib/libopencv_video.a
RT_inference_multiModel: /usr/local/lib/libopencv_legacy.a
RT_inference_multiModel: /usr/local/lib/libopencv_objdetect.a
RT_inference_multiModel: /usr/local/lib/libopencv_photo.a
RT_inference_multiModel: /usr/local/lib/libopencv_gpu.a
RT_inference_multiModel: /usr/local/lib/libopencv_videostab.a
RT_inference_multiModel: /usr/local/lib/libopencv_ts.a
RT_inference_multiModel: /usr/local/lib/libopencv_ocl.a
RT_inference_multiModel: /usr/local/lib/libopencv_superres.a
RT_inference_multiModel: /usr/local/lib/libopencv_nonfree.a
RT_inference_multiModel: /usr/local/lib/libopencv_stitching.a
RT_inference_multiModel: /usr/local/lib/libopencv_contrib.a
RT_inference_multiModel: /usr/local/lib/libopencv_nonfree.a
RT_inference_multiModel: /usr/local/lib/libopencv_gpu.a
RT_inference_multiModel: /usr/local/lib/libopencv_legacy.a
RT_inference_multiModel: /usr/local/lib/libopencv_photo.a
RT_inference_multiModel: /usr/local/lib/libopencv_ocl.a
RT_inference_multiModel: /usr/local/lib/libopencv_calib3d.a
RT_inference_multiModel: /usr/local/lib/libopencv_features2d.a
RT_inference_multiModel: /usr/local/lib/libopencv_flann.a
RT_inference_multiModel: /usr/local/lib/libopencv_ml.a
RT_inference_multiModel: /usr/local/lib/libopencv_video.a
RT_inference_multiModel: /usr/local/lib/libopencv_objdetect.a
RT_inference_multiModel: /usr/local/lib/libopencv_highgui.a
RT_inference_multiModel: /usr/local/lib/libopencv_imgproc.a
RT_inference_multiModel: /usr/local/lib/libopencv_core.a
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libjpeg.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libpng.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libtiff.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libjasper.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libjpeg.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libpng.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libtiff.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libjasper.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libz.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libImath.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libIlmImf.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libIex.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libHalf.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libIlmThread.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libQtOpenGL.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libQtGui.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libQtTest.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libQtCore.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libbz2.so
RT_inference_multiModel: /usr/local/cuda-8.0/lib64/libcudart.so
RT_inference_multiModel: /usr/local/cuda-8.0/lib64/libnppc.so
RT_inference_multiModel: /usr/local/cuda-8.0/lib64/libnppi.so
RT_inference_multiModel: /usr/local/cuda-8.0/lib64/libnpps.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libGLU.so
RT_inference_multiModel: /usr/lib/x86_64-linux-gnu/libGL.so
RT_inference_multiModel: /usr/local/cuda-8.0/lib64/libcufft.so
RT_inference_multiModel: CMakeFiles/RT_inference_multiModel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/tensorrt2.1/RT_inference_multiModel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable RT_inference_multiModel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RT_inference_multiModel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RT_inference_multiModel.dir/build: RT_inference_multiModel

.PHONY : CMakeFiles/RT_inference_multiModel.dir/build

CMakeFiles/RT_inference_multiModel.dir/requires: CMakeFiles/RT_inference_multiModel.dir/RT_inference_multiModel.cpp.o.requires

.PHONY : CMakeFiles/RT_inference_multiModel.dir/requires

CMakeFiles/RT_inference_multiModel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RT_inference_multiModel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RT_inference_multiModel.dir/clean

CMakeFiles/RT_inference_multiModel.dir/depend:
	cd /home/ubuntu/tensorrt2.1/RT_inference_multiModel/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/tensorrt2.1/RT_inference_multiModel /home/ubuntu/tensorrt2.1/RT_inference_multiModel /home/ubuntu/tensorrt2.1/RT_inference_multiModel/build /home/ubuntu/tensorrt2.1/RT_inference_multiModel/build /home/ubuntu/tensorrt2.1/RT_inference_multiModel/build/CMakeFiles/RT_inference_multiModel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RT_inference_multiModel.dir/depend

