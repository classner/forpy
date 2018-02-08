ExternalProject_Add (glog
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/glog"
  # Just providing a SOURCE_DIR doesn't seem to work for all builders (e.g.,
  # ninja) and I want to include all dependencies.
  DOWNLOAD_COMMAND echo ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog-build && mkdir -p ${CMAKE_CURRENT_BINARY_DIR} && cp -r ${CMAKE_CURRENT_SOURCE_DIR}/glog-build ${CMAKE_CURRENT_BINARY_DIR}/glog/src
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DWITH_GFLAGS=Off
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/glog/lib/libglog.a"
)
ExternalProject_Get_Property(glog install_dir)
set (GLOG_BINARY  "${CMAKE_CURRENT_BINARY_DIR}/glog/lib/libglog.a" PARENT_SCOPE)
set (GLOG_INCLUDE  "${CMAKE_CURRENT_BINARY_DIR}/glog/include" PARENT_SCOPE)
