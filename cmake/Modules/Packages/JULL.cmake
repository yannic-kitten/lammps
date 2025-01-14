find_package(MPI REQUIRED)
set(DOWNLOAD_ALL_DEFAULT ON)
find_package(ALL QUIET)
if(ALL_FOUND)
  set(DOWNLOAD_ALL_DEFAULT OFF)
endif()
option(DOWNLOAD_ALL "Download ALL library instead of using an already installed one" ${DOWNLOAD_ALL_DEFAULT})

if(DOWNLOAD_ALL)
  message(STATUS "ALL download requested - we will build our own")
  set(ALL_URL "https://gitlab.jsc.fz-juelich.de/SLMS/loadbalancing/-/archive/v0.9.3/loadbalancing-v0.9.3.tar.gz" CACHE STRING "URL for ALL tarball")
  set(ALL_MD5 "9fc008711a7dfaf35e957411f8ed1504" CACHE STRING "MD5 checksum of ALL tarball")
  mark_as_advanced(ALL_URL)
  mark_as_advanced(ALL_MD5)
  GetFallbackURL(ALL_URL ALL_FALLBACK)

  include(ExternalProject)
  ExternalProject_Add(ALL_build
    URL     ${ALL_URL} ${ALL_FALLBACK}
    URL_MD5 ${ALL_MD5}
    CMAKE_ARGS 
               -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               -DCMAKE_INSTALL_LIBDIR=lib
  )
  ExternalProject_get_property(ALL_build INSTALL_DIR)
  file(MAKE_DIRECTORY ${INSTALL_DIR}/include)
  add_library(LAMMPS::ALL INTERFACE IMPORTED)
  add_dependencies(LAMMPS::ALL ALL_build)
  set_target_properties(LAMMPS::ALL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${INSTALL_DIR}/include")
  target_link_libraries(lammps PRIVATE LAMMPS::ALL)
else()
    find_package(ALL REQUIRED)
    target_link_libraries(lammps PRIVATE ALL::ALL)
endif()
