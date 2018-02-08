configure_file(${CMAKE_CURRENT_SOURCE_DIR}/forpy/version.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/forpy/version.h @ONLY)

# Include a target for headers as workaround to have them shown up in an IDE.
FILE(GLOB_RECURSE LibFiles "*.h" PATTERN "*.in" EXCLUDE)
add_custom_target(headers SOURCES ${LibFiles})
install (DIRECTORY forpy DESTINATION include PATTERN "*.in" EXCLUDE)
