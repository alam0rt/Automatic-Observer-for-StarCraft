cmake_minimum_required(VERSION 3.24)
project(sc_extract C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(STARDUST_DIR "$ENV{STARDUST_DIR}" CACHE PATH "Stardust source tree; its 3rdparty/ provides OpenBW, BWAPI, BWEM and FAP")
if (NOT EXISTS "${STARDUST_DIR}/3rdparty/openbw/openbw/bwgame.h")
    message(FATAL_ERROR "STARDUST_DIR must point at a Stardust checkout (got '${STARDUST_DIR}')")
endif ()

set(SD3P ${STARDUST_DIR}/3rdparty)

# Read by OpenBWData's find_path
set(OPENBW_DIR ${SD3P}/openbw/openbw)

add_subdirectory(${SD3P}/openbw/bwapi bwapi)

# BWAPILIB's own include path is ${CMAKE_SOURCE_DIR}/3rdparty/..., which only works
# when Stardust is the top-level project
target_include_directories(BWAPILIB SYSTEM PUBLIC ${SD3P}/openbw/bwapi/bwapi/include)

add_subdirectory(${SD3P}/BWEM BWEM)
add_subdirectory(${SD3P}/FAP FAP)
add_subdirectory(${SD3P}/nlohmann nlohmann)

add_executable(sc-extract
        src/main.cpp
        src/Fights.cpp
        src/UnitValue.cpp
        )

target_include_directories(sc-extract SYSTEM PRIVATE
        ${SD3P}  # for <nlohmann/json.hpp>, as Stardust includes it
        ${SD3P}/openbw/bwapi/bwapi/Util/Source
        ${SD3P}/openbw/bwapi/bwapi/BWAPI/openbw
        ${SD3P}/openbw/bwapi/bwapi/BWAPI/openbw/BWAPI
        ${SD3P}/openbw/bwapi/bwapi/BWAPI/Source
        ${SD3P}/openbw/bwapi/bwapi/BWAPICore
        ${SD3P}/openbw/bwapi/bwapi/OpenBWData
        ${SD3P}/openbw/bwapi/bwapi/Shared
        ${OPENBW_DIR}
        )

target_link_libraries(sc-extract PRIVATE BWAPI BWAPILIB BWEM FAP nlohmann)
target_compile_options(sc-extract PRIVATE -Wall)

set_target_properties(sc-extract PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

install(TARGETS sc-extract RUNTIME DESTINATION bin)
foreach (lib BWAPI OpenBWData BWAPILIB BWEM)
    get_target_property(type ${lib} TYPE)
    if (type STREQUAL "SHARED_LIBRARY")
        install(TARGETS ${lib} LIBRARY DESTINATION lib)
    endif ()
endforeach ()
