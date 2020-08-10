cmake_minimum_required(VERSION 3.8)

set(ANTLR_EXECUTABLE /usr/local/lib/antlr-4.7.2-complete.jar)
find_package(ANTLR REQUIRED)

antlr_target(GrammarOutput dada.g4 OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/generated VISITOR)

add_library(
  grammar
  ${ANTLR_GrammarOutput_CXX_OUTPUTS}
  visitor.cpp
)

target_link_libraries(
  grammar PUBLIC
  antlr4_static
)

target_include_directories(grammar PUBLIC ${CMAKE_SOURCE_DIR})