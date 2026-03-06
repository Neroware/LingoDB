add_library(semantics RdfGraph.cpp)
target_link_libraries(semantics PRIVATE rdf4cpp::rdf4cpp)