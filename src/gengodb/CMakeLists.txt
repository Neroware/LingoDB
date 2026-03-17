add_library(gengodb RdfGraph.cpp GraphCatalogEntry.cpp CreateRdfGraphDef.cpp)
target_link_libraries(gengodb PRIVATE rdf4cpp::rdf4cpp LibXml2)