#include "gengodb/semantics/RdfGraph.h"

namespace gengodb::semantics {

RdfGraph RdfGraph::create(const IRI& name,const Graph& rdfGraph) {
    return RdfGraph{.name = name, .storage = nullptr, .nodes = {}, .relations = {}, .literalTypes = {}};
}

} // lingodb::semantics