#include "gengodb/semantics/RdfGraph.h"

namespace lingodb::semantics {

RdfGraph RdfGraph::create(const Graph& rdfGraph, const IRI& name) {
    return RdfGraph{.name = name, .storage = nullptr, .iris = {}, .bnodes = {}, .relTypes = {}, .literalTypes = {}};
}

} // lingodb::semantics