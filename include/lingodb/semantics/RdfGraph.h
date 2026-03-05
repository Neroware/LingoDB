#ifndef LINGODB_SEMANTICS_RDFGRAPH_H
#define LINGODB_SEMANTICS_RDFGRAPH_H

#include <rdf4cpp.hpp>
#include "lingodb/runtime/Graph/PropertyGraph.h"

using namespace rdf4cpp;

namespace lingodb::semantics {

struct extra_namespaces {
    const Namespace PI = Namespace("https://www.uni-augsburg.de/de/fakultaet/fai/informatik/prof/pi#");
    const Namespace SOBOT = Namespace("https://www.forsocialrobots.de/ontologies/sobots.owl#");
    const Namespace LINGODB = Namespace("https://www.lingo-db.com/rdf#");
};
struct RdfGraph {
    IRI name;
    runtime::PropertyGraph* storage;
    std::unordered_map<IRI, uint32_t> iris;
    std::unordered_map<BlankNode, uint32_t> bnodes;
    std::unordered_map<IRI, uint32_t> relTypes;
    std::unordered_map<IRI, uint32_t> literalTypes;
    static RdfGraph create(const Graph& rdfGraph, const IRI& name);
};

}

#endif // LINGODB_SEMANTICS_RDFGRAPH_H