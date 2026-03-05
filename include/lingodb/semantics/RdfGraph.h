#ifndef LINGODB_SEMANTICS_RDFGRAPH_H
#define LINGODB_SEMANTICS_RDFGRAPH_H

#include "rdf4cpp.hpp"

using namespace rdf4cpp;

namespace lingodb::semantics {

struct RdfGraph {
    IRI foo() {
        auto iri = IRI{"http://ex.com/MyGraph"};
        return iri;
    }
};

}

#endif // LINGODB_SEMANTICS_RDFGRAPH_H