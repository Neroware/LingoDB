#ifndef GENGODB_CREATERDFGRAPHDEF_H
#define GENGODB_CREATERDFGRAPHDEF_H

#include "lingodb/utility/Serialization.h"

#include <rdf4cpp.hpp>
#include <rdf4cpp/parser/RDFFileParser.hpp>

namespace gengodb::catalog {
using namespace lingodb;
struct CreateRdfGraphDef {
    std::string name;
    rdf4cpp::IRI graph;
    rdf4cpp::parser::ParsingFlag format;

    void serialize(utility::Serializer& serializer) const;
    static CreateRdfGraphDef deserialize(utility::Deserializer& deserializer);
};
}

#endif // GENGODB_CREATERDFGRAPHDEF_H