#include "lingodb/utility/Serialization.h"

#include <rdf4cpp.hpp>

namespace gengodb::catalog {
using namespace lingodb;
struct TurtleFileRef {
    std::string path;
    rdf4cpp::IRI graph;

    void serialize(utility::Serializer& serializer) const;
    static TurtleFileRef deserialize(utility::Deserializer& deserializer);
};
}