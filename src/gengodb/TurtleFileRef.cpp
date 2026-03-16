#include "gengodb/TurtleFileRef.h"

#include <rdf4cpp.hpp>

void gengodb::catalog::TurtleFileRef::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(1, path);
   serializer.writeProperty(2, graph.identifier());
}
gengodb::catalog::TurtleFileRef gengodb::catalog::TurtleFileRef::deserialize(utility::Deserializer& deserializer) {
   auto path = deserializer.readProperty<std::string>(1);
   auto graph = deserializer.readProperty<std::string>(2);
   return TurtleFileRef{path, rdf4cpp::IRI{graph}};
}
