#include "gengodb/CreateRdfGraphDef.h"

#include <rdf4cpp.hpp>

void gengodb::catalog::CreateRdfGraphDef::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(1, name);
   serializer.writeProperty(2, graph.identifier());
   serializer.writeProperty(3, (int) format);
}
gengodb::catalog::CreateRdfGraphDef gengodb::catalog::CreateRdfGraphDef::deserialize(utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(1);
   auto graph = deserializer.readProperty<std::string>(2);
   auto format = deserializer.readProperty<int>(3);
   return CreateRdfGraphDef{name, rdf4cpp::IRI{graph}, (rdf4cpp::parser::ParsingFlag) format};
}
