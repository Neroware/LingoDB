#include "gengodb/semantics/RdfGraph.h"

namespace gengodb::semantics {

RdfGraph RdfGraph::create(const IRI& name, const Graph& rdfGraph) {
    auto g = RdfGraph::create(name);
    for (auto it = rdfGraph.begin(); it != rdfGraph.end(); ++it) {
        g.addTriple(it->subject(), it->predicate(), it->object());
    }
    return g;
}
RdfGraph RdfGraph::create(const IRI& name, void* node_ptr, size_t node_l, void* rel_ptr, size_t rel_l, void* prop_ptr, size_t prop_l) {
    auto g = RdfGraph::create(name);
    void* dst_node_ptr = runtime::GraphStorageHelper::getNodeBufferPtr((uint8_t*) g.storage);
    memcpy(dst_node_ptr, node_ptr, node_l);
    void* dst_rel_ptr = runtime::GraphStorageHelper::getRelBufferPtr((uint8_t*) g.storage);
    memcpy(dst_rel_ptr, rel_ptr, rel_l);
    void* dst_prop_ptr = runtime::GraphStorageHelper::getPropBufferPtr((uint8_t*) g.storage);
    memcpy(dst_prop_ptr, prop_ptr, prop_l);
    return g;
}
void RdfGraphRegistry::add(const RdfGraph& g) {
    knownGraphs.insert(std::make_pair(g.name, g));
}
RdfGraph RdfGraphRegistry::get(const IRI& name) const { 
    auto g = knownGraphs.find(name);
    if (g == knownGraphs.end()) {
        return RdfGraph::create(name);
    }
    return g->second;
}
void RdfGraphRegistry::loadAll() {
    // TODO implement
}
void RdfGraphRegistry::load(const IRI& g) {
    // TODO implement
}
RdfGraph RdfGraphRegistry::loadFromFile(const std::string& file, const IRI& name) {
    // TODO implement
    assert(false && "not implemented");
}
const std::unordered_set<IRI> RdfDatatypeInlineHelper::inlinedIRIs = {
    IRI(datatypes::xsd::Boolean::identifier),
    IRI(datatypes::xsd::Byte::identifier),
    IRI(datatypes::xsd::Double::identifier),
    IRI(datatypes::xsd::Float::identifier),
    IRI(datatypes::xsd::Int::identifier),
    IRI(datatypes::xsd::Long::identifier),
    IRI(datatypes::xsd::Short::identifier),
    IRI(datatypes::xsd::UnsignedByte::identifier),
    IRI(datatypes::xsd::UnsignedInt::identifier),
    IRI(datatypes::xsd::UnsignedLong::identifier),
    IRI(datatypes::xsd::UnsignedShort::identifier),
};

} // lingodb::semantics