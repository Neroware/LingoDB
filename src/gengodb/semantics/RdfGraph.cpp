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
    g.storage->nodeCounter = node_l / sizeof(runtime::PropertyGraph::NodeEntry);
    void* dst_rel_ptr = runtime::GraphStorageHelper::getRelBufferPtr((uint8_t*) g.storage);
    memcpy(dst_rel_ptr, rel_ptr, rel_l);
    g.storage->relCounter = rel_l / sizeof(runtime::PropertyGraph::RelationshipEntry);
    void* dst_prop_ptr = runtime::GraphStorageHelper::getPropBufferPtr((uint8_t*) g.storage);
    memcpy(dst_prop_ptr, prop_ptr, prop_l);
    g.storage->propCounter = prop_l / sizeof(runtime::PropertyGraph::PropertyEntry);
    return g;
}
inline void NodeHelper::ensureNode() {
    if (g->storage->nodeCounter < g->nodes.size()) {
        auto id = g->storage->addNode();
        g->storage->addNodeProperty(id, -1, -1, id);
    }
}
inline int32_t NodeHelper::resolveNode(const BlankNode& b) {
    const auto ident = b.identifier();
    auto it = g->bnodes.find(ident);
    if (it != g->bnodes.end())
        return it->second;
    auto id = g->nodes.insert(IRI{});
    ensureNode();
    g->bnodes.emplace(ident, id);
    return id;
}
inline int32_t NodeHelper::resolveNode(const IRI& iri) {
    auto id = g->nodes.get_or_insert(iri);
    ensureNode();
    return id;
}
inline int32_t NodeHelper::resolvePredicate(const IRI& p) {
    return g->relations.get_or_insert(p);
}
inline void NodeHelper::addLiteral(int32_t sid, int32_t pid, const Literal& o) {
    RdfDatatypeInlineHelper inlineHelper;
    auto l = o.as_literal();
    auto datatype = g->literalTypes.get_or_insert(l.datatype());
    uint64_t v = 0;
    assert(inlineHelper.isInlined(l.datatype()) && "only inlined literals supported");
    inlineHelper.inlineValue(&v, l.value(), l.datatype());
    g->storage->addNodeProperty(sid, pid, datatype, v);
}
void RdfGraph::addTriple(const IRI& s, const IRI& p, const IRI& o) {
    storage->addRelationship(nodeHelper.resolveNode(s), nodeHelper.resolveNode(o), nodeHelper.resolvePredicate(p));
}
void RdfGraph::addTriple(const IRI& s, const IRI& p, const BlankNode& o) {
    storage->addRelationship(nodeHelper.resolveNode(s), nodeHelper.resolveNode(o), nodeHelper.resolvePredicate(p));
}
void RdfGraph::addTriple(const IRI& s, const IRI& p, const Literal& o) {
    nodeHelper.addLiteral(nodeHelper.resolveNode(s), nodeHelper.resolvePredicate(p), o);
}
void RdfGraph::addTriple(const BlankNode& s, const IRI& p, const IRI& o) {
    storage->addRelationship(nodeHelper.resolveNode(s), nodeHelper.resolveNode(o), nodeHelper.resolvePredicate(p));
}
void RdfGraph::addTriple(const BlankNode& s, const IRI& p, const BlankNode& o) {
    storage->addRelationship(nodeHelper.resolveNode(s), nodeHelper.resolveNode(o), nodeHelper.resolvePredicate(p));
}
void RdfGraph::addTriple(const BlankNode& s, const IRI& p, const Literal& o) {
    nodeHelper.addLiteral(nodeHelper.resolveNode(s), nodeHelper.resolvePredicate(p), o);
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
    assert(false && "not implemented");
}
void RdfGraphRegistry::load(const IRI& g) {
    // TODO implement
    assert(false && "not implemented");
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