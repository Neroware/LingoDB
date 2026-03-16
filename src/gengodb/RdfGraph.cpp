#include "gengodb/RdfGraph.h"

#include <rdf4cpp/Graph.hpp>
#include <rdf4cpp/parser/RDFFileParser.hpp>

namespace gengodb::semantics {
using namespace rdf4cpp::parser;

inline void NodeHelper::ensureNode() {
    if (g->storage->nodeCounter < g->nodes.size()) {
        g->storage->addNode();
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
std::unique_ptr<RdfGraph> RdfGraph::create(const gengodb::catalog::CreateRdfGraphDef& def) {
    auto storage = runtime::GengoDBGraph::create(def.name);
    auto dbDir = storage->getDBDir();
    auto rdfGraph = std::make_unique<RdfGraph>(def.graph, std::move(storage));
    RDFFileParser parser(dbDir + def.name + ".rdf", def.format);
    for (const auto &v : parser) {
        if (!v.has_value())
            break;
        auto quad = v.value();
        rdfGraph->addTriple(quad.subject(), quad.predicate(), quad.object());
    }
    storage->ensureLoaded();
    return rdfGraph;
}
std::unique_ptr<RdfGraph> create(const std::string& name, const IRI& iri) {
    auto storage = runtime::GengoDBGraph::create(name);
    auto rdfGraph = std::make_unique<RdfGraph>(iri ? iri : extra_namespaces().GENGODB + name, std::move(storage));
    storage->ensureLoaded();
    return rdfGraph;
}


} // lingodb::semantics