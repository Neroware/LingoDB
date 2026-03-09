#ifndef GENGODB_SEMANTICS_RDFGRAPH_H
#define GENGODB_SEMANTICS_RDFGRAPH_H

#include "gengodb/runtime/PropertyGraph.h"
#include <rdf4cpp.hpp>

#include <iostream>

namespace gengodb::semantics {
using namespace rdf4cpp;
using namespace lingodb;

struct extra_namespaces {
    const Namespace PI = Namespace("https://www.uni-augsburg.de/de/fakultaet/fai/informatik/prof/pi#");
    const Namespace SOBOT = Namespace("https://www.forsocialrobots.de/ontologies/sobots.owl#");
    const Namespace LINGODB = Namespace("https://www.lingo-db.com/rdf#");
    const Namespace GENGODB = Namespace("https://github.com/Neroware/LingoDB#");
};
class ResourceDictionary {
private:
    std::unordered_map<Node, int32_t> res_to_id;
    std::vector<Node> id_to_res;
public:
    ResourceDictionary(const std::initializer_list<Node>& l) {
        for (auto it = l.begin(); it != l.end(); it++) {
            int32_t id = static_cast<int32_t>(id_to_res.size());
            id_to_res.push_back(*it);
            res_to_id.emplace(id_to_res.back(), id);
        }
    }
    ResourceDictionary() {}
    ~ResourceDictionary() {}
    int32_t get_or_insert(const Node& res) {
        auto it = res_to_id.find(res);
        if (it != res_to_id.end())
            return it->second;
        int32_t id = static_cast<int32_t>(id_to_res.size());
        id_to_res.push_back(res);
        res_to_id.emplace(id_to_res.back(), id);
        return id;
    }
    int32_t get_safe(const Node& res) const {
        auto it = res_to_id.find(res);
        if (it == res_to_id.end())
            return -1;
        return it->second;
    }
    size_t size() const { return id_to_res.size(); }
};
struct RdfGraph;
class RdfGraphRegistry {
private:
    std::unordered_map<IRI, RdfGraph> knownGraphs;
    static RdfGraphRegistry* singleton_;
public:
    RdfGraphRegistry() {}
    ~RdfGraphRegistry() {}
    static RdfGraphRegistry* singleton() {
        assert(singleton_ != nullptr && "init first");
        return singleton_;
    }
    static void init() { if (singleton_ == nullptr) singleton_ = new RdfGraphRegistry(); }
    void loadAll();
    void load(const IRI& g);
    RdfGraph loadFromFile(const std::string& file, const IRI& name);
    void add(const RdfGraph& g);
    RdfGraph get(const IRI& name);
};
/**
 * Checks if a literal value fits into the property table
 */
static bool isInlined(const IRI& datatype) {
    // TODO
    return true;
}
/**
 * Returns if a literal value can be inlined and writes the inlined value into 'out'
 */
static bool inlineValue(uint64_t& out, const std::any& in, const IRI& datatype) {
    if (!isInlined(datatype)) {
        return false;
    }
    // TODO
    out = 42;
    return true;
}
struct RdfGraph {
    IRI name;
    runtime::PropertyGraph* storage;
    ResourceDictionary nodes;
    ResourceDictionary relations;
    ResourceDictionary literalTypes;
    static RdfGraph create(const IRI& name) {
        auto* storage = runtime::PropertyGraph::create(DEFAULT_CAPACITY, DEFAULT_CAPACITY, DEFAULT_CAPACITY);
        RdfGraph graph{.name = name, .storage = storage, .nodes = {}, .relations = {}, .literalTypes = {}};
        RdfGraphRegistry::singleton()->add(graph);
        return graph;
    }
    static RdfGraph create(const IRI& name, const Graph& rdfGraph);
    void addTriple(const Node& s, const Node& p, const Node& o) {
        int32_t sid = nodes.get_or_insert(s);
        if (storage->nodeCounter < nodes.size()) {
            storage->addNode();
        }
        int32_t oid = nodes.get_or_insert(o);
        if (storage->nodeCounter < nodes.size()) {
            storage->addNode();
        }
        int32_t pid = relations.get_or_insert(p);
        storage->addRelationship(sid, oid, pid);
        if (o.is_literal()) {
            auto l = o.as_literal();
            auto datatype = literalTypes.get_or_insert(l.datatype());
            uint64_t v = 0;
            if (!inlineValue(v, l.value(), l.datatype())) {
                // TODO
                assert(false && "only inlined literals supported");
            }
            storage->addNodeProperty(oid, DATATYPE_PROPERTY_KEY, datatype, v);
        }
    }
    int32_t nodeId(const Node& res) const { return nodes.get_safe(res); }
    int32_t relationId(const IRI& iri) const { return relations.get_safe(iri); }
    int32_t typeId(const IRI& t) const { return literalTypes.get_safe(t); }
    static const size_t DEFAULT_CAPACITY = 1024;
    static const size_t DATATYPE_PROPERTY_KEY = 1;
};

}

#endif // GENGODB_SEMANTICS_RDFGRAPH_H