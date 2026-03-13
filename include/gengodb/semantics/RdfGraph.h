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
class IriDictionary {
private:
    std::unordered_map<IRI, int32_t> iri_to_id;
    std::vector<IRI> id_to_iri;
public:
    IriDictionary(const std::initializer_list<IRI>& l) {
        for (auto it = l.begin(); it != l.end(); it++) {
            int32_t id = static_cast<int32_t>(id_to_iri.size());
            id_to_iri.push_back(*it);
            iri_to_id.emplace(id_to_iri.back(), id);
        }
    }
    IriDictionary() {}
    ~IriDictionary() {}
    int32_t insert(const IRI& iri) {
        int32_t id = static_cast<int32_t>(id_to_iri.size());
        id_to_iri.push_back(iri);
        iri_to_id.emplace(id_to_iri.back(), id);
        return id;
    }
    int32_t get_or_insert(const IRI& iri) {
        auto it = iri_to_id.find(iri);
        if (it != iri_to_id.end())
            return it->second;
        int32_t id = static_cast<int32_t>(id_to_iri.size());
        id_to_iri.push_back(iri);
        iri_to_id.emplace(id_to_iri.back(), id);
        return id;
    }
    int32_t get_safe(const IRI& iri) const {
        auto it = iri_to_id.find(iri);
        if (it == iri_to_id.end())
            return -1;
        return it->second;
    }
    size_t size() const { return id_to_iri.size(); }
}; // IriDictionary
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
    RdfGraph get(const IRI& name) const;
};
struct RdfDatatypeInlineHelper {
    static const std::unordered_set<IRI> inlinedIRIs;
    /**
     * Checks if a literal value is inlined into the property table
     */
    bool isInlined(const IRI& datatype) const {
        auto it = inlinedIRIs.find(datatype);
        return it != inlinedIRIs.end();
    }
    /**
     * Writes the inlined value into 'out', undefined behavior if type cannot be inlined
     */
    void inlineValue(uint64_t* out, const std::any& in, const IRI& datatype) const {
        Namespace xsd = namespaces::XSD();
        if (datatype == xsd + "boolean")                inlineValue<bool>((bool*) out, in);
        else if (datatype == xsd + "byte")              inlineValue<int8_t>((int8_t*) out, in);
        else if (datatype == xsd + "double")            inlineValue<double>((double*) out, in);
        else if (datatype == xsd + "float")             inlineValue<float>((float*) out, in);
        else if (datatype == xsd + "int")               inlineValue<int32_t>((int32_t*) out, in);
        else if (datatype == xsd + "long")              inlineValue<int64_t>((int64_t*) out, in);
        else if (datatype == xsd + "short")             inlineValue<int16_t>((int16_t*) out, in);
        else if (datatype == xsd + "unsignedByte")      inlineValue<uint8_t>((uint8_t*) out, in);
        else if (datatype == xsd + "unsignedInt")       inlineValue<uint32_t>((uint32_t*) out, in);
        else if (datatype == xsd + "unsignedLong")      inlineValue<uint64_t>((uint64_t*) out, in);
        else if (datatype == xsd + "unsignedShort")     inlineValue<uint16_t>((uint16_t*) out, in);
        else assert(false && "unsupported datatype for inlining");
    }
    template<typename T>
    void inlineValue(T* out, const std::any& in) const {
        T v = std::any_cast<T>(in);
        T* ptr = (T*) out;
        *ptr = v;
    }
};
class NodeHelper {
private:
    RdfGraph* g;
public:
    NodeHelper(RdfGraph* g) : g(g) {}
    inline void ensureNode();
    inline int32_t resolveNode(const BlankNode& b);
    inline int32_t resolveNode(const IRI& iri);
    inline int32_t resolvePredicate(const IRI& p);
    inline void addLiteral(int32_t sid, int32_t pid, const Literal& o);
};
struct RdfGraph {
    IRI name;
    runtime::PropertyGraph* storage;
    IriDictionary nodes;
    IriDictionary relations;
    IriDictionary literalTypes;
    std::unordered_map<std::string_view, int32_t> bnodes;
    RdfGraph(const IRI& name, runtime::PropertyGraph* storage) 
        : name(name), storage(storage), nodeHelper(this) {}
    static RdfGraph create(const IRI& name) {
        auto* storage = runtime::PropertyGraph::create(DEFAULT_CAPACITY, DEFAULT_CAPACITY, DEFAULT_CAPACITY);
        RdfGraph graph(name, storage);
        RdfGraphRegistry::singleton()->add(graph);
        return graph;
    }
    static RdfGraph create(const IRI& name, const Graph& rdfGraph);
    static RdfGraph create(const IRI& name, void* node_ptr, size_t node_l, void* rel_ptr, size_t rel_l, void* prop_ptr, size_t prop_l);
    void addTriple(const Node& s, const Node& p, const Node& o) {
        if (!p.is_iri()) assert(false && "predicate must be an IRI");
        const auto& pred = p.as_iri();
        if (s.is_iri()) {
            const auto& subj = s.as_iri();
            if (o.is_iri())                 addTriple(subj, pred, o.as_iri());
            else if (o.is_blank_node())     addTriple(subj, pred, o.as_blank_node());
            else if (o.is_literal())        addTriple(subj, pred, o.as_literal());
            else assert(false && "triple object invalid");

        } else if (s.is_blank_node()) {
            const auto& subj = s.as_blank_node();

            if (o.is_iri())                 addTriple(subj, pred, o.as_iri());
            else if (o.is_blank_node())     addTriple(subj, pred, o.as_blank_node());
            else if (o.is_literal())        addTriple(subj, pred, o.as_literal());
            else assert(false && "triple object invalid");

        } else {
            assert(false && "triple subject invalid");
        }
    }
    void addTriple(const IRI& s, const IRI& p, const IRI& o);
    void addTriple(const IRI& s, const IRI& p, const BlankNode& o);
    void addTriple(const IRI& s, const IRI& p, const Literal& o);
    void addTriple(const BlankNode& s, const IRI& p, const IRI& o);
    void addTriple(const BlankNode& s, const IRI& p, const BlankNode& o);
    void addTriple(const BlankNode& s, const IRI& p, const Literal& o);
    int32_t nodeId(const IRI& res) const { return nodes.get_safe(res); }
    int32_t relationId(const IRI& iri) const { return relations.get_safe(iri); }
    int32_t typeId(const IRI& t) const { return literalTypes.get_safe(t); }
    static const size_t DEFAULT_CAPACITY = 1024;
private:
    NodeHelper nodeHelper;
};

}

#endif // GENGODB_SEMANTICS_RDFGRAPH_H