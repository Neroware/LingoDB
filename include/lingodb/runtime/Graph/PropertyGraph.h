#ifndef LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H
#define LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H

#include "lingodb/runtime/Graph/Graph.h"

namespace lingodb::runtime {

typedef int64_t property_id_t;
typedef uint64_t property_type_id_t;
typedef uint64_t property_key_t;

// Implementation of a native property graph following Graph Databases, 2nd Edition by Ian Robinson, Jim Webber & Emil Eifrem
// See: https://www.oreilly.com/library/view/graph-databases-2nd/9781491930885/ (Figure 6-4)
class PropertyGraph : public LingoDBGraph {
    private:
    struct PropertyEntry {
        bool inUse;
        property_id_t id;
        property_id_t nextProp;
        property_id_t prevProp;
        property_key_t key;
        property_type_id_t type;
        uint64_t value;
    }; // PropertyEntry
    runtime::LegacyFixedSizedBuffer<PropertyEntry> properties;
    std::vector<PropertyEntry*> unusedPropEntries;
    PropertyGraph(size_t maxNodeCapacity, size_t maxRelCapacity, size_t maxPropCapacity) 
        : LingoDBGraph(maxNodeCapacity, maxRelCapacity), properties(maxPropCapacity) {}
    
    property_id_t propBufferSize = 0;

    property_id_t getPropertyId(PropertyEntry* prop) const;
    PropertyEntry* getProperty(property_id_t prop) const;

    public:
    node_id_t addNode() { return Graph::addNode(-1); }
    edge_id_t addRelationship(node_id_t from, node_id_t to, relation_type_id_t type) { return Graph::addRelationship(from, to, type, -1); }
    node_id_t removeNode(node_id_t node) { return Graph::removeNode(node); }
    edge_id_t removeRelationship(edge_id_t rel) { return Graph::removeRelationship(rel); }

    property_id_t addNodeProperty(node_id_t node, property_key_t key, property_type_id_t type, uint64_t initial_value = 0);
    property_id_t addRelationshipProperty(edge_id_t rel, property_key_t key, property_type_id_t type, uint64_t initial_value = 0);

    property_id_t removeProperty(property_id_t prop);
    void setProperty(property_id_t prop, uint64_t value);

    static PropertyGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity, size_t initialPropertyCapacity);
    static void destroy(PropertyGraph*);

    Buffer getPropBuffer() const { return Buffer{(size_t) propBufferSize * sizeof(PropertyEntry), (uint8_t*) properties.ptr }; }
    BufferIterator* createPropIterator();
    void* getPropBufferPtr() const { return (void*) properties.ptr; }
    size_t getPropBufferLen() const { return propBufferSize * sizeof(PropertyEntry); }
    
    void* getNodePropertyLListHead(void* nodeRef) const;
    void* getEdgePropertyLListHead(void* relRef) const;

}; // PropertyGraph

} // lingodb::runtime

#endif // LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H