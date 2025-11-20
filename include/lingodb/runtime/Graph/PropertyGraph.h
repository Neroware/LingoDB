#ifndef LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H
#define LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H

#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/Buffer.h"

namespace lingodb::runtime {

typedef int64_t node_id_t;
typedef int64_t edge_id_t;
typedef uint64_t relation_type_id_t;
typedef int64_t property_id_t;
typedef uint64_t property_type_id_t;
typedef uint64_t property_key_t;

// Implementation of a native property graph following Graph Databases, 2nd Edition by Ian Robinson, Jim Webber & Emil Eifrem
// See: https://www.oreilly.com/library/view/graph-databases-2nd/9781491930885/ (Figure 6-4)
class PropertyGraph {
    private:
    struct NodeEntry {
        bool inUse;
        node_id_t id;
        edge_id_t nextRelationship;
        uint64_t property;
    }; // NodeEntry
    struct RelationshipEntry {
        bool inUse;
        edge_id_t id;
        node_id_t firstNode;
        node_id_t secondNode;
        relation_type_id_t type;
        edge_id_t firstPrevRelation;
        edge_id_t firstNextRelation;
        edge_id_t secondPrevRelation;
        edge_id_t secondNextRelation;
        uint64_t property;
    }; // RelationshipEntry
    struct PropertyEntry {
        property_id_t id;
        property_id_t nextProp;
        property_id_t prevProp;
        property_key_t key;
        property_type_id_t type;
        uint64_t value;
    }; // PropertyEntry
    runtime::LegacyFixedSizedBuffer<NodeEntry> nodes;
    runtime::LegacyFixedSizedBuffer<RelationshipEntry> relationships;
    runtime::LegacyFixedSizedBuffer<PropertyEntry> properties;
    std::vector<NodeEntry*> unusedNodeEntries;
    std::vector<RelationshipEntry*> unusedRelEntries;
    PropertyGraph(size_t maxNodeCapacity, size_t maxRelCapacity, size_t maxPropCapacity) 
        : nodes(maxNodeCapacity), relationships(maxRelCapacity), properties(maxPropCapacity) {}

    node_id_t nodeBufferSize = 0;
    edge_id_t relBufferSize = 0;
    property_id_t propBufferSize = 0;

    node_id_t getNodeId(NodeEntry* node) const;
    edge_id_t getRelationshipId(RelationshipEntry* rel) const;
    NodeEntry* getNode(node_id_t node) const;
    RelationshipEntry* getRelationship(edge_id_t rel) const;

    public:
    node_id_t addNode();
    edge_id_t addRelationship(node_id_t from, node_id_t to, relation_type_id_t type = 0);

    node_id_t removeNode(node_id_t node);
    edge_id_t removeRelationship(edge_id_t rel);

    void setNodeProperty(node_id_t id, uint64_t value);
    uint64_t getNodeProperty(node_id_t id) const;
    void setRelationshipProperty(edge_id_t id, uint64_t value);
    uint64_t getRelationshipProperty(edge_id_t id) const;

    static PropertyGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity, size_t initialPropertyCapacity);
    static PropertyGraph* createTestGraph();
    static void destroy(PropertyGraph*);

    // Methods aiding in grapth iterations

    Buffer getNodeBuffer() const { return Buffer{(size_t) nodeBufferSize * sizeof(NodeEntry), (uint8_t*) nodes.ptr }; }
    Buffer getEdgeBuffer() const { return Buffer{(size_t) relBufferSize * sizeof(RelationshipEntry), (uint8_t*) relationships.ptr }; }
    Buffer getPropBuffer() const { return Buffer{(size_t) propBufferSize * sizeof(PropertyEntry), (uint8_t*) properties.ptr }; }
    BufferIterator* createNodeIterator();
    BufferIterator* createEdgeIterator();
    BufferIterator* createPropIterator();
    void* getNodeBufferPtr() const { return (void*) nodes.ptr; }
    void* getEdgeBufferPtr() const { return (void*) relationships.ptr; }
    void* getPropBufferPtr() const { return (void*) properties.ptr; }
    size_t getNodeBufferLen() const { return nodeBufferSize; }
    size_t getEdgeBufferLen() const { return relBufferSize; }
    size_t getPropBufferLen() const { return propBufferSize; }
    void* getLinkedEgdesLListHead(void* nodeRef) const;

    // Resolves a graph reference to its graph instance

    static PropertyGraph* getGraphByNodeRef(void* ref);
    static PropertyGraph* getGraphByEdgeRef(void* ref);

    // Keeps track of all Property Graph states
    static std::unordered_map<void*, PropertyGraph*> graphs;

}; // PropertyGraphLinkedRelationshipsSet
} // lingodb::runtime::graph

#endif // LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H