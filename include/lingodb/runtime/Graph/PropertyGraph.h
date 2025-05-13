#ifndef LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H
#define LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H

#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/Buffer.h"

namespace lingodb::runtime::graph {

// Implementation of a native property graph following Graph Databases, 2nd Edition by Ian Robinson, Jim Webber & Emil Eifrem
// See: https://www.oreilly.com/library/view/graph-databases-2nd/9781491930885/ (Figure 6-4)
class PropertyGraph {
    struct NodeEntry;
    struct RelationshipEntry;
    struct NodeEntry {
        bool inUse;
        RelationshipEntry* nextRelationship;
        int64_t property; // TODO for now we only support a single node property of type i64
        int64_t id;
    }; // NodeEntry
    struct RelationshipEntry {
        bool inUse;
        NodeEntry* firstNode;
        NodeEntry* secondNode;
        int64_t type;
        RelationshipEntry* firstPrevRelation;
        RelationshipEntry* firstNextRelation;
        RelationshipEntry* secondPrevRelation;
        RelationshipEntry* secondNextRelation;
        int64_t property; // TODO for now we only support a single edge property of type i64
        int64_t id;
        bool firstInChain;
    }; // RelationshipEntry
    runtime::FlexibleBuffer nodes;
    runtime::FlexibleBuffer relationships;
    PropertyGraph(size_t initialNodeCapacity, size_t initialRelationshipCapacity) : nodes(initialNodeCapacity, sizeof(NodeEntry)), relationships(initialRelationshipCapacity, sizeof(RelationshipEntry)) {}

    public:
    bool addNode(int64_t id);
    bool addRelationship(int64_t id, int64_t from, int64_t to);
    bool removeNode(int64_t id);
    bool removeRelationship(int64_t id);

    bool setNodeProperty(int64_t id, int64_t value);
    int64_t getNodeProperty(int64_t id);
    bool setRelationshipProperty(int64_t id, int64_t value);
    int64_t getRelationshipProperty(int64_t id);

    static PropertyGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity);
    static void destroy(PropertyGraph*);

    runtime::BufferIterator* createNodeIterator();
    runtime::BufferIterator* createEdgeIterator();

}; // PropertyGraph
} // lingodb::runtime::graph

#endif // LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H