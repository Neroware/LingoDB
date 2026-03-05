#ifndef LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H
#define LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H

#include "lingodb/runtime/Graph/Graph.h"

namespace lingodb::runtime {

// Implementation of a native property graph following Graph Databases, 2nd Edition by Ian Robinson, Jim Webber & Emil Eifrem
// See: https://www.oreilly.com/library/view/graph-databases-2nd/9781491930885/ (Figure 6-4)
//
// The property table can hold up to 8 bytes of inlined data.
class PropertyGraph : public Graph<property_id_t, property_id_t, uint64_t> {
private:
    PropertyGraph(size_t maxNodeCapacity, size_t maxRelCapacity, size_t maxPropCapacity) 
        : Graph(maxNodeCapacity, maxRelCapacity, maxPropCapacity) {}
    node_id_t getNodeId(NodeEntry* node) const;
    NodeEntry* getNode(node_id_t node) const;
    relation_id_t getRelationshipId(RelationshipEntry* rel) const;
    RelationshipEntry* getRelationship(relation_id_t rel) const;
    property_id_t getPropertyId(PropertyEntry* prop) const;
    PropertyEntry* getProperty(property_id_t prop) const;

public:
    node_id_t addNode();
    relation_id_t addRelationship(node_id_t from, node_id_t to, relation_type_id_t type);
    node_id_t removeNode(node_id_t node);
    relation_id_t removeRelationship(relation_id_t rel);
    property_id_t addNodeProperty(node_id_t node, property_key_t key, property_type_id_t type, uint64_t initial_value = 0);
    property_id_t addRelationshipProperty(relation_id_t rel, property_key_t key, property_type_id_t type, uint64_t initial_value = 0);
    property_id_t removeProperty(property_id_t prop);
    void setProperty(property_id_t prop, uint64_t value);
    static PropertyGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity, size_t initialPropertyCapacity);
    static void destroy(PropertyGraph* graph) { delete graph; }
}; // PropertyGraph
struct PropertyGraphStorageHelper : public GraphStorageHelper {
    static uint8_t* getNodePropertyLListHeadOf(uint8_t* nodeRef);
    static uint8_t* getRelPropertyLListHeadOf(uint8_t* relRef);
}; // PropertyGraphStorageHelper

} // lingodb::runtime

#endif // LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H