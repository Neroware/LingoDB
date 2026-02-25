#ifndef LINGODB_RUNTIME_GRAPH_GRAPH_H
#define LINGODB_RUNTIME_GRAPH_GRAPH_H

#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/Buffer.h"
#include <cassert>

namespace lingodb::runtime {

typedef uint32_t node_id_t;
typedef uint32_t relation_id_t;
typedef uint32_t relation_type_id_t;
typedef uint32_t property_id_t;
typedef uint32_t property_type_id_t;
typedef uint32_t property_key_t;

struct GraphBase {
    struct NodeEntry {
        bool inUse;
        relation_id_t nextRelId;
    };
    struct RelationshipEntry {
        bool inUse;
        node_id_t firstNode;
        node_id_t secondNode;
        relation_type_id_t relationshipType;
        relation_id_t firstPrevRelId;
        relation_id_t firstNextRelId;
        relation_id_t secondPrevRelId;
        relation_id_t secondNextRelId;
    };
    size_t nodeCounter;
    size_t relCounter;
    size_t propCounter;
    GraphBase() : nodeCounter(0), relCounter(0), propCounter(0) {}
};
// Basic graph implementation following Graph Databases, 2nd Edition by Ian Robinson, Jim Webber & Emil Eifrem
// See: https://www.oreilly.com/library/view/graph-databases-2nd/9781491930885/ (Figure 6-4)
template<typename T, typename U, typename V = void*> 
struct Graph : public GraphBase {
    struct NodeEntry : GraphBase::NodeEntry {
        T nextPropId;
        uint8_t labels[5];
        uint8_t extra;
    };
    struct RelationshipEntry : GraphBase::RelationshipEntry {
        U nextPropId;
        bool firstInChainMarker;
    };
    struct PropertyEntry {
        bool inUse;
        property_id_t nextProp;
        property_id_t prevProp;
        property_key_t key;
        property_type_id_t type;
        V value;
    }; // PropertyEntry
    runtime::LegacyFixedSizedBuffer<NodeEntry> nodes;
    runtime::LegacyFixedSizedBuffer<RelationshipEntry> relationships;
    runtime::LegacyFixedSizedBuffer<PropertyEntry> properties;
    std::vector<NodeEntry*> unusedNodeEntries;
    std::vector<RelationshipEntry*> unusedRelEntries;
    std::vector<PropertyEntry*> unusedPropEntries;
    Graph(size_t maxNodeCapacity, size_t maxRelCapacity, size_t maxPropCapacity) 
        : GraphBase(), nodes(maxNodeCapacity), relationships(maxRelCapacity), 
        properties(maxPropCapacity) {}
}; // Graph
// A basic graph without a property table using i64 as property value
class GengoDBGraph : public Graph<uint64_t, uint64_t> {
private:
    node_id_t getNodeId(NodeEntry* node) const;
    NodeEntry* getNode(node_id_t node) const;
    relation_id_t getRelationshipId(RelationshipEntry* rel) const;
    RelationshipEntry* getRelationship(relation_id_t rel) const;
    GengoDBGraph(size_t maxNodeCapacity, size_t maxRelCapacity)
        : Graph(maxNodeCapacity, maxRelCapacity, 1) {}
public:
    node_id_t addNode();
    relation_id_t addRelationship(node_id_t from, node_id_t to);
    node_id_t removeNode(node_id_t node);
    relation_id_t removeRelationship(relation_id_t rel);
    void setNodeValue(node_id_t node, uint64_t value) const;
    void setRelationshipValue(relation_id_t edge, uint64_t value) const;
    uint64_t getNodeValue(node_id_t node) const;
    uint64_t getRelationshipValue(relation_id_t edge) const;
    size_t getNodeCount() const { return nodeCounter; }
    size_t getRelCount() const { return relCounter; }
    static GengoDBGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity);
    static void destroy(GengoDBGraph* graph) { delete graph; }
}; // GengoDBGraph
struct GraphStorageHelper {
    struct GraphStorageData {
        GraphBase* graphPtr;
        size_t nodeEntrySize;
        size_t relEntrySize;
        size_t propEntrySize;
        uint8_t* nodeBufferPtr;
        uint8_t* relBufferPtr;
        uint8_t* propBufferPtr;
        size_t nodeBufferSize;
        size_t relBufferSize;
        size_t propBufferSize;
    };
    static std::unordered_map<GraphBase*, GraphStorageData> graphs;
    template<typename T, typename U, typename V>
    static void addGraph(Graph<T, U, V>* graph, size_t maxNodeCapacity, size_t maxRelCapacity, size_t maxPropCapacity){
        GraphStorageData graphData {
            (GraphBase*) graph,
            sizeof(typename Graph<T, U, V>::NodeEntry),
            sizeof(typename Graph<T, U, V>::RelationshipEntry),
            sizeof(typename Graph<T, U, V>::PropertyEntry),
            (uint8_t*) graph->nodes.ptr,
            (uint8_t*) graph->relationships.ptr,
            (uint8_t*) graph->properties.ptr,
            maxNodeCapacity * sizeof(typename Graph<T, U, V>::NodeEntry),
            maxRelCapacity * sizeof(typename Graph<T, U, V>::RelationshipEntry),
            maxPropCapacity * sizeof(typename Graph<T, U, V>::PropertyEntry)
        };
        graphs[(GraphBase*) graph] = graphData;
    }
    // Based on an address in memory, determine the graph storage
    static GraphBase* getGraphByRef(uint8_t* ref) {
        for (const auto& graphData : graphs) {
            if ((graphData.second.nodeBufferPtr <= ref && ref < graphData.second.nodeBufferPtr + graphData.second.nodeBufferSize)
                || (graphData.second.relBufferPtr <= ref && ref < graphData.second.relBufferPtr + graphData.second.relBufferSize)
                || (graphData.second.propBufferPtr <= ref && ref < graphData.second.propBufferPtr + graphData.second.propBufferSize)){
                    return graphData.first;
            }
        }
        return nullptr;
    }
    static uint8_t* getNodeBufferPtr(GraphBase* graph) { return graphs[graph].nodeBufferPtr; }
    static uint8_t* getRelBufferPtr(GraphBase* graph) { return graphs[graph].relBufferPtr; }
    static uint8_t* getPropBufferPtr(GraphBase* graph) { return graphs[graph].propBufferPtr; }
    static size_t getNodeBufferLen(GraphBase* graph) { return graph->nodeCounter * graphs[graph].nodeEntrySize; }
    static size_t getRelBufferLen(GraphBase* graph) { return graph->relCounter * graphs[graph].relEntrySize; }
    static size_t getPropBufferLen(GraphBase* graph) { return graph->propCounter * graphs[graph].propEntrySize; }
    static BufferIterator* createNodeIterator(GraphBase* graph);
    static BufferIterator* createRelIterator(GraphBase* graph);
    static BufferIterator* createPropIterator(GraphBase* graph);
    static uint8_t* getRelationshipLListHead(GraphBase* graph, uint8_t* node);
    static GraphBase* createTestGraph(uint64_t whichOne);
}; // GraphStorageHelper

}; // namespace lingodb::runtime

#endif // LINGODB_RUNTIME_GRAPH_GRAPH_H