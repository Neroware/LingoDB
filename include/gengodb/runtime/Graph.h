#ifndef GENGODB_RUNTIME_GRAPH_H
#define GENGODB_RUNTIME_GRAPH_H

#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/Buffer.h"
#include <cassert>

namespace lingodb::runtime {

typedef int32_t node_id_t;
typedef int32_t relation_id_t;
typedef uint32_t relation_type_id_t;
typedef int32_t property_id_t;
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
    static std::vector<GraphStorageData> graphs;
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
        graphs.push_back(graphData);
    }
    static size_t getNodeCount(GraphBase* g) { return g->nodeCounter; }
    static size_t getRelationshipCount(GraphBase* g) { return g->relCounter; }
    static size_t getPropertyCount(GraphBase* g) { return g->propCounter; }
    static uint8_t* getNodeBufferPtr(uint8_t* ref);
    static uint8_t* getRelBufferPtr(uint8_t* ref);
    static uint8_t* getPropBufferPtr(uint8_t* ref);
    static size_t getNodeBufferLen(uint8_t* ref);
    static size_t getRelBufferLen(uint8_t* ref);
    static size_t getPropBufferLen(uint8_t* ref);
    static BufferIterator* createNodeIterator(uint8_t* ref);
    static BufferIterator* createRelIterator(uint8_t* ref);
    static BufferIterator* createPropIterator(uint8_t* ref);
    static node_id_t getNodeId(uint8_t* node);
    static relation_id_t getRelationshipId(uint8_t* rel);
    static property_id_t getPropId(uint8_t* prop);
    static uint8_t* getRelationshipLListHeadOf(uint8_t* node);
protected:
    // Based on an address in memory, determine the graph storage
    static const GraphStorageData& getGraphInfo(uint8_t* ref) {
        for (const auto& graphData : graphs) {
            if (((uint8_t*) graphData.graphPtr == ref)
                || (graphData.nodeBufferPtr <= ref && ref < graphData.nodeBufferPtr + graphData.nodeBufferSize)
                || (graphData.relBufferPtr <= ref && ref < graphData.relBufferPtr + graphData.relBufferSize)
                || (graphData.propBufferPtr <= ref && ref < graphData.propBufferPtr + graphData.propBufferSize)) {
                    return graphData;
            }
        }
        assert(false && "should not happen");
    }
}; // GraphStorageHelper

}; // namespace lingodb::runtime

#endif // GENGODB_RUNTIME_GRAPH_H