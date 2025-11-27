#ifndef LINGODB_RUNTIME_GRAPH_GRAPH_H
#define LINGODB_RUNTIME_GRAPH_GRAPH_H

#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/Buffer.h"
#include <cassert>

namespace lingodb::runtime {

typedef int64_t node_id_t;
typedef int64_t edge_id_t;
typedef uint64_t relation_type_id_t;
struct GraphBase {
    uint8_t* nodeBufferPtr;
    size_t nodeBufferLen;
    uint8_t* relBufferPtr;
    size_t relBufferLen;
}; // GraphBase
struct NodeEntryBase {
    bool inUse;
    node_id_t id;
    edge_id_t nextRelationship;
}; // NodeEntryBase
struct RelationshipEntryBase {
    bool inUse;
    edge_id_t id;
    node_id_t firstNode;
    node_id_t secondNode;
    relation_type_id_t type;
    edge_id_t firstPrevRelation;
    edge_id_t firstNextRelation;
    edge_id_t secondPrevRelation;
    edge_id_t secondNextRelation;
}; // RelationshipEntryBase

// Basic graph implementation following Graph Databases, 2nd Edition by Ian Robinson, Jim Webber & Emil Eifrem
// See: https://www.oreilly.com/library/view/graph-databases-2nd/9781491930885/ (Figure 6-4)
template<typename T, typename U>
class Graph : public GraphBase {
    protected:
    node_id_t nodeCounter = 0;
    edge_id_t relCounter = 0;
    struct NodeEntry : public NodeEntryBase {
        T property;
    }; // NodeEntry
    struct RelationshipEntry : public RelationshipEntryBase {
        U property;
    }; // RelationshipEntry
    runtime::LegacyFixedSizedBuffer<NodeEntry> nodes;
    runtime::LegacyFixedSizedBuffer<RelationshipEntry> relationships;
    std::vector<NodeEntry*> unusedNodeEntries;
    std::vector<RelationshipEntry*> unusedRelEntries;
    Graph(size_t maxNodeCapacity, size_t maxRelCapacity) 
        : nodeCounter(0), relCounter(0), nodes(maxNodeCapacity), relationships(maxRelCapacity) {
            nodeBufferPtr = (uint8_t*) nodes.ptr;
            nodeBufferLen = 0;
            relBufferPtr = (uint8_t*) relationships.ptr;
            relBufferLen = 0;
    }
    
    node_id_t getNodeId(NodeEntry* node) const {
        return node - nodes.ptr;
    }
    NodeEntry* getNode(node_id_t node) const {
        return nodes.ptr + node;
    }
    edge_id_t getRelationshipId(RelationshipEntry* rel) const {
        return rel - relationships.ptr;
    }
    RelationshipEntry* getRelationship(edge_id_t rel) const {
        return relationships.ptr + rel;
    }
    node_id_t addNode(T property) {
        NodeEntry* node;
        if (unusedNodeEntries.empty()) {
            node = nodes.getPtr(nodeCounter++);
            nodeBufferLen += sizeof(NodeEntry);
        }
        else {
            node = unusedNodeEntries.back();
            unusedNodeEntries.pop_back();
        }
        assert(!node->inUse && "should not happen");
        node_id_t nodeId = getNodeId(node);
        node->inUse = true;
        node->id = nodeId;
        node->nextRelationship = -1;
        node->property = property;
        return nodeId;
    }
    edge_id_t addRelationship(node_id_t from, node_id_t to, relation_type_id_t type, U property) {
        RelationshipEntry* rel;
        NodeEntry *fromNode = getNode(from), *toNode = getNode(to);
        if (unusedRelEntries.empty()) {
            rel = relationships.getPtr(relCounter++);
            relBufferLen += sizeof(RelationshipEntry);
        }
        else {
            rel = unusedRelEntries.back();
            unusedRelEntries.pop_back();
        }
        assert(!rel->inUse && "should not happen");
        edge_id_t relId = getRelationshipId(rel);
        rel->inUse = true;
        rel->id = relId;
        rel->firstNode = from;
        rel->secondNode = to;
        rel->type = type;
        rel->firstNextRelation = rel->firstPrevRelation = rel->secondNextRelation = rel->secondPrevRelation = -1;
        rel->property = property;
        if (fromNode->nextRelationship >= 0) {
            RelationshipEntry* head = getRelationship(fromNode->nextRelationship);
            if (head->firstNode == from) {
                head->firstPrevRelation = relId;
                rel->firstNextRelation = head->id;   
            }
            else {
                head->secondPrevRelation = relId;
                rel->firstNextRelation = head->id;
            }
        }
        fromNode->nextRelationship = relId;
        if (from != to) {
            if (toNode->nextRelationship >= 0) {
                RelationshipEntry* head = getRelationship(toNode->nextRelationship);
                if (head->firstNode == to) {
                    head->firstPrevRelation = relId;
                    rel->secondNextRelation = head->id;   
                }
                else {
                    head->secondPrevRelation = relId;
                    rel->secondNextRelation = head->id;
                }
            }
            toNode->nextRelationship = relId;
        }

        return relId;
    }
    node_id_t removeNode(node_id_t node) {
        assert(false && "not impelemented"); // TODO implement
    }
    edge_id_t removeRelationship(edge_id_t rel) {
        assert(false && "not impelemented"); // TODO implement
    }
}; // Graph
struct GraphHelper {
    static BufferIterator* createNodeIterator(GraphBase* graph);
    static BufferIterator* createEdgeIterator(GraphBase* graph);
    static void* getNodeBufferPtr(GraphBase* graph) { return graph->nodeBufferPtr; }
    static size_t getNodeBufferLen(GraphBase* graph) { return graph->nodeBufferLen; }
    static void* getEdgeBufferPtr(GraphBase* graph) { return graph->relBufferPtr; }
    static size_t getEdgeBufferLen(GraphBase* graph) { return graph->relBufferLen; }
    static void* getLinkedEgdesLListHeadOf(GraphBase* graph, uint8_t* ref, size_t refSize);

    // Resolves a graph reference to its graph instance

    static GraphBase* getGraphByNodeRef(uint8_t* ref, size_t refSize);
    static GraphBase* getGraphByEdgeRef(uint8_t* ref, size_t refSize);

    // Keeps track of all graph states
    static std::unordered_map<uint8_t*, GraphBase*> graphs;
}; // GraphHelper
class TestGraph : public Graph<uint64_t, uint64_t> {
private:
    TestGraph(size_t maxNodeCapacity, size_t maxRelCapacity) 
        : Graph(maxNodeCapacity, maxRelCapacity) {}
public:
    static TestGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity);
    static TestGraph* createTestGraph();
    node_id_t addNode() { return Graph::addNode(0); }
    edge_id_t addRelationship(node_id_t from, node_id_t to, relation_type_id_t type) { 
        return Graph::addRelationship(from, to, type, 0); 
    }
    size_t getNodeCount() const { return nodeCounter; }
    size_t getEdgeCount() const { return relCounter; }
}; // TestGraph

} // lingodb::runtime::graph

#endif // LINGODB_RUNTIME_GRAPH_GRAPH_H