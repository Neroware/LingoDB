#include <iostream>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <vector>

struct MemoryHelper {
   static uint8_t* resize(uint8_t* old, size_t oldNumBytes, size_t newNumBytes) {
      uint8_t* newBytes = (uint8_t*) malloc(newNumBytes);
      memcpy(newBytes, old, oldNumBytes);
      free(old);
      return newBytes;
   }
   static void fill(uint8_t* ptr, uint8_t val, size_t size) {
      memset(ptr, val, size);
   }
   static void zero(uint8_t* ptr, size_t size) { fill(ptr, 0, size); }
};

template <class T>
struct LegacyFixedSizedBuffer {
   LegacyFixedSizedBuffer(size_t size) : ptr((T*) malloc(size * sizeof(T))) {
      MemoryHelper::zero((uint8_t*) ptr, size * sizeof(T));
   }
   T* ptr;
   void setNewSize(size_t newSize) {
      free(ptr);
      ptr = (T*) malloc(newSize * sizeof(T));
      MemoryHelper::zero((uint8_t*) ptr, newSize * sizeof(T));
   }
   T& at(size_t i) {
      return ptr[i];
   }
   T* getPtr(size_t i) {
      return &ptr[i];
   }
   ~LegacyFixedSizedBuffer() {
      free(ptr);
   }
};

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
    public:
    node_id_t nodeCounter = 0;
    edge_id_t relCounter = 0;
    struct NodeEntry : public NodeEntryBase {
        T property;
    }; // NodeEntry
    struct RelationshipEntry : public RelationshipEntryBase {
        U property;
    }; // RelationshipEntry
    LegacyFixedSizedBuffer<NodeEntry> nodes;
    LegacyFixedSizedBuffer<RelationshipEntry> relationships;
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
    edge_id_t addRelationship(node_id_t from, node_id_t to, U property) {
        return addRelationship(from, to, 0, property);
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
}; // Graph

struct node_property_t {
    double rank;
    double newRank;
    int l;
};
struct rel_property_t {
    // intentionally empty...
};

Graph<node_property_t, rel_property_t>* createGraph() {
    auto g = new Graph<node_property_t, rel_property_t>(8, 64);
    for (int i = 0; i < 5; i++) {
        g->addNode(node_property_t{0.0, 0.0, 0});
    }
    auto property = rel_property_t{};
    g->addRelationship(0, 1, property);
    g->addRelationship(0, 3, property);
    g->addRelationship(1, 2, property);
    g->addRelationship(2, 4, property);
    g->addRelationship(3, 4, property);
    g->addRelationship(4, 1, property);
    return g;
}

void pagerank(Graph<node_property_t, rel_property_t>* graph, int repeats, double damping) {
    int nNodes = graph->nodeCounter;
    for (int n = 0; n < nNodes; n++) {
        auto node = graph->getNode(n);
        node->property.rank = 1.0 / nNodes;
        auto nextRel = node->nextRelationship;
        int degree = 0;
        while (nextRel >= 0) {
            auto rel = graph->getRelationship(nextRel);
            if (rel->inUse && rel->firstNode == n) 
                degree++;
            nextRel = n == rel->firstNode ? rel->firstNextRelation : rel->secondNextRelation;
        }
        node->property.l = degree;
    }

    for (int iter = 0; iter < repeats; iter++) {
        for(int n = 0; n < nNodes; n++) {
            graph->getNode(n)->property.newRank = 0.15 / nNodes;
        }
        for (int n = 0; n < nNodes; n++) {
            auto node = graph->getNode(n);
            auto nextRel = node->nextRelationship;
            while (nextRel >= 0) {
                auto rel = graph->getRelationship(nextRel);
                if (rel->inUse && rel->firstNode == n) {
                    auto toNode = graph->getNode(rel->secondNode);
                    toNode->property.newRank += damping * (node->property.rank / node->property.l);
                }
                nextRel = n == rel->firstNode ? rel->firstNextRelation : rel->secondNextRelation;
            }
        }
        for (int n = 0; n < nNodes; n++) {
            auto node = graph->getNode(n);
            node->property.rank = node->property.newRank;
            node->property.newRank = 0.0;
        }
    }
    
}

int main() {
    std::cout << "Running PageRank..." << std::endl;
    auto g = createGraph();
    pagerank(g, 1000, 0.85);

    for (int n = 0; n < g->nodeCounter; n++) {
        std::cout << "n = " << n << ", rank = " << g->getNode(n)->property.rank << ", l = " << g->getNode(n)->property.l << std::endl;
    }

    delete g;
    return 0;
}