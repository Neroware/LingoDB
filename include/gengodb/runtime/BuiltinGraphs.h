#ifndef GENGODB_RUNTIME_BUILTINGRAPHS_H
#define GENGODB_RUNTIME_BUILTINGRAPHS_H

#include "gengodb/runtime/Graph.h"

namespace lingodb::runtime {

// A builtin graph type without a property table using i64 as property value
class SimpleGraph : public Graph<uint64_t, uint64_t> {
private:
    node_id_t getNodeId(NodeEntry* node) const;
    NodeEntry* getNode(node_id_t node) const;
    relation_id_t getRelationshipId(RelationshipEntry* rel) const;
    RelationshipEntry* getRelationship(relation_id_t rel) const;
    SimpleGraph(size_t maxNodeCapacity, size_t maxRelCapacity)
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
    static SimpleGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity);
    static void destroy(SimpleGraph* graph) { delete graph; }
}; // SimpleGraph
struct PageRankGraphStorage {
    struct Data {
        double rank;
        double nextRank;
        int l;
    };
}; // PageRankGraphStorage
// A builtin graph type without a property table for the presented PageRank algorithm
class PageRankGraph : public PageRankGraphStorage, public Graph<PageRankGraphStorage::Data, void*> {
private:
    node_id_t getNodeId(NodeEntry* node) const;
    NodeEntry* getNode(node_id_t node) const;
    relation_id_t getRelationshipId(RelationshipEntry* rel) const;
    RelationshipEntry* getRelationship(relation_id_t rel) const;
    PageRankGraph(size_t maxNodeCapacity, size_t maxRelCapacity)
        : Graph(maxNodeCapacity, maxRelCapacity, 1) {}
public:
    node_id_t addNode();
    relation_id_t addRelationship(node_id_t from, node_id_t to);
    size_t getNodeCount() const { return nodeCounter; }
    size_t getRelCount() const { return relCounter; }
    static PageRankGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity);
    static void destroy(PageRankGraph* graph) { delete graph; }
}; // PageRankGraph

} // namespace lingodb::runtime

#endif // GENGODB_RUNTIME_BUILTINGRAPHS_H