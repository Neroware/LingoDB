#ifndef GENGODB_RUNTIME_PAGERANK_H
#define GENGODB_RUNTIME_PAGERANK_H

#include "gengodb/runtime/Graph.h"

namespace lingodb::runtime {

struct PageRankGraphStorage {
    struct Data {
        double rank;
        double nextRank;
        int l;
    };
}; // PageRankGraphStorage
// A basic graph without a property table for the presented PageRank algorithm
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

#endif // GENGODB_RUNTIME_PAGERANK_H