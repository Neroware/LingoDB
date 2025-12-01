#include "lingodb/runtime/Graph/Graph.h"

namespace lingodb::runtime {

class GraphNodeTableIterator : public BufferIterator {
    const GraphBase& graph;
    bool valid;

    public:
    GraphNodeTableIterator(const GraphBase& graph) 
        : graph(graph), valid(true) {}
    bool isValid() override { return valid; }
    void next() override { valid = false; }
    Buffer getCurrentBuffer() override { return Buffer{graph.nodeBufferLen, graph.nodeBufferPtr}; };
    void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void* contextPtr) override {
        // TODO No parallelism in graph iterators yet...
        auto buffer = getCurrentBuffer();
        forEachChunk(buffer, contextPtr);
    }
}; // GraphNodeTableIterator
class GraphEdgeTableIterator : public BufferIterator {
    const GraphBase& graph;
    bool valid;

    public:
    GraphEdgeTableIterator(const GraphBase& graph) 
        : graph(graph), valid(true) {}
    bool isValid() override { return valid; }
    void next() override { valid = false; }
    Buffer getCurrentBuffer() override { return Buffer{graph.relBufferLen, graph.relBufferPtr}; }
    void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void* contextPtr) override {
        // TODO No parallelism in graph iterators yet...
        auto buffer = getCurrentBuffer();
        forEachChunk(buffer, contextPtr);
    }
}; // GraphEdgeTableIterator
BufferIterator* GraphHelper::createNodeIterator(GraphBase* graph) {
    return new GraphNodeTableIterator(*graph);
}
BufferIterator* GraphHelper::createEdgeIterator(GraphBase* graph) {
    return new GraphEdgeTableIterator(*graph);
}
void* GraphHelper::getLinkedEgdesLListHeadOf(GraphBase* graph, uint8_t* ref, size_t refSize) {
    auto node = (NodeEntryBase*) ref;
    return graph->relBufferPtr + node->nextRelationship * refSize;
}
GraphBase* GraphHelper::getGraphByNodeRef(uint8_t* ref, size_t refSize) {
    auto node = (NodeEntryBase*) ref;
    return GraphHelper::graphs[(ref - node->id * refSize)];
}
GraphBase* GraphHelper::getGraphByEdgeRef(uint8_t* ref, size_t refSize) {
    auto rel = (RelationshipEntryBase*) ref;
    return GraphHelper::graphs[(ref - rel->id * refSize)];
}
TestGraph* TestGraph::create(size_t initialNodeCapacity, size_t initialRelationshipCapacity) {
    return new TestGraph(initialNodeCapacity, initialRelationshipCapacity);
}
TestGraph* TestGraph::createTestGraph() {
    auto g = new TestGraph(16, 256);
    GraphHelper::graphs.insert({g->nodeBufferPtr, g});
    GraphHelper::graphs.insert({g->relBufferPtr, g});
    for (int i = 0; i < 6; i++) {
        g->addNode();
    }
    relation_type_id_t relType = 0;
    g->addRelationship(0, 2, relType);
    g->addRelationship(1, 0, relType);
    g->addRelationship(1, 2, relType);
    g->addRelationship(1, 4, relType);
    g->addRelationship(2, 4, relType);
    g->addRelationship(2, 3, relType);
    g->getRelationship(0)->property = 4242;
    g->getRelationship(1)->property = 111;
    g->getRelationship(2)->property = 222;
    g->getRelationship(3)->property = 333;
    g->getRelationship(4)->property = 444;
    g->getRelationship(5)->property = 555;
    g->getNode(0)->property = 42;
    g->getNode(1)->property = 11;
    g->getNode(2)->property = 22;
    g->getNode(3)->property = 33;
    g->getNode(4)->property = 44;
    g->getNode(5)->property = 55;
    return g;
}

std::unordered_map<uint8_t*, GraphBase*> GraphHelper::graphs;

} // lingodb::runtime