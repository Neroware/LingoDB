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
LingoDBGraph* LingoDBGraph::create(size_t initialNodeCapacity, size_t initialRelationshipCapacity) {
    return new LingoDBGraph(initialNodeCapacity, initialRelationshipCapacity);
}
GraphBase* GraphHelper::createTestGraph() {
    auto g = LingoDBGraph::create(16, 256);
    GraphHelper::graphs.insert({g->nodeBufferPtr, g});
    GraphHelper::graphs.insert({g->relBufferPtr, g});
    for (int i = 0; i < 6; i++) {
        g->addNode();
    }
    g->addEdge(0, 2);
    g->addEdge(1, 0);
    g->addEdge(1, 2);
    g->addEdge(1, 4);
    g->addEdge(2, 4);
    g->addEdge(2, 3);
    g->setEdgeValue(0, 4242);
    g->setEdgeValue(1, 111);
    g->setEdgeValue(2, 222);
    g->setEdgeValue(3, 333);
    g->setEdgeValue(4, 444);
    g->setEdgeValue(5, 555);
    g->setNodeValue(0, 42);
    g->setNodeValue(1, 11);
    g->setNodeValue(2, 22);
    g->setNodeValue(3, 33);
    g->setNodeValue(4, 44);
    g->setNodeValue(5, 55);
    return g;
}

std::unordered_map<uint8_t*, GraphBase*> GraphHelper::graphs;

} // lingodb::runtime