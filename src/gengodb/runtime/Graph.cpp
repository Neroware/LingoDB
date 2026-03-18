#include "gengodb/runtime/Graph.h"

namespace lingodb::runtime {
std::vector<GraphStorageHelper::GraphStorageData> GraphStorageHelper::graphs;
class GraphTableIterator : public BufferIterator {
private:
    size_t entrySize;
    size_t entryCount;
    uint8_t* ptr;
    bool valid;
public:
    GraphTableIterator(size_t entrySize, size_t entryCount, uint8_t* ptr) 
        : entrySize(entrySize), entryCount(entryCount), ptr(ptr), valid(true) {}
    bool isValid() override { return valid; }
    void next() override { valid = false; }
    Buffer getCurrentBuffer() override { return Buffer{entrySize * entryCount, ptr}; };
    void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void* contextPtr) override {
        // TODO No parallelism in graph iterators yet...
        auto buffer = getCurrentBuffer();
        forEachChunk(buffer, contextPtr);
    }
}; // GraphTableIterator
uint8_t* GraphStorageHelper::getNodeBufferPtr(uint8_t* ref) {
    return getGraphInfo(ref).nodeBufferPtr;
}
uint8_t* GraphStorageHelper::getRelBufferPtr(uint8_t* ref) {
    return getGraphInfo(ref).relBufferPtr;
}
uint8_t* GraphStorageHelper::getPropBufferPtr(uint8_t* ref) {
    return getGraphInfo(ref).propBufferPtr;
}
size_t GraphStorageHelper::getNodeBufferLen(uint8_t* ref) {
    const auto& graph = getGraphInfo(ref);
    return graph.graphPtr->nodeCounter * graph.nodeEntrySize;
}
size_t GraphStorageHelper::getRelBufferLen(uint8_t* ref) {
    const auto& graph = getGraphInfo(ref);
    return graph.graphPtr->relCounter * graph.relEntrySize;
}
size_t GraphStorageHelper::getPropBufferLen(uint8_t* ref) {
    const auto& graph = getGraphInfo(ref);
    return graph.graphPtr->propCounter * graph.propEntrySize;
}
BufferIterator* GraphStorageHelper::createNodeIterator(uint8_t* ref) {
    GraphStorageData graphData = getGraphInfo(ref);
    return new GraphTableIterator(graphData.nodeEntrySize, graphData.graphPtr->nodeCounter, graphData.nodeBufferPtr);
}
BufferIterator* GraphStorageHelper::createRelIterator(uint8_t* ref) {
    GraphStorageData graphData = getGraphInfo(ref);
    return new GraphTableIterator(graphData.relEntrySize, graphData.graphPtr->relCounter, graphData.relBufferPtr);
}
BufferIterator* GraphStorageHelper::createPropIterator(uint8_t* ref) {
    GraphStorageData graphData = getGraphInfo(ref);
    return new GraphTableIterator(graphData.propEntrySize, graphData.graphPtr->propCounter, graphData.propBufferPtr);
}
node_id_t GraphStorageHelper::getNodeId(uint8_t* node) {
    const auto& graphData = getGraphInfo(node);
    return (node_id_t) ((node - graphData.nodeBufferPtr) / graphData.nodeEntrySize);
}
relation_id_t GraphStorageHelper::getRelationshipId(uint8_t* rel) {
    const auto& graphData = getGraphInfo(rel);
    return (relation_id_t) ((rel - graphData.relBufferPtr) / graphData.relEntrySize);
}
property_id_t GraphStorageHelper::getPropId(uint8_t* prop) {
    const auto& graphData = getGraphInfo(prop);
    return (property_id_t) ((prop - graphData.propBufferPtr) / graphData.propEntrySize);
}
uint8_t* GraphStorageHelper::getRelationshipLListHeadOf(uint8_t* ref) {
    GraphStorageData graphData = getGraphInfo(ref);
    auto node = (GraphBase::NodeEntry*) ref;
    return graphData.relBufferPtr + node->nextRelId * graphData.relEntrySize;
}

} // namespace lingodb::runtime