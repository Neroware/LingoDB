#include "gengodb/runtime/Graph.h"

#include "gengodb/runtime/PropertyGraph.h"

namespace lingodb::runtime {

node_id_t SimpleGraph::getNodeId(NodeEntry* node) const {
    return node - nodes.ptr;
}
SimpleGraph::NodeEntry* SimpleGraph::getNode(node_id_t node) const {
    return nodes.ptr + node;
}
relation_id_t SimpleGraph::getRelationshipId(SimpleGraph::RelationshipEntry* rel) const {
    return rel - relationships.ptr;
}
SimpleGraph::RelationshipEntry* SimpleGraph::getRelationship(relation_id_t rel) const {
    return relationships.ptr + rel;
}
node_id_t SimpleGraph::addNode() {
    NodeEntry* node;
    if (unusedNodeEntries.empty()) {
        node = nodes.getPtr(nodeCounter++);
    }
    else {
        node = unusedNodeEntries.back();
        unusedNodeEntries.pop_back();
    }
    assert(!node->inUse && "should not happen");
    node_id_t nodeId = getNodeId(node);
    node->inUse = true;
    node->nextRelId = -1;
    node->nextPropId = 0;
    return nodeId;
}
relation_id_t SimpleGraph::addRelationship(node_id_t from, node_id_t to) {
    RelationshipEntry* rel;
    NodeEntry *fromNode = getNode(from), *toNode = getNode(to);
    if (unusedRelEntries.empty()) {
        rel = relationships.getPtr(relCounter++);
    }
    else {
        rel = unusedRelEntries.back();
        unusedRelEntries.pop_back();
    }
    assert(!rel->inUse && "should not happen");
    relation_id_t relId = getRelationshipId(rel);
    rel->inUse = true;
    rel->firstNode = from;
    rel->secondNode = to;
    rel->relationshipType = 0;
    rel->firstNextRelId = rel->firstPrevRelId = rel->secondNextRelId = rel->secondPrevRelId = -1;
    rel->nextPropId = 0;
    rel->firstInChainMarker = true;
    if (fromNode->nextRelId >= 0) {
        RelationshipEntry* head = getRelationship(fromNode->nextRelId);
        head->firstInChainMarker = false;
        if (head->firstNode == from) {
            head->firstPrevRelId = relId;
            rel->firstNextRelId = fromNode->nextRelId;
        }
        else {
            head->secondPrevRelId = relId;
            rel->firstNextRelId = fromNode->nextRelId;
        }
    }
    fromNode->nextRelId = relId;
    if (from != to) {
        if (toNode->nextRelId >= 0) {
            RelationshipEntry* head = getRelationship(toNode->nextRelId);
            head->firstInChainMarker = false;
            if (head->firstNode == to) {
                head->firstPrevRelId = relId;
                rel->secondNextRelId = toNode->nextRelId;   
            }
            else {
                head->secondPrevRelId = relId;
                rel->secondNextRelId = toNode->nextRelId;
            }
        }
        toNode->nextRelId = relId;
    }
    return relId;
}
node_id_t SimpleGraph::removeNode(node_id_t node) {
    assert(false && "not impelemented"); // TODO implement
}
relation_id_t SimpleGraph::removeRelationship(relation_id_t rel) {
    assert(false && "not impelemented"); // TODO implement
}
void SimpleGraph::setNodeValue(node_id_t node, uint64_t value) const { 
    getNode(node)->nextPropId = value;
}
void SimpleGraph::setRelationshipValue(relation_id_t edge, uint64_t value) const { 
    getRelationship(edge)->nextPropId = value; 
}
uint64_t SimpleGraph::getNodeValue(node_id_t node) const { 
    return getNode(node)->nextPropId; 
}
uint64_t SimpleGraph::getRelationshipValue(relation_id_t edge) const { 
    return getRelationship(edge)->nextPropId; 
}
SimpleGraph* SimpleGraph::create(size_t initialNodeCapacity, size_t initialRelationshipCapacity) { 
    SimpleGraph* g = new SimpleGraph(initialNodeCapacity, initialRelationshipCapacity);
    GraphStorageHelper::addGraph(g, initialNodeCapacity, initialRelationshipCapacity, 1);
    return g;
}
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