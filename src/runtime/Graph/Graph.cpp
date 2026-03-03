#include "lingodb/runtime/Graph/Graph.h"

#include "lingodb/runtime/Graph/PageRank.h"
#include "lingodb/runtime/Graph/PropertyGraph.h"

namespace lingodb::runtime {

node_id_t GengoDBGraph::getNodeId(NodeEntry* node) const {
    return node - nodes.ptr;
}
GengoDBGraph::NodeEntry* GengoDBGraph::getNode(node_id_t node) const {
    return nodes.ptr + node;
}
relation_id_t GengoDBGraph::getRelationshipId(GengoDBGraph::RelationshipEntry* rel) const {
    return rel - relationships.ptr;
}
GengoDBGraph::RelationshipEntry* GengoDBGraph::getRelationship(relation_id_t rel) const {
    return relationships.ptr + rel;
}
node_id_t GengoDBGraph::addNode() {
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
relation_id_t GengoDBGraph::addRelationship(node_id_t from, node_id_t to) {
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
node_id_t GengoDBGraph::removeNode(node_id_t node) {
    assert(false && "not impelemented"); // TODO implement
}
relation_id_t GengoDBGraph::removeRelationship(relation_id_t rel) {
    assert(false && "not impelemented"); // TODO implement
}
void GengoDBGraph::setNodeValue(node_id_t node, uint64_t value) const { 
    getNode(node)->nextPropId = value;
}
void GengoDBGraph::setRelationshipValue(relation_id_t edge, uint64_t value) const { 
    getRelationship(edge)->nextPropId = value; 
}
uint64_t GengoDBGraph::getNodeValue(node_id_t node) const { 
    return getNode(node)->nextPropId; 
}
uint64_t GengoDBGraph::getRelationshipValue(relation_id_t edge) const { 
    return getRelationship(edge)->nextPropId; 
}
GengoDBGraph* GengoDBGraph::create(size_t initialNodeCapacity, size_t initialRelationshipCapacity) { 
    GengoDBGraph* g = new GengoDBGraph(initialNodeCapacity, initialRelationshipCapacity);
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
node_id_t PageRankGraph::getNodeId(NodeEntry* node) const {
    return node - nodes.ptr;
}
PageRankGraph::NodeEntry* PageRankGraph::getNode(node_id_t node) const {
    return nodes.ptr + node;
}
relation_id_t PageRankGraph::getRelationshipId(PageRankGraph::RelationshipEntry* rel) const {
    return rel - relationships.ptr;
}
PageRankGraph::RelationshipEntry* PageRankGraph::getRelationship(relation_id_t rel) const {
    return relationships.ptr + rel;
}
node_id_t PageRankGraph::addNode() {
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
    node->nextPropId = Data {0.0, 0.0, 0};
    return nodeId;
}
relation_id_t PageRankGraph::addRelationship(node_id_t from, node_id_t to) {
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
PageRankGraph* PageRankGraph::create(size_t initialNodeCapacity, size_t initialRelationshipCapacity) { 
    PageRankGraph* g = new PageRankGraph(initialNodeCapacity, initialRelationshipCapacity);
    GraphStorageHelper::addGraph(g, initialNodeCapacity, initialRelationshipCapacity, 1);
    return g;
}

GraphBase* createDefaultTestGraph() {
    auto g = GengoDBGraph::create(16, 256);
    for (int i = 0; i < 6; i++) {
        g->addNode();
    }
    g->addRelationship(0, 2);
    g->addRelationship(1, 0);
    g->addRelationship(1, 2);
    g->addRelationship(1, 4);
    g->addRelationship(2, 4);
    g->addRelationship(2, 3);
    g->setRelationshipValue(0, 4242);
    g->setRelationshipValue(1, 111);
    g->setRelationshipValue(2, 222);
    g->setRelationshipValue(3, 333);
    g->setRelationshipValue(4, 444);
    g->setRelationshipValue(5, 555);
    g->setNodeValue(0, 42);
    g->setNodeValue(1, 11);
    g->setNodeValue(2, 22);
    g->setNodeValue(3, 33);
    g->setNodeValue(4, 44);
    g->setNodeValue(5, 55);
    return g;
}
PageRankGraph* createMichaelsPageRankGraph() {
    auto g = PageRankGraph::create(16, 256);
    for (int i = 0; i < 5; i++) {
        g->addNode();
    }
    g->addRelationship(0, 1);
    g->addRelationship(1, 2);
    g->addRelationship(2, 4);
    g->addRelationship(3, 4);
    g->addRelationship(4, 1);
    g->addRelationship(0, 3);
    return g;
}
PropertyGraph* createDefaultTestPropertyGraph() {
    auto g = PropertyGraph::create(16, 256, 256);
    for (int i = 0; i < 6; i++) {
        g->addNode();
    }
    g->addRelationship(0, 2, 0);
    g->addRelationship(1, 0, 0);
    g->addRelationship(1, 2, 0);
    g->addRelationship(1, 4, 0);
    g->addRelationship(2, 4, 0);
    g->addRelationship(2, 3, 0);
    g->addNodeProperty(0, 0, 0, 424);
    g->addNodeProperty(1, 11, 11, 100);
    g->addNodeProperty(1, 11, 11, 101);
    g->addNodeProperty(1, 11, 11, 102);
    g->addNodeProperty(2, 22, 22, 200);
    g->addNodeProperty(3, 33, 33, 300);
    g->addNodeProperty(5, 55, 55, 501);
    g->addNodeProperty(5, 55, 55, 502);
    g->addRelationshipProperty(0, 0, 0, 4242);
    g->addRelationshipProperty(1, 11, 11, 1000);
    g->addRelationshipProperty(2, 22, 22, 2000);
    g->addRelationshipProperty(3, 33, 33, 3000);
    return g;
}
GraphBase* GraphStorageHelper::createTestGraph(uint64_t whichOne) {
    GraphBase* g;
    switch (whichOne) {
        case 0: g = createDefaultTestGraph();
            break;
        case 1: g = createMichaelsPageRankGraph();
            break;
        case 2: g = createDefaultTestPropertyGraph();
            break;
        default: g = createDefaultTestGraph();
            break;
    }
    return g;
}

} // namespace lingodb::runtime