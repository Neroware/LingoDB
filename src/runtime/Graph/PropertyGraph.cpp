#include "lingodb/runtime/Graph/PropertyGraph.h"
#include <cassert>

namespace lingodb::runtime {

node_id_t PropertyGraph::getNodeId(NodeEntry* node) const {
    return node - nodes.ptr;
}
PropertyGraph::NodeEntry* PropertyGraph::getNode(node_id_t node) const {
    return nodes.ptr + node;
}
edge_id_t PropertyGraph::getRelationshipId(RelationshipEntry* rel) const {
    return rel - relationships.ptr;
}
PropertyGraph::RelationshipEntry* PropertyGraph::getRelationship(edge_id_t rel) const {
    return relationships.ptr + rel;
}
node_id_t PropertyGraph::addNode() {
    NodeEntry* node;
    if (unusedNodeEntries.empty()) {
        node = nodes.getPtr(nodeBufferSize++);
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
    node->property = 0;
    return nodeId;
}
edge_id_t PropertyGraph::addRelationship(node_id_t from, node_id_t to, relation_type_id_t type) {
    RelationshipEntry* rel;
    NodeEntry *fromNode = getNode(from), *toNode = getNode(to);
    if (unusedRelEntries.empty()) {
        rel = relationships.getPtr(relBufferSize++);
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
node_id_t PropertyGraph::removeNode(node_id_t node) {
    assert(false && "not impelemented"); // TODO implement
}
edge_id_t PropertyGraph::removeRelationship(edge_id_t rel) {
    assert(false && "not impelemented"); // TODO implement
}
PropertyGraph* PropertyGraph::create(size_t initialNodeCapacity, size_t initialRelationshipCapacity, size_t initialPropertyCapacity) {
    return new PropertyGraph(initialNodeCapacity, initialRelationshipCapacity, initialPropertyCapacity);
}
PropertyGraph* PropertyGraph::createTestGraph() {
    auto g = new PropertyGraph(16, 256, 256);
    graphs.insert({(void*) g->nodes.ptr, g});
    graphs.insert({(void*) g->relationships.ptr, g});
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
void PropertyGraph::destroy(PropertyGraph* graph) {
    delete graph;
}
property_id_t PropertyGraph::getPropertyId(PropertyEntry* prop) const {
    return prop - properties.ptr;
}
PropertyGraph::PropertyEntry* PropertyGraph::getProperty(property_id_t prop) const {
    return properties.ptr + prop;
}
property_id_t PropertyGraph::addNodeProperty(node_id_t node, property_key_t key, property_type_id_t type, uint64_t initial_value) {
    PropertyEntry* prop;
    if (unusedPropEntries.empty()) {
        prop = properties.getPtr(propBufferSize++);
    }
    else {
        prop = unusedPropEntries.back();
        unusedPropEntries.pop_back();
    }
    assert(!prop->inUse && "should not happen");
    property_id_t propId = getPropertyId(prop);
    prop->inUse = true;
    prop->id = propId;
    prop->key = key;
    prop->nextProp = prop->prevProp = -1;
    prop->type = type;
    prop->value = initial_value;
    NodeEntry* nodeEntry = getNode(node);
    if (nodeEntry->property >= 0) {
        PropertyEntry* head = getProperty(nodeEntry->property);
        head->prevProp = propId;
        prop->nextProp = head->id;
    }
    nodeEntry->property = propId;
    return propId;
}
property_id_t PropertyGraph::addRelationshipProperty(edge_id_t rel, property_key_t key, property_type_id_t type, uint64_t initial_value) {
    PropertyEntry* prop;
    if (unusedPropEntries.empty()) {
        prop = properties.getPtr(propBufferSize++);
    }
    else {
        prop = unusedPropEntries.back();
        unusedPropEntries.pop_back();
    }
    assert(!prop->inUse && "should not happen");
    property_id_t propId = getPropertyId(prop);
    prop->inUse = true;
    prop->id = propId;
    prop->key = key;
    prop->nextProp = prop->prevProp = -1;
    prop->type = type;
    prop->value = initial_value;
    RelationshipEntry* relEntry = getRelationship(rel);
    if (relEntry->property >= 0) {
        PropertyEntry* head = getProperty(relEntry->property);
        head->prevProp = propId;
        prop->nextProp = head->id;
    }
    relEntry->property = propId;
    return propId;
}
property_id_t PropertyGraph::removeProperty(property_id_t prop) {
    assert(false && "not impelemented"); // TODO implement
}
void PropertyGraph::setProperty(property_id_t prop, uint64_t value) {
    properties.at(prop).value = value;
}

class PropertyGraphNodeTableIterator : public BufferIterator {
    const PropertyGraph& graph;
    bool valid;

    public:
    PropertyGraphNodeTableIterator(const PropertyGraph& graph) 
        : graph(graph), valid(true) {}
    bool isValid() override { return valid; }
    void next() override { valid = false; }
    Buffer getCurrentBuffer() override { return graph.getNodeBuffer(); }
    void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void* contextPtr) override {
        // TODO No parallelism in PropertyGraph iterators yet...
        auto buffer = getCurrentBuffer();
        forEachChunk(buffer, contextPtr);
    }
}; // PropertyGraphNodeTableIterator
class PropertyGraphEdgeTableIterator : public BufferIterator {
    const PropertyGraph& graph;
    bool valid;

    public:
    PropertyGraphEdgeTableIterator(const PropertyGraph& graph) 
        : graph(graph), valid(true) {}
    bool isValid() override { return valid; }
    void next() override { valid = false; }
    Buffer getCurrentBuffer() override { return graph.getEdgeBuffer(); }
    void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void* contextPtr) override {
        // TODO No parallelism in PropertyGraph iterators yet...
        auto buffer = getCurrentBuffer();
        forEachChunk(buffer, contextPtr);
    }
}; // PropertyGraphEdgeTableIterator
class PropertyGraphPropertyTableIterator : public BufferIterator {
    const PropertyGraph& graph;
    bool valid;

    public:
    PropertyGraphPropertyTableIterator(const PropertyGraph& graph) 
        : graph(graph), valid(true) {}
    bool isValid() override { return valid; }
    void next() override { valid = false; }
    Buffer getCurrentBuffer() override { return graph.getPropBuffer(); }
    void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void* contextPtr) override {
        // TODO No parallelism in PropertyGraph iterators yet...
        auto buffer = getCurrentBuffer();
        forEachChunk(buffer, contextPtr);
    }
}; // PropertyGraphPropertyTableIterator
BufferIterator* PropertyGraph::createNodeIterator() {
    return new PropertyGraphNodeTableIterator(*this);
}
BufferIterator* PropertyGraph::createEdgeIterator() {
    return new PropertyGraphEdgeTableIterator(*this);
}
BufferIterator* PropertyGraph::createPropIterator() {
    return new PropertyGraphPropertyTableIterator(*this);
}
void* PropertyGraph::getLinkedEgdesLListHead(void* ref) const {
    NodeEntry* node = (NodeEntry*) ref;
    return (void*) getRelationship(node->nextRelationship);
}
PropertyGraph* PropertyGraph::getGraphByNodeRef(void* ref) {
    NodeEntry* node = (NodeEntry*) ref;
    return PropertyGraph::graphs[(void*) (node - node->id)];
}
PropertyGraph* PropertyGraph::getGraphByEdgeRef(void* ref) {
    RelationshipEntry* rel = (RelationshipEntry*) ref;
    return PropertyGraph::graphs[(void*) (rel - rel->id)];
}
std::unordered_map<void*, PropertyGraph*> PropertyGraph::graphs;

} // lingodb::runtime::graph