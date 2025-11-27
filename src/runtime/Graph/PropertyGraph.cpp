#include "lingodb/runtime/Graph/PropertyGraph.h"
#include <cassert>

namespace lingodb::runtime {

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

PropertyGraph* PropertyGraph::create(size_t initialNodeCapacity, size_t initialRelationshipCapacity, size_t initialPropertyCapacity) {
    return new PropertyGraph(initialNodeCapacity, initialRelationshipCapacity, initialPropertyCapacity);
}
PropertyGraph* PropertyGraph::createTestGraph() {
    assert(false && "not implemented"); // TODO implement
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

} // lingodb::runtime