#include "gengodb/runtime/PropertyGraph.h"

namespace lingodb::runtime {

node_id_t PropertyGraph::getNodeId(NodeEntry* node) const {
    return node - nodes.ptr;
}
PropertyGraph::NodeEntry* PropertyGraph::getNode(node_id_t node) const {
    return nodes.ptr + node;
}
relation_id_t PropertyGraph::getRelationshipId(PropertyGraph::RelationshipEntry* rel) const {
    return rel - relationships.ptr;
}
PropertyGraph::RelationshipEntry* PropertyGraph::getRelationship(relation_id_t rel) const {
    return relationships.ptr + rel;
}
property_id_t PropertyGraph::getPropertyId(PropertyEntry* prop) const {
    return prop - properties.ptr;
}
PropertyGraph::PropertyEntry* PropertyGraph::getProperty(property_id_t prop) const {
    return properties.ptr + prop;
}
property_id_t PropertyGraph::removeProperty(property_id_t prop) {
    assert(false && "not impelemented"); // TODO implement
}
void PropertyGraph::setProperty(property_id_t prop, property_t value) {
    properties.at(prop).value = value;
}
node_id_t PropertyGraph::addNode() {
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
    node->nextPropId = -1;
    return nodeId;
}
relation_id_t PropertyGraph::addRelationship(node_id_t from, node_id_t to, relation_type_id_t type) {
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
    rel->relationshipType = type;
    rel->firstNextRelId = rel->firstPrevRelId = rel->secondNextRelId = rel->secondPrevRelId = -1;
    rel->nextPropId = -1;
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
node_id_t PropertyGraph::removeNode(node_id_t node) {
    assert(false && "not impelemented"); // TODO implement
}
relation_id_t PropertyGraph::removeRelationship(relation_id_t rel) {
    assert(false && "not impelemented"); // TODO implement
}
property_id_t PropertyGraph::addNodeProperty(node_id_t node, property_key_t key, property_type_id_t type, property_t initial_value) {
    PropertyEntry* prop;
    if (unusedPropEntries.empty()) {
        prop = properties.getPtr(propCounter++);
    }
    else {
        prop = unusedPropEntries.back();
        unusedPropEntries.pop_back();
    }
    assert(!prop->inUse && "should not happen");
    property_id_t propId = getPropertyId(prop);
    prop->inUse = true;
    prop->key = key;
    prop->nextProp = prop->prevProp = -1;
    prop->type = type;
    prop->value = initial_value;
    NodeEntry* nodeEntry = getNode(node);
    if (nodeEntry->nextPropId >= 0) {
        PropertyEntry* head = getProperty(nodeEntry->nextPropId);
        head->prevProp = propId;
        prop->nextProp = nodeEntry->nextPropId;
    }
    nodeEntry->nextPropId = propId;
    return propId;
}
property_id_t PropertyGraph::addRelationshipProperty(relation_id_t rel, property_key_t key, property_type_id_t type, property_t initial_value) {
    PropertyEntry* prop;
    if (unusedPropEntries.empty()) {
        prop = properties.getPtr(propCounter++);
    }
    else {
        prop = unusedPropEntries.back();
        unusedPropEntries.pop_back();
    }
    assert(!prop->inUse && "should not happen");
    property_id_t propId = getPropertyId(prop);
    prop->inUse = true;
    prop->key = key;
    prop->nextProp = prop->prevProp = -1;
    prop->type = type;
    prop->value = initial_value;
    RelationshipEntry* relEntry = getRelationship(rel);
    if (relEntry->nextPropId >= 0) {
        PropertyEntry* head = getProperty(relEntry->nextPropId);
        head->prevProp = propId;
        prop->nextProp = relEntry->nextPropId;
    }
    relEntry->nextPropId = propId;
    return propId;
}
PropertyGraph* PropertyGraph::create(size_t initialNodeCapacity, size_t initialRelationshipCapacity, size_t initialPropertyCapacity) { 
    PropertyGraph* g = new PropertyGraph(initialNodeCapacity, initialRelationshipCapacity, initialPropertyCapacity);
    GraphStorageHelper::addGraph(g, initialNodeCapacity, initialRelationshipCapacity, initialPropertyCapacity);
    return g;
}

uint8_t* PropertyGraphStorageHelper::getNodePropertyLListHeadOf(uint8_t* ref) {
    const auto& graphData = PropertyGraphStorageHelper::getGraphInfo(ref);
    auto node = (PropertyGraph::NodeEntry*) ref;
    return graphData.propBufferPtr + node->nextPropId * graphData.propEntrySize;
}
uint8_t* PropertyGraphStorageHelper::getRelPropertyLListHeadOf(uint8_t* ref) {
    const auto& graphData = PropertyGraphStorageHelper::getGraphInfo(ref);
    auto rel = (PropertyGraph::RelationshipEntry*) ref;
    return graphData.propBufferPtr + rel->nextPropId * graphData.propEntrySize;
}

} // END namespace lingodb::runtime