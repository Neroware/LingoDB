#include "gengodb/runtime/BuiltinGraphs.h"

namespace lingodb::runtime {

PageRankGraph* PageRankGraph::create(size_t initialNodeCapacity, size_t initialRelationshipCapacity) { 
    PageRankGraph* g = new PageRankGraph(initialNodeCapacity, initialRelationshipCapacity);
    GraphStorageHelper::addGraph(g, initialNodeCapacity, initialRelationshipCapacity, 1);
    return g;
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

}