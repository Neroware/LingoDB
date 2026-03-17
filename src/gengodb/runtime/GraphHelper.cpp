#include "gengodb/runtime/GraphHelper.h"

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
GraphBase* GraphHelper::createBuiltinGraph(lingodb::runtime::VarLen32 graph) {
    const std::unordered_map<std::string, int32_t> builtin {
        {"builtin:default", 0},
        {"builtin:pagerank", 1},
        {"builtin:property-graph", 2},
    };
    auto id_it = builtin.find(graph);
    auto id = id_it == builtin.end() ? 0 : id_it->second;
    switch (id) {
        case 1: {
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
        } break;
        case 2: {
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
        } break;
        default: {
            auto g = SimpleGraph::create(16, 256);
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
    };
}
void GraphHelper::createGraph(lingodb::runtime::VarLen32 meta) {
    
}

} // namespace lingodb::runtime