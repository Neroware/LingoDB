#ifndef LINGODB_RUNTIME_GRAPH_GRAPHSET_H
#define LINGODB_RUNTIME_GRAPH_GRAPHSET_H

#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/Buffer.h"

namespace lingodb::runtime {

typedef int64_t node_id_t;
typedef int64_t edge_id_t;

class PropertyGraph;
// Represents a set that contains all nodes of a graph
struct GraphNodeSet {
    virtual PropertyGraph* getGraph() = 0;
    virtual BufferIterator* createIterator() = 0;
    virtual void* getNodesBuf() = 0;
    virtual size_t getNodesBufLen() = 0;
    static BufferIterator* nodeSetCreateIterator(GraphNodeSet* nodeSet) { return nodeSet->createIterator(); }
    static PropertyGraph* nodeSetGetGraph(GraphNodeSet* nodeSet) { return nodeSet->getGraph(); }
    static void* nodeSetGetNodesBuf(GraphNodeSet* nodeSet) { return nodeSet->getNodesBuf(); }
    static size_t nodeSetGetNodesBufLen(GraphNodeSet* nodeSet) { return nodeSet->getNodesBufLen(); }
    virtual ~GraphNodeSet() {}
}; // GraphNodeSet
// Represents a set that contains all edges of a graph
struct GraphEdgeSet {
    virtual PropertyGraph* getGraph() = 0;
    virtual BufferIterator* createIterator() = 0;
    virtual void* getEdgesBuf() = 0;
    virtual size_t getEdgesBufLen() = 0;
    static BufferIterator* edgeSetCreateIterator(GraphEdgeSet* edgeSet) { return edgeSet->createIterator(); }
    static PropertyGraph* edgeSetGetGraph(GraphEdgeSet* edgeSet) { return edgeSet->getGraph(); }
    static void* edgeSetGetEdgesBuf(GraphEdgeSet* edgeSet) { return edgeSet->getEdgesBuf(); }
    static size_t edgeSetGetEdgesBufLen(GraphEdgeSet* edgeSet) { return edgeSet->getEdgesBufLen(); }
    virtual ~GraphEdgeSet() {}
}; // GraphEdgeSet
// Represents a set that contains linked edges connected to a node
struct GraphNodeLinkedEdgesSet {
    virtual PropertyGraph* getGraph() = 0;
    virtual edge_id_t getFirstEdge(node_id_t node) = 0;
    virtual void* getEdgesBuf() = 0;
    virtual size_t getEdgesBufLen() = 0;
    static PropertyGraph* edgeSetGetGraph(GraphNodeLinkedEdgesSet* edgeSet) { return edgeSet->getGraph(); }
    static edge_id_t edgeSetGetFirstEdge(GraphNodeLinkedEdgesSet* edgeSet, node_id_t node) { return edgeSet->getFirstEdge(node); }
    static void* edgeSetGetEdgesBuf(GraphNodeLinkedEdgesSet* edgeSet) { return edgeSet->getEdgesBuf(); }
    static size_t edgeSetGetEdgesBufLen(GraphNodeLinkedEdgesSet* edgeSet) { return edgeSet->getEdgesBufLen(); }
    virtual ~GraphNodeLinkedEdgesSet() {}
}; // GraphNodeLinkedEdgesSet

} // namespace lingodb::runtime

#endif // LINGODB_RUNTIME_GRAPH_GRAPHSET_H