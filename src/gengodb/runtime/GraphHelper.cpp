#include "gengodb/runtime/GraphHelper.h"

#include "gengodb/runtime/BuiltinGraphs.h"

namespace lingodb::runtime {

GraphBase* GraphHelper::allocAndPopulateBuiltinGraph(int32_t builtin) {
    auto* context = getCurrentExecutionContext();
    assert(context);
    switch (builtin) {
        case 1: {
            auto g = PageRankGraph::create(16, 256);
            context->registerState({(static_cast<void*>(g)), [](void* p) { delete (static_cast<PageRankGraph*>(p)); }});
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
            context->registerState({(static_cast<void*>(g)), [](void* p) { delete (static_cast<PropertyGraph*>(p)); }});
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
            context->registerState({(static_cast<void*>(g)), [](void* p) { delete (static_cast<SimpleGraph*>(p)); }});
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
GraphBase* GraphHelper::allocGraphState(size_t nodeBufLen, size_t relBufLen, size_t propBufLen) {
    auto* context = getCurrentExecutionContext();
    assert(context);
    auto ptr = static_cast<void*>(PropertyGraph::create(nodeBufLen, relBufLen, propBufLen));
    context->registerState({ptr, [](void* p) { delete (static_cast<PropertyGraph*>(p)); }});
    return static_cast<GraphBase*>(ptr);
}
void GraphHelper::createGraph(lingodb::runtime::VarLen32 meta) {
    
}

} // namespace lingodb::runtime