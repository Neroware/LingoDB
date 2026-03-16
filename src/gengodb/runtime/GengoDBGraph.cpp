#include "gengodb/runtime/GengoDBGraph.h"

namespace lingodb::runtime {

void GengoDBGraph::flush() {
    // TODO implement
    assert(false && "not implemented");
}
void GengoDBGraph::ensureLoaded() {
    if (loaded) return;
    // TODO implement
    assert(false && "not implemented");
}
void GengoDBGraph::serialize(lingodb::utility::Serializer& serializer) const {
    // TODO implement
    assert(false && "not implemented");
}
std::unique_ptr<GengoDBGraph> GengoDBGraph::deserialize(lingodb::utility::Deserializer& deserializer) {
    // TODO implement
    assert(false && "not implemented");
    return nullptr;
}
std::unique_ptr<GengoDBGraph> GengoDBGraph::create(const gengodb::catalog::CreateRdfGraphDef& def) {
    auto g = std::make_unique<GengoDBGraph>(def.path + ".dat");
    GraphStorageHelper::addGraph(g.get(), GengoDBGraph::DEFAULT_CAPACITY, GengoDBGraph::DEFAULT_CAPACITY, GengoDBGraph::DEFAULT_CAPACITY);
    return g;
}

} // namespace lingodb::runtime