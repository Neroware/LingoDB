#include "gengodb/runtime/GengoDBGraph.h"

#include <filesystem>

namespace lingodb::runtime {

void loadGraph(PropertyGraph* storage, std::string path) {
    assert(false && "not implemented");
}
void GengoDBGraph::flush() {
    // TODO implement
    assert(false && "not implemented");
}
void GengoDBGraph::ensureLoaded() {
    if (nodeCounter > 0) {
        loaded = true;
        return;
    }
    if (!loaded) {
        loaded = true;
        if (fileName.empty() || dbDir.empty()) {
            return;
        }
        if (!std::filesystem::exists(dbDir + "/" + fileName)) {
            return;
        }
        loadGraph(this, dbDir + "/" + fileName);
    }
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
std::unique_ptr<GengoDBGraph> GengoDBGraph::create(std::string name) {
    auto g = std::make_unique<GengoDBGraph>(name + ".dat");
    GraphStorageHelper::addGraph(g.get(), GengoDBGraph::DEFAULT_CAPACITY, GengoDBGraph::DEFAULT_CAPACITY, GengoDBGraph::DEFAULT_CAPACITY);
    return g;
}

} // namespace lingodb::runtime