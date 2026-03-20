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
    if (!loaded) {
        loaded = true;
        if (nodeCounter > 0 || fileName.empty() || dbDir.empty()) {
            return;
        }
        if (!std::filesystem::exists(dbDir + "/" + fileName)) {
            return;
        }
        loadGraph(this, dbDir + "/" + fileName);
    }
}
void GengoDBGraph::serialize(lingodb::utility::Serializer& serializer) const {
    serializer.writeProperty<std::string>(1, fileName);
}
std::unique_ptr<GengoDBGraph> GengoDBGraph::deserialize(lingodb::utility::Deserializer& deserializer) {
    auto fileName = deserializer.readProperty<std::string>(1);
    return GengoDBGraph::create(fileName);
}
std::unique_ptr<GengoDBGraph> GengoDBGraph::create(std::string name) {
    auto g = std::make_unique<GengoDBGraph>(name);
    GraphStorageHelper::addGraph(g.get(), GengoDBGraph::DEFAULT_CAPACITY, GengoDBGraph::DEFAULT_CAPACITY, GengoDBGraph::DEFAULT_CAPACITY);
    return g;
}

} // namespace lingodb::runtime