#include "gengodb/GraphCatalogEntry.h"

namespace gengodb::catalog {

void RDFGraphCatalogEntry::serializeEntry(lingodb::utility::Serializer& serializer) const {
    // TODO implement
    assert(false && "not implemented");
}
std::shared_ptr<RDFGraphCatalogEntry> RDFGraphCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
    // TODO implement
    assert(false && "not implemented");
    return nullptr;
}
IRI RDFGraphCatalogEntry::getNodeIri(int32_t node) const {
    return impl->nodes.get_iri(node);
}
IRI RDFGraphCatalogEntry::getRelationIri(int32_t rel) const {
    return impl->relations.get_iri(rel);
}
std::string_view RDFGraphCatalogEntry::getLocalId(int32_t node) const {
    for (const auto& pair : impl->bnodes) {
        if (pair.second == node) 
            return pair.first;
    }
    return "";
}
lingodb::runtime::PropertyGraph& RDFGraphCatalogEntry::getStorage() {
    return *(impl->storage);
}
void RDFGraphCatalogEntry::flush() {
    impl->storage->flush();
}    
void RDFGraphCatalogEntry::ensureFullyLoaded() {
    impl->storage->ensureLoaded();
}
void RDFGraphCatalogEntry::setShouldPersist(bool shouldPersist) {
    impl->storage->setPersist(shouldPersist);
}
void RDFGraphCatalogEntry::setDBDir(std::string dbDir) {
    impl->storage->setDBDir(dbDir);
}
std::shared_ptr<RDFGraphCatalogEntry> RDFGraphCatalogEntry::createFromCreateRdfGraphDef(const CreateRdfGraphDef& def) {
    // TODO Create RdfGraph with GengoDBGraph storage in main memory, load from RDF triple file if no .dat-file is present. 
    assert(false && "not implemented");
    return nullptr;
}

} // namespace gengodb::catalog