#ifndef GENGODB_GRAPHCATALOGENTRY_H
#define GENGODB_GRAPHCATALOGENTRY_H

#include "lingodb/catalog/Catalog.h"
#include "gengodb/RdfGraph.h"
#include "gengodb/CreateRdfGraphDef.h"

#include <rdf4cpp.hpp>

namespace gengodb::semantics {
struct RdfGraph;
} // namespace gengodb::semantics

namespace gengodb::catalog {
using namespace lingodb::catalog;
using namespace rdf4cpp;
class GraphCatalogEntry : public CatalogEntry {
protected:
    std::string name;
public:
    static constexpr std::array<CatalogEntryType, 1> entryTypes = {CatalogEntryType::GENGODB_GRAPH_ENTRY};
    GraphCatalogEntry(CatalogEntryType entryType, std::string name) : CatalogEntry(entryType), name(name) {}
    std::string getName() override { return name; }
    size_t nodeCount() { return getStorage().nodeCounter; }
    size_t edgeCount() { return getStorage().relCounter; }
    size_t propertyCount() { return getStorage().propCounter; }
    virtual lingodb::runtime::PropertyGraph& getStorage() = 0;
};

class RDFGraphCatalogEntry : public GraphCatalogEntry {
    std::unique_ptr<semantics::RdfGraph> impl;

    public:
    RDFGraphCatalogEntry(std::unique_ptr<semantics::RdfGraph> impl);

    static constexpr std::array<CatalogEntryType, 1> entryTypes = {CatalogEntryType::GENGODB_GRAPH_ENTRY};
    void serializeEntry(lingodb::utility::Serializer& serializer) const override;
    static std::shared_ptr<RDFGraphCatalogEntry> deserialize(lingodb::utility::Deserializer& deserializer);
    ~RDFGraphCatalogEntry() override = default;
    IRI getIri() const { return IRI{name}; }
    IRI getNodeIri(int32_t node) const;
    IRI getRelationIri(int32_t rel) const;
    std::string_view getLocalId(int32_t node) const;
    lingodb::runtime::PropertyGraph& getStorage() override;
    virtual void flush() override;
    virtual void ensureFullyLoaded() override;
    virtual void setShouldPersist(bool shouldPersist) override;
    virtual void setDBDir(std::string dbDir) override;
    static std::shared_ptr<RDFGraphCatalogEntry> createFromCreateRdfGraphDef(const CreateRdfGraphDef& def);
};
} // lingodb::semantics

#endif // GENGODB_GRAPHCATALOGENTRY_H