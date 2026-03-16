#ifndef GENGODB_RUNTIME_GENGODBGRAPH_H
#define GENGODB_RUNTIME_GENGODBGRAPH_H

#include "gengodb/runtime/PropertyGraph.h"

namespace lingodb::runtime {

class GengoDBGraph : public PropertyGraph {
public:
    static const size_t DEFAULT_CAPACITY = 1024;
private:
    bool persist;
    std::string fileName;
    std::string dbDir;

    bool loaded = false;
public:
    GengoDBGraph(std::string fileName) 
        : PropertyGraph(DEFAULT_CAPACITY, DEFAULT_CAPACITY, DEFAULT_CAPACITY), persist(false), fileName(std::move(fileName)) {}
    GengoDBGraph(std::string fileName, size_t maxNodeCapacity, size_t maxRelCapacity, size_t maxPropCapacity) 
        : PropertyGraph(maxNodeCapacity, maxRelCapacity, maxPropCapacity), persist(false), fileName(std::move(fileName)) {}
    void setPersist(bool persist) {
        this->persist = persist;
        if (persist) {
            flush();
        }
    }
    virtual ~GengoDBGraph() = default;
    //flushes the data to disk
    void flush();
    //ensures that the data is loaded
    void ensureLoaded();
    virtual void setDBDir(std::string dbDir) {
        this->dbDir = dbDir;
    };
    virtual std::string getDBDir() const { return this->dbDir; }
    void serialize(lingodb::utility::Serializer& serializer) const;
    static std::unique_ptr<GengoDBGraph> deserialize(lingodb::utility::Deserializer& deserializer);
    static std::unique_ptr<GengoDBGraph> create(std::string name);
}; // GengoDBGraph

} // namespace lingodb::runtime

#endif // GENGODB_RUNTIME_GENGODBGRAPH_H