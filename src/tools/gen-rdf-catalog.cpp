#include <fstream>
#include <iostream>
#include <string>

#include "features.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/scheduler/Scheduler.h"

#include "gengodb/catalog/CreateRdfGraphDef.h"
#include "gengodb/catalog/GraphCatalogEntry.h"
#include "gengodb/semantics/RdfFileFormat.h"

using namespace gengodb::semantics;
using namespace gengodb::catalog;

int main(int argc, char** argv) {
   using namespace lingodb;

   if (argc < 2 || std::string(argv[1]) == "--help") {
      std::cout << "Generates a LingoDB catalog from RDF files. Usage: gen-rdf-catalog [RDF_DIR]" << std::endl;
      return 0;
   }

   bool eagerLoading = std::getenv("LINGODB_BACKEND_ONLY");
   std::shared_ptr<runtime::Session> session = runtime::Session::createSession(std::string(argv[1]), eagerLoading);

   CreateRdfGraphDef def{"coffee", rdf4cpp::IRI{"https://github.com/Neroware/LingoDB/tree/rz/main/resources/ttl/coffee.ttl#rdf"}, RDFFileFormat::TURTLE};
   auto entry = RDFGraphCatalogEntry::createFromCreateRdfGraphDef(def);

   // TODO generate for all

   entry->setDBDir(session->getCatalog()->getDbDir());
   entry->ensureFullyLoaded();
   
   // session->getCatalog()->insertEntry(entry);
   session->getCatalog()->setShouldPersist(true);
   session->getCatalog()->persist();

   return 0;
}
