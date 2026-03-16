#include "features.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/Timing.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/utility/Setting.h"
#include "gengodb/RdfGraph.h"
#include "gengodb/GraphCatalogEntry.h"

#include <fstream>
#include <iostream>
#include <string>

namespace {
lingodb::utility::GlobalSetting<bool> eagerLoading("system.eager_loading", false);
} // namespace
int main(int argc, char** argv) {
   using namespace lingodb;

   if (argc == 2 && std::string(argv[1]) == "--features") {
      printFeatures();
      return 0;
   }

   if (argc <= 2) {
      std::cerr << "USAGE: run-sparql *.sql database (still behaves like run-sql but loads RDF graph data)" << std::endl;
      return 1;
   }
   std::string inputFileName = std::string(argv[1]);
   std::string directory = std::string(argv[2]);
   std::cout << "Loading Database from: " << directory << '\n';
   auto session = runtime::Session::createSession(directory, eagerLoading.getValue());
   // TODO Move this into a session! Add RdfGraphCatalogEntry!
   //
   // TODO This breaks linking because XMLLib is missing (yet another dependency). 
   // >>> rdf4cpp needs to be included differently, directly through its build state from Conan...
   //
   // TODO Use Catalog for Property Graph storage
   //
   // TODO Use Catalog for RdfGraph registry...
   // gengodb::semantics::RdfGraphRegistry::instance().add(gengodb::semantics::RdfGraph::create(rdf4cpp::IRI{"foo:bar"}));

   lingodb::compiler::support::eval::init();
   execution::ExecutionMode runMode = execution::getExecutionMode();
   auto queryExecutionConfig = execution::createQueryExecutionConfig(runMode, true);
   unsetenv("PERF_BUILDID_DIR");
   queryExecutionConfig->timingProcessor = std::make_unique<execution::TimingPrinter>(inputFileName);

   auto scheduler = scheduler::startScheduler();
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
   executer->fromFile(inputFileName);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
   return 0;
}
