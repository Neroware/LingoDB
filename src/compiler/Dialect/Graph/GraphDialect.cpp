#include "lingodb/compiler/Dialect/Graph/GraphDialect.h"

#include "lingodb/compiler/Dialect/Graph/GraphOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace lingodb::compiler::dialect::graph;
namespace {
   using namespace lingodb::compiler::dialect;
   static mlir::LogicalResult parseStateMembers(mlir::AsmParser& parser, subop::StateMembersAttr& stateMembersAttr) {
      if (parser.parseLSquare()) return mlir::failure();
      std::vector<mlir::Attribute> names;
      std::vector<mlir::Attribute> types;
      while (true) {
         if (!parser.parseOptionalRSquare()) { break; }
         llvm::StringRef colName;
         mlir::Type t;
         if (parser.parseKeyword(&colName) || parser.parseColon() || parser.parseType(t)) { return mlir::failure(); }
         names.push_back(parser.getBuilder().getStringAttr(colName));
         types.push_back(mlir::TypeAttr::get(t));
         if (!parser.parseOptionalComma()) { continue; }
         if (parser.parseRSquare()) { return mlir::failure(); }
         break;
      }
      stateMembersAttr = subop::StateMembersAttr::get(parser.getContext(), parser.getBuilder().getArrayAttr(names), parser.getBuilder().getArrayAttr(types));
      return mlir::success();
   }
   static void printStateMembers(mlir::AsmPrinter& p, subop::StateMembersAttr stateMembersAttr) {
      p << "[";
      auto first = true;
      for (size_t i = 0; i < stateMembersAttr.getNames().size(); i++) {
         auto name = mlir::cast<mlir::StringAttr>(stateMembersAttr.getNames()[i]).str();
         auto type = mlir::cast<mlir::TypeAttr>(stateMembersAttr.getTypes()[i]).getValue();
         if (first) {
            first = false;
         } else {
            p << ", ";
         }
         p << name << " : " << type;
      }
      p << "]";
   }
   } // namespace

#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOpsTypes.cpp.inc"

void GraphDialect::initialize() {
    addOperations<
    #define GET_OP_LIST
    #include "lingodb/compiler/Dialect/Graph/GraphOps.cpp.inc"
    
          >();
    
       addTypes<
    #define GET_TYPEDEF_LIST
    #include "lingodb/compiler/Dialect/Graph/GraphOpsTypes.cpp.inc"
    
          >();
       addAttributes<
    #define GET_ATTRDEF_LIST
    #include "lingodb/compiler/Dialect/Graph/GraphOpsAttributes.cpp.inc"
    
          >();
    
       // mlir::arith::CmpIOp::attachInterface<ArithCmpICmpInterface>(*getContext());
}
#include "lingodb/compiler/Dialect/Graph/GraphOpsDialect.cpp.inc"