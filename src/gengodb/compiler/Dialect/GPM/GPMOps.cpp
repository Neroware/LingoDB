#include "gengodb/compiler/Dialect/GPM/IR/GPMOps.h"

#include "gengodb/compiler/Dialect/GPM/IR/GPMDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace lingodb::compiler::dialect;
using namespace gengodb::compiler::dialect;

namespace {
using namespace lingodb::compiler::dialect;
tuples::ColumnManager& getColumnManager(::mlir::OpAsmParser& parser) {
   return parser.getBuilder().getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
}
void printCustRef(OpAsmPrinter& p, mlir::Operation* op, tuples::ColumnRefAttr attr) {
   p << attr.getName();
}
ParseResult parseCustRefArr(OpAsmParser& parser, ArrayAttr& attr) {
   ArrayAttr parsedAttr;
   std::vector<Attribute> attributes;
   if (parser.parseAttribute(parsedAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
      return failure();
   }
   for (auto a : parsedAttr) {
      SymbolRefAttr parsedSymbolRefAttr = mlir::dyn_cast<SymbolRefAttr>(a);
      tuples::ColumnRefAttr attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
      attributes.push_back(attr);
   }
   attr = ArrayAttr::get(parser.getBuilder().getContext(), attributes);
   return success();
}
void printCustRefArr(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   std::vector<Attribute> attributes;
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      tuples::ColumnRefAttr parsedSymbolRefAttr = mlir::dyn_cast<tuples::ColumnRefAttr>(a);
      p << parsedSymbolRefAttr.getName();
   }
   p << "]";
}
ParseResult parseBinding(OpAsmParser& parser, mlir::Attribute& result) {
    SymbolRefAttr attrSymbolAttr;
    if (parser.parseAttribute(attrSymbolAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
    std::string attrName(attrSymbolAttr.getLeafReference().getValue());
    if (parser.parseOptionalLParen().succeeded()) {
        DictionaryAttr dictAttr;
        if (parser.parseAttribute(dictAttr)) { return failure(); }
        mlir::ArrayAttr fromExisting;
        if (parser.parseRParen()) { return failure(); }
        if (parser.parseOptionalEqual().succeeded()) {
            if (parseCustRefArr(parser, fromExisting)) {
                return failure();
            }
        }
        auto attr = getColumnManager(parser).createDef(attrSymbolAttr, fromExisting);
        auto propType = mlir::dyn_cast<TypeAttr>(dictAttr.get("type")).getValue();
        attr.getColumn().type = propType;
        result = attr;
        return success();
    }
    result = getColumnManager(parser).createRef(attrSymbolAttr);
    return success();
}
void printCustDef(OpAsmPrinter& p, mlir::Operation* op, tuples::ColumnDefAttr attr) {
   p << attr.getName();
   std::vector<mlir::NamedAttribute> relAttrDefProps;
   MLIRContext* context = attr.getContext();
   const tuples::Column& relationalAttribute = attr.getColumn();
   relAttrDefProps.push_back({mlir::StringAttr::get(context, "type"), mlir::TypeAttr::get(relationalAttribute.type)});
   p << "(" << mlir::DictionaryAttr::get(context, relAttrDefProps) << ")";
   Attribute fromExisting = attr.getFromExisting();
   if (fromExisting) {
      ArrayAttr fromExistingArr = mlir::dyn_cast_or_null<ArrayAttr>(fromExisting);
      p << "=";
      printCustRefArr(p, op, fromExistingArr);
   }
}
ParseResult parseTerm(OpAsmParser& parser, mlir::Attribute& attr) {
    auto ctxt = parser.getContext();
    if (!parser.parseOptionalKeyword("id")) {
        std::string ident;
        if (parser.parseLBrace() || parser.parseString(&ident) || parser.parseRBrace()) {
            return failure();
        }
        attr = gpm::IdentifierTermAttr::get(ctxt, StringAttr::get(ctxt, ident));
        return success();
    }
    else if (!parser.parseOptionalKeyword("_")) {
        std::string localId;
        if (parser.parseLBrace() || parser.parseString(&localId) || parser.parseRBrace()) {
            return failure();
        }
        attr = gpm::BNodeTermAttr::get(ctxt, StringAttr::get(ctxt, localId));
        return success();
    }
    else if (!parser.parseOptionalQuestion()) {
        std::string var;
        if (parser.parseString(&var) || parser.parseLBrace()) {
            return failure();
        }
        mlir::Attribute binding;
        if (parseBinding(parser, binding)) {
            return failure();
        }
        if (binding && parser.parseRBrace().succeeded()) { 
            attr = gpm::VariableTermAttr::get(ctxt, StringAttr::get(ctxt, var), binding);
            return success();
        }
    }
    return failure();
}
void printTerm(OpAsmPrinter& p, mlir::Operation* op, mlir::Attribute attr) {
    llvm::TypeSwitch<::mlir::Attribute>(attr)
        .Case<gpm::IdentifierTermAttr>([&](auto idTerm) {
            idTerm.print(p);
        })
        .Case<gpm::BNodeTermAttr>([&](auto bnode) {
            bnode.print(p);
        })
        .Case<gpm::VariableTermAttr>([&](auto varTerm) {
            p << "?";
            p << varTerm.getName();
            p << "{";
            if (varTerm.hasBinding()) {
                printCustRef(p, op, varTerm.getBindingReference());
            }
            else {
                printCustDef(p, op, varTerm.getProducedBinding());
            }
            p << "}";
        });
}
} // namespace

::mlir::LogicalResult gpm::TriplePatternOp::verify() {
    auto isValidTerm = [](mlir::Attribute attr) {
        return mlir::isa<
            IdentifierTermAttr,
            BNodeTermAttr,
            VariableTermAttr
        >(attr);
    };
    if (!isValidTerm(getS())) {
        return emitOpError("subject must be a GPM term attribute");
    }
    if (!isValidTerm(getP())) {
        return emitOpError("predicate must be a GPM term attribute");
    } 
    if (!isValidTerm(getO())) {
        return emitOpError("object must be a GPM term attribute");
    }
    if (mlir::isa<BNodeTermAttr>(getP())) {
        return emitOpError("predicate cannot be a blank node");
    }
    return mlir::success();
}

#define GET_OP_CLASSES
#include "gengodb/compiler/Dialect/GPM/IR/GPMOps.cpp.inc"