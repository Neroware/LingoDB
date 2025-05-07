#ifndef LINGODB_DIALECT_GRAPH_DIALECT
#define LINGODB_DIALECT_GRAPH_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "lingodb/compiler/Dialect/SubOperator/SubOperatorBase.td"

//===----------------------------------------------------------------------===//
// Graph dialect definition.
//===----------------------------------------------------------------------===//

def Graph_Dialect : Dialect {
    let name = "graph";
    let summary = "A dialect for universal graph processing";
    let cppNamespace = "::lingodb::compiler::dialect::graph";
    let useDefaultTypePrinterParser=1;
    let useDefaultAttributePrinterParser=1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class Graph_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Graph_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def MyExampleState : Graph_Type<"MyExampleState", "example_state",[State]> {
    let summary = "an example state";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}

def Graph : Graph_Type<"Graph", "graph"> {
    let summary = "graph type";
}
def Node : Graph_Type<"Node", "node"> {
    let summary = "node type";
}
def Edge : Graph_Type<"Edge", "edge"> {
    let summary = "edge type";
}
def NodeSet : Graph_Type<"NodeSet", "node_set"> {
    let summary = "set of nodes";
}
def EdgeSet : Graph_Type<"EdgeSet", "edge_set"> {
    let summary = "set of edges";
}
// def NodeIterator : Graph_Type<"NodeIterator", "node_iterator"> {
//     let summary = "iterator over a set of nodes";
// }
// def EdgeIterator : Graph_Type<"EdgeIterator", "edge_iterator"> {
//     let summary = "iterator over a set of edges";
// }

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

class Graph_Attr<string name> : AttrDef<Graph_Dialect, name>;

def PropertyRefAttr : Graph_Attr<"PropertyRef"> {
    let mnemonic = "propertyref";
    let parameters = (ins "mlir::SymbolRefAttr":$name); // TODO Add type information
    let assemblyFormat = "`<` $name `>`"; // TODO use custom assembly format parser/printer
}

#endif // LINGODB_DIALECT_GRAPH_DIALECT