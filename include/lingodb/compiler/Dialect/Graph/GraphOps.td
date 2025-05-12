#ifndef LINGODB_DIALECT_GRAPH_OPS
#define LINGODB_DIALECT_GRAPH_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "lingodb/compiler/Dialect/Graph/GraphDialect.td"

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

class Graph_Op<string mnemonic, list<Trait> traits = []> : Op<Graph_Dialect, mnemonic, traits> {
    let hasCustomAssemblyFormat = 1;
}

def CreateGraphOp : Graph_Op<"subop.create_graph", [Pure,StateCreator]> {
    let summary = "create new graph";
    let results = (outs Graph : $res);
    let assemblyFormat = [{ type($res) attr-dict }];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Iterating over nodes and edges
////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO

////////////////////////////////////////////////////////////////////////////////////////////////////
// Properties
////////////////////////////////////////////////////////////////////////////////////////////////////

// def GetNodePropertyOp : Graph_Op<"get_node_property", [Pure]> {
//     let summary = "get node property operation";

//     let arguments = (ins Node:$node,PropertyRefAttr:$propRef);
//     let results = (outs AnyType : $res);
//     let assemblyFormat = [{  $node $propRef `:` type($res) attr-dict }];
// }

// def GetEdgePropertyOp : Graph_Op<"get_edge_property", [Pure]> {
//     let summary = "get edge property operation";

//     let arguments = (ins Node:$node,PropertyRefAttr:$propRef);
//     let results = (outs AnyType : $res);
//     let assemblyFormat = [{  $node $propRef `:` type($res) attr-dict }];
// }

#endif // LINGODB_DIALECT_GRAPH_OPS