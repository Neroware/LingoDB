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

def ScanGraphOp : Graph_Op<"subop.scan_graph", [Pure,SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scan state (sequentially)";
    let arguments = (ins AnyType : $graph, ColumnDefAttr: $nodeSet, ColumnDefAttr: $edgeSet);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{ $graph `:` type($graph) custom<CustDef>($nodeSet) `,` custom<CustDef>($edgeSet) attr-dict }];
    let extraClassDeclaration = [{
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}

def ScanNodeSetOp : Graph_Op<"subop.scan_node_set", [Pure,SubOperator,ReferenceProducer,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scan state (sequentially)";
    let arguments = (ins AnyType : $node_set, ColumnDefAttr: $ref);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{ $node_set `:` type($node_set) custom<CustDef>($ref) attr-dict }];
    let extraClassDeclaration = [{
        dialect::tuples::ColumnDefAttr getProducedReference(){ return getRef(); }
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}
def ScanEdgeSetOp : Graph_Op<"subop.scan_edge_set", [Pure,SubOperator,ReferenceProducer,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scan state (sequentially)";
    let arguments = (ins AnyType : $edge_set, ColumnDefAttr: $ref);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{ $edge_set `:` type($edge_set) custom<CustDef>($ref) attr-dict }];
    let extraClassDeclaration = [{
        dialect::tuples::ColumnDefAttr getProducedReference(){ return getRef(); }
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Property Graph
////////////////////////////////////////////////////////////////////////////////////////////////////

// The following operations are strictly runtime-specific for the given PropertyGraph data structure 
// adding the processing of Neo4j-style graphs to LingoDB. This means they are not necessary for a 
// general graph processing IR, which is minimally complete with lowerings for CreateOp, ScanOp, 
// GatherOp and ScatterOp.

def RelationshipTypeOp : Graph_Op<"relationship_type", [Pure,SubOperator,ReferenceProducer]> {
    let summary = "gather relationship type";

    let arguments = (ins TupleStream : $stream, ColumnRefAttr: $edgeRef, ColumnDefAttr:$valueRef );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream custom<CustRef>($edgeRef) `is` `a` custom<CustDef>($valueRef) attr-dict }];
    let extraClassDeclaration = [{
        dialect::tuples::ColumnDefAttr getProducedReference(){ return getValueRef(); }
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}

def NodeCountOp : Graph_Op<"node_count", [Pure,SubOperator,ReferenceProducer,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "property graph node count";

    let arguments = (ins TupleStream : $stream, Graph : $graph, ColumnDefAttr:$ref);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream `,` $graph `->` custom<CustDef>($ref) `:` type($graph) attr-dict }];
    let extraClassDeclaration = [{
        dialect::tuples::ColumnDefAttr getProducedReference(){ return getRef(); }
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}

def RelationshipCountOp : Graph_Op<"relationship_count", [Pure,SubOperator,ReferenceProducer,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "property graph relationship count";

    let arguments = (ins TupleStream : $stream, Graph : $graph, ColumnDefAttr:$ref);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream `,` $graph `->` custom<CustDef>($ref) `:` type($graph) attr-dict }];
    let extraClassDeclaration = [{
        dialect::tuples::ColumnDefAttr getProducedReference(){ return getRef(); }
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Graph IR <-> SubOp IR
////////////////////////////////////////////////////////////////////////////////////////////////////

// def AdjacencyTableOp : Graph_Op<"adjacency_table", [Pure,StateCreator]> {
//     let summary = "create adjacency table for relationship";
//     let results = (outs Buffer : $res);
//     let assemblyFormat = [{ type($res) attr-dict }];
// }

#endif // LINGODB_DIALECT_GRAPH_OPS