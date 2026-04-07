#ifndef GENGODB_DIALECT_GPM_IR_GPMOPS
#define GENGODB_DIALECT_GPM_IR_GPMOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgInterfaces.td"
include "gengodb/compiler/Dialect/GPM/IR/GPMBase.td"

include "GPMInterfaces.td"

//===----------------------------------------------------------------------===//
// Base GPM operation definition.
//===----------------------------------------------------------------------===//

class GPM_Op<string mnemonic, list<Trait> traits = []> : Op<GPM_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

def TriplePatternOp : GPM_Op<"triple_pattern", [Pure,GPMOperator]> {
  let summary = "triple pattern";
  let arguments = (ins TupleStream:$rel, ColumnRefAttr:$graphRef, AnyAttr:$s, AnyAttr:$p, AnyAttr:$o);
  let results = (outs TupleStream:$res);
  let assemblyFormat = [{  $rel custom<CustRef>($graphRef) `(` custom<Term>($s) `,` custom<Term>($p) `,` custom<Term>($o) `)` attr-dict }];
  let hasVerifier = 1;
}

def BaseGraphOp : GPM_Op<"basegraph", [Pure,GPMOperator]> {
  let summary = "base graph";
  let arguments = (ins StrAttr:$graph, ColumnDefAttr:$def);
  let results = (outs TupleStream:$res);
  let assemblyFormat = [{  `graph` `:` $graph `,` `column` `:` custom<CustDef>($def) attr-dict }];
}

def BasicGraphPatternOp : GPM_Op<"basic_graph_pattern", [Pure,GPMOperator]> {
  let summary = "basic graph pattern";
  let arguments = (ins TupleStream:$rel);
  let results = (outs TupleStream:$res);
  let regions = (region SizedRegion<1>:$pattern);
  let assemblyFormat = [{  $rel custom<CustRegion>($pattern) attr-dict  }];
  let extraClassDeclaration = [{
    llvm::SmallVector<TriplePatternOp, 16> getTriples();
  }];
  let hasVerifier = 1;
}

#endif // GENGODB_DIALECT_GPM_IR_GPMOPS