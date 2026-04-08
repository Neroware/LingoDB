#ifndef GENGODB_DIALECT_GPM_IR_GPMINTERFACES
#define GENGODB_DIALECT_GPM_IR_GPMINTERFACES
include "mlir/IR/OpBase.td"

def GPMOperator : OpInterface<"GPMOperator"> {
let methods = [
InterfaceMethod<
    "get all variable bindings created by this operator",
    "lingodb::compiler::dialect::relalg::ColumnSet", "getCreatedVariables",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getCreatedVariables(this->getOperation());
}]
>,
InterfaceMethod<
    "get all bound variables used by this operator",
    "lingodb::compiler::dialect::relalg::ColumnSet", "getBoundVariables",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getBoundVariables(this->getOperation());
}]
>,
InterfaceMethod<
    "get all variables used by this operator",
    "lingodb::compiler::dialect::relalg::ColumnSet", "getUsedVariables",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getUsedVariables(this->getOperation());
}]
>,
InterfaceMethod<
    "get variables that are available after this operator",
    "lingodb::compiler::dialect::relalg::ColumnSet", "getAvailableVariables",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getAvailableVariables(this->getOperation());
}]
>,
InterfaceMethod<
    "",
    "void", "moveSubTreeBefore",(ins "Operator":$before),"", /*defaultImplementation=*/[{
    gengodb::compiler::dialect::gpm::detail::moveSubTreeBefore(this->getOperation(),before);
}]
>,
InterfaceMethod<
        "get children of operator",
        "llvm::SmallVector<Operator,4>", "getChildren",(ins ),"", /*defaultImplementation=*/[{
    llvm::SmallVector<Operator,4> children;
    auto op=this->getOperation();
    for (auto operand : op->getOperands()) {
        mlir::Type t=operand.getType();
        bool isreltype=mlir::isa<lingodb::compiler::dialect::tuples::TupleStreamType>(t);
        auto childOp=operand.getDefiningOp();
        if (isreltype&&childOp) {
            auto childOperator=mlir::dyn_cast_or_null<Operator>(childOp);
            if(childOperator){
                    children.push_back(childOperator);
            }
        }
    }
    return children;
}]
>,
InterfaceMethod<
        "get children of operator",
        "llvm::SmallVector<Operator,4>", "getAllSubOperators",(ins ),"", /*defaultImplementation=*/[{
    llvm::SmallVector<Operator,4> suboperators;
    suboperators.push_back(mlir::dyn_cast_or_null<Operator>(this->getOperation()));
    auto children=this->getChildren();
    for (auto child : children) {
        auto subsub=child.getAllSubOperators();
        suboperators.insert(suboperators.end(),subsub.begin(),subsub.end());
    }
    return suboperators;
}]
>,
InterfaceMethod<
        "set children of operator",
        "void ", "setChildren",(ins "llvm::SmallVector<Operator,4>":$children),"", /*defaultImplementation=*/[{
    auto op=this->getOperation();
    size_t i=0;
    size_t j=0;
    for (auto operand : op->getOperands()) {
        mlir::Type t=operand.getType();
        bool isreltype=mlir::isa<lingodb::compiler::dialect::tuples::TupleStreamType>(t);
        auto childOp=operand.getDefiningOp();
        if (isreltype&&childOp) {
            op->setOperand(i,children[j].asRelation());
            j++;
        }
        i++;
    }
}]
>,
InterfaceMethod<
    "get resulting relation of operator",
    "mlir::Value", "asRelation",(ins ),"", /*defaultImplementation=*/[{
    return this->getOperation()->getResult(0);
}]
>,
];
}

#endif // GENGODB_DIALECT_GPM_IR_GPMINTERFACES