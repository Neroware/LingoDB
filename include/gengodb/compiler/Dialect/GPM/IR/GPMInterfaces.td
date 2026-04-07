#ifndef GENGODB_DIALECT_GPM_IR_GPMINTERFACES
#define GENGODB_DIALECT_GPM_IR_GPMINTERFACES
include "mlir/IR/OpBase.td"

def GPMOperator : OpInterface<"GPMOperator"> {
let methods = [
InterfaceMethod<
    "get all columns that are created by this operator",
    "lingodb::compiler::dialect::relalg::ColumnSet", "getCreatedColumns",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getCreatedColumns(this->getOperation());
}]
>,
InterfaceMethod<
    "get all 'free' columns",
    "lingodb::compiler::dialect::relalg::ColumnSet", "getFreeColumns",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getFreeColumns(this->getOperation());
}]
>,
InterfaceMethod<
    "get all bound variables",
    "llvm::SmallVector<gengodb::compiler::dialect::gpm::VariableTermAttr, 4>", "getBoundVariables",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getBoundVariables(this->getOperation());
}]
>,
InterfaceMethod<
    "get all unbound variables",
    "llvm::SmallVector<gengodb::compiler::dialect::gpm::VariableTermAttr, 4>", "getUnboundVariables",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getUnboundVariables(this->getOperation());
}]
>,
];
}

#endif // GENGODB_DIALECT_GPM_IR_GPMINTERFACES