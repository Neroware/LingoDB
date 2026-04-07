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
    "lingodb::compiler::dialect::relalg::ColumnSet", "getAllVariables",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getAllVariables(this->getOperation());
}]
>,
];
}

#endif // GENGODB_DIALECT_GPM_IR_GPMINTERFACES