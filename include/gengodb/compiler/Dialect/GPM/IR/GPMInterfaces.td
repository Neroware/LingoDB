#ifndef GENGODB_DIALECT_GPM_IR_GPMINTERFACES
#define GENGODB_DIALECT_GPM_IR_GPMINTERFACES
include "mlir/IR/OpBase.td"

def GPMOperator : OpInterface<"GPMOperator"> {
let methods = [
InterfaceMethod<
    "get all columns bound by variables",
    "lingodb::compiler::dialect::relalg::ColumnSet", "getBoundVariables",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getBoundVariables(this->getOperation());
}]
>,
InterfaceMethod<
    "get all columns of still unbound variables",
    "lingodb::compiler::dialect::relalg::ColumnSet", "getUnboundVariables",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getUnboundVariables(this->getOperation());
}]
>,
InterfaceMethod<
    "get all columns of bound and unbound variables",
    "lingodb::compiler::dialect::relalg::ColumnSet", "getAllVariables",(ins ),"", /*defaultImplementation=*/[{
    return gengodb::compiler::dialect::gpm::detail::getAllVariables(this->getOperation());
}]
>
];
}

#endif // GENGODB_DIALECT_GPM_IR_GPMINTERFACES