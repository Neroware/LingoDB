#ifndef GENGODB_DIALECT_GPM_IR_GPMINTERFACES
#define GENGODB_DIALECT_GPM_IR_GPMINTERFACES
include "mlir/IR/OpBase.td"

def GPMOperator : OpInterface<"GPMOperator"> {
let methods = [
InterfaceMethod<
    "get the answer",
    "int", "getTheAnswer", (ins ), "", /*defaultImplementation=*/[{
        return 42;
    }]
>
// InterfaceMethod<
//     "get all variables that are used directly by this operator",
//     "lingodb::compiler::dialect::relalg::ColumnSet", "getUsedVariables",(ins ),"", /*defaultImplementation=*/[{
//     return lingodb::compiler::dialect::relalg::detail::getUsedColumns(this->getOperation());
// }]
// >,
];
}

#endif // GENGODB_DIALECT_GPM_IR_GPMINTERFACES