#ifndef GENGODB_DIALECT_GPM_IR_GPMINTERFACES
#define GENGODB_DIALECT_GPM_IR_GPMINTERFACES
include "mlir/IR/OpBase.td"

def Operator : OpInterface<"Operator"> {
let methods = [
InterfaceMethod<
        "a dummy method",
        "uint64_t", "getAnswer",(ins ),"", /*defaultImplementation=*/[{
    return 42;
}]
>,
];
}

#endif // GENGODB_DIALECT_GPM_IR_GPMINTERFACES