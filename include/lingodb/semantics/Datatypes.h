#ifndef LINGODB_SEMANTICS_DATATYPES_H
#define LINGODB_SEMANTICS_DATATYPES_H

namespace lingodb::semantics {

// Supported XSD Datatypes following alphabetical ordering as on the table
// at https://www.w3.org/2011/rdf-wg/wiki/XSD_Datatypes
enum XSDType {
    XSD_UNDEFINED = 0,
    XSD_TYPE_BOOLEAN = 3,
    XSD_TYPE_BYTE = 4,
    XSD_TYPE_DOUBLE = 10,
    XSD_TYPE_FLOAT = 11,
    XSD_TYPE_INT = 18,
    XSD_TYPE_INTEGER = 19,
    XSD_TYPE_LONG = 21,
    XSD_TYPE_SHORT = 30,
    XSD_TYPE_STRING = 31,
    XSD_TYPE_UNSIGNED_BYTE = 34,
    XSD_TYPE_UNSIGNED_INT = 35,
    XSD_TYPE_UNSIGNED_LONG = 36,
    XSD_TYPE_UNSIGNED_SHORT = 37,
};

} // namespace lingodb::semantics

#endif // LINGODB_SEMANTICS_DATATYPES_H