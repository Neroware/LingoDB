#ifndef LINGODB_SEMANTICS_DATATYPES_H
#define LINGODB_SEMANTICS_DATATYPES_H

namespace lingodb::semantics {

// Supported XSD Datatypes following alphabetical ordering as on the table
// at https://www.w3.org/2011/rdf-wg/wiki/XSD_Datatypes
enum XSDType {
    UNKNOWN = -1,
    XSD_ANY_URI = 0,
    XSD_TYPE_BOOLEAN = 2,
    XSD_TYPE_BYTE = 3,
    XSD_TYPE_DOUBLE = 7,
    XSD_TYPE_FLOAT = 11,
    XSD_TYPE_INT = 21,
    XSD_TYPE_LONG = 24,
    XSD_TYPE_SHORT = 36,
    XSD_TYPE_STRING = 37,
    XSD_TYPE_UNSIGNED_BYTE = 40,
    XSD_TYPE_UNSIGNED_INT = 41,
    XSD_TYPE_UNSIGNED_LONG = 42,
    XSD_TYPE_UNSIGNED_SHORT = 43,
};

} // namespace lingodb::semantics

#endif // LINGODB_SEMANTICS_DATATYPES_H