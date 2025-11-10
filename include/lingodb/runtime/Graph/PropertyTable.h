#ifndef LINGODB_RUNTIME_GRAPH_PROPERTYTABLE_H
#define LINGODB_RUNTIME_GRAPH_PROPERTYTABLE_H

#include "lingodb/runtime/helpers.h"

namespace lingodb::runtime {

typedef int64_t property_id_t;
typedef int64_t property_type_id_t;
typedef int64_t property_key_t;

struct PropertyEntry {
    property_id_t id;
    property_id_t nextProp;
    property_id_t prevProp;
    property_key_t key;
    property_type_id_t type;
    void* value;
};

} // END namespace lingodb::runtime

#endif // LINGODB_RUNTIME_GRAPH_PROPERTYTABLE_H