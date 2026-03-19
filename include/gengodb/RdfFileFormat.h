#ifndef GENGODB_RDFFILEFORMAT_H
#define GENGODB_RDFFILEFORMAT_H

#include <rdf4cpp.hpp>
#include <rdf4cpp/parser/RDFFileParser.hpp>

namespace gengodb::semantics {

enum class RDFFileFormat {
    DEFAULT,
    BINARY,
    TURTLE
};

inline rdf4cpp::parser::ParsingFlag getRDFParseFlags(RDFFileFormat format) {
    switch (format) {
        case RDFFileFormat::TURTLE: return rdf4cpp::parser::ParsingFlag::Turtle;
        default: return (rdf4cpp::parser::ParsingFlag) 0;
    }
}

}


#endif // GENGODB_RDFFILEFORMAT_H