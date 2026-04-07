module {
    // Get named graph "coffee" whose IRI is already stored in the catalog
    // %0 = gpm.basegraph graph : "coffee", column : @graphs::@coffee({type = !gpm.graph_ref})

    // Download and get the RDF graph from the semantic web at the provided IRI
    // %0 = gpm.basegraph graph : "https://raw.githubusercontent.com/Neroware/LingoDB/refs/heads/rz/main/resources/ttl/coffee.ttl#rdf", column : @graphs::@coffee({type = !gpm.graph_ref})
    
    // Load the graph from the local RDF file.
    %0 = gpm.basegraph graph : "file://resources/ttl/coffee.ttl", column : @graphs::@coffee({type = !gpm.graph_ref})
    %bgp = gpm.basic_graph_pattern %0 (%arg : !tuples.tuplestream){
        %1 = gpm.triple_pattern %arg @graphs::@coffee(_{"guy"}, id{"ex:drinks"}, id{"ex:Coffee"})
        %2 = gpm.triple_pattern %1 @graphs::@coffee(?"who"{@bindings::@who({type = !gpm.variable_binding})}, id{"ex:drinks"}, id{"ex:Coffee"})
        %3 = gpm.triple_pattern %2 @graphs::@coffee(?"who"{@bindings::@who}, id{"rdf:type"}, id{"ex:Person"})
        tuples.return %3 : !tuples.tuplestream
    }

}

