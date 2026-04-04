module {
    %0 = gpm.basegraph graph : "coffee", column : @graphs::@coffee({type = !gpm.graph_ref})
    %1 = gpm.triple_pattern %0 @graphs::@coffee(_{"guy"}, id{"ex:drinks"}, id{"ex:Coffee"})
    %2 = gpm.triple_pattern %1 @graphs::@coffee(?"who"{@bindings::@who({type = !gpm.variable_binding})}, id{"ex:drinks"}, id{"ex:Coffee"})
    %3 = gpm.triple_pattern %2 @graphs::@coffee(?"who"{@bindings::@who}, id{"rdf:type"}, id{"ex:Person"})
}

