module {
    %0 = relalg.const_relation columns : [@t::@col1({type = !db.string})] values : [["A"],["B"]]
    %1 = gpm.triple_pattern %0 (_{"guy"}, id{"ex:drinks"}, id{"ex:Coffee"})
    %2 = gpm.triple_pattern %1 (?"who"{@bindings::@who({type = i64})}, id{"ex:drinks"}, id{"ex:Coffee"})
    %3 = gpm.triple_pattern %2 (?"who"{@bindings::@who}, id{"rdf:type"}, id{"ex:Person"})
}

