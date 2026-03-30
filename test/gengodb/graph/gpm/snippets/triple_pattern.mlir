module {
    %0 = relalg.const_relation columns : [@t::@col1({type = !db.string})] values : [["A"],["B"]]
    %1 = gpm.triple_pattern %0 (?"who"{@bindings::@who({type = i64})}, id{"ex:drinks"}, id{"ex:Coffee"})
}

