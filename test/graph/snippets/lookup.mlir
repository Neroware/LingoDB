module {
    func.func @main() {
    	%subop_result = subop.execution_group (){
            %g = graph.subop.create_graph "", iri = "builtin:testgraph" : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]>
            %g_scan = graph.subop.scan_graph %g : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]> @nodes::@set({type = !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>}), @edges::@set({type = !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>})
            %lookup_index = subop.create_simple_state !subop.simple_state<[lookupIdxI32: i32]> initial: {
                %c2_i32 = arith.constant 1 : i32
                tuples.return %c2_i32 : i32
            }
            %vx = subop.nested_map %g_scan [@nodes::@set] (%arg0, %arg1){
                %lookup_stream0 = subop.scan %lookup_index : !subop.simple_state<[lookupIdxI32: i32]> {lookupIdxI32 => @nodes::@lookupIdxI32({type = i32})}
                %lookup_stream1 = subop.lookup %lookup_stream0 %arg1[@nodes::@lookupIdxI32] : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]> @nodes::@ref({type = !graph.node_ref<[node_id : i32],[incoming : !graph.edge_set<[incoming_it : !graph.graph_set_iterator<["incoming"]>]>],[outgoing : !graph.edge_set<[outgoing_it : !graph.graph_set_iterator<["outgoing"]>]>],[property : i64]>})
                tuples.return %lookup_stream1 : !tuples.tuplestream
            }
            %result_nodes = subop.gather %vx @nodes::@ref {node_id => @nodes::@id({type = i32})}
            %outgoing_sets = subop.gather %result_nodes @nodes::@ref {outgoing => @outgoing::@set({type = !graph.edge_set<[outgoing_it : !graph.graph_set_iterator<["outgoing"]>]>})}

            %ex = subop.nested_map %outgoing_sets [@outgoing::@set] (%arg0, %arg1){
                %edge_stream = graph.subop.scan_edge_set %arg1 : !graph.edge_set<[outgoing_it : !graph.graph_set_iterator<["outgoing"]>]> @edges::@ref({type = !graph.edge_ref<[edge_id : i32],[from : !graph.node_ref<[node_id1 : i32],[incoming1 : !graph.edge_set<[incoming_it1 : !graph.graph_set_iterator<["incoming"]>]>],[outgoing1 : !graph.edge_set<[outgoing_it1 : !graph.graph_set_iterator<["outgoing"]>]>],[property1 : i64]>],[to : !graph.node_ref<[node_id2 : i32],[incoming2 : !graph.edge_set<[incoming_it2 : !graph.graph_set_iterator<["incoming"]>]>],[outgoing2 : !graph.edge_set<[outgoing_it2 : !graph.graph_set_iterator<["outgoing"]>]>],[property2 : i64]>],[edge_prop : i64]>})
                tuples.return %edge_stream : !tuples.tuplestream
            }
            %resultEdges = subop.gather %ex @edges::@ref {to => @edges::@toRef({type = !graph.node_ref<[node_id2 : i32],[incoming2 : !graph.edge_set<[incoming_it2 : !graph.graph_set_iterator<["incoming"]>]>],[outgoing2 : !graph.edge_set<[outgoing_it2 : !graph.graph_set_iterator<["outgoing"]>]>],[property2 : i64]>})}
            %resultEdges0 = subop.gather %resultEdges @edges::@toRef {node_id2 => @nodes2::@id({type = i32})}

            %0 = subop.create !subop.result_table<[int32p0 : i32, int32p1 : i32]>
            subop.materialize %resultEdges0 {@nodes::@id => int32p0, @nodes2::@id => int32p1}, %0 : !subop.result_table<[int32p0 : i32, int32p1 : i32]>
            %res = subop.create_from ["int32", "int32"] %0 : !subop.result_table<[int32p0 : i32, int32p1 : i32]> -> !subop.local_table<[int32p0 : i32, int32p1 : i32], ["int32", "int32"]>
            subop.execution_group_return %res : !subop.local_table<[int32p0 : i32, int32p1 : i32], ["int32", "int32"]>
        
        } -> !subop.table<[int32n0 : i32]>
        subop.set_result 0 %subop_result : !subop.table<[int32n0 : i32]>
        return
    }
}