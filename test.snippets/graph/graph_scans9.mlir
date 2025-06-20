// ./build/mlir-db-opt test.snippets/graph/graph_scans.mlir

module {
    func.func @main() {
    	%subop_result = subop.execution_group (){
            
            %g = graph.subop.create_graph !graph.graph<[vx : !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.edge_set_iterator<["all"]>]>]>
            
            // Stream A
            %streamA0 = graph.subop.scan_graph %g : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.edge_set_iterator<["all"]>]>]> @nodes::@set({type = !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]>}), @edges::@set({type = !graph.edge_set<[ex_it : !graph.edge_set_iterator<["all"]>]>})
            %streamA1 = subop.nested_map %streamA0 [@nodes::@set] (%arg0, %arg1){
                %node_stream = graph.subop.scan_node_set %arg1 : !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]> @nodes::@ref({type = !graph.node_ref<[node_id : i64],[incoming : !graph.edge_set<[incoming_it : !graph.edge_set_iterator<["incoming"]>]>],[outgoing : !graph.edge_set<[outgoing_it : !graph.edge_set_iterator<["outgoing"]>]>],[property : i64]>})
                tuples.return %node_stream : !tuples.tuplestream
            }
            %streamA2 = subop.gather %streamA1 @nodes::@ref {outgoing => @outgoing::@set({type = !graph.edge_set<[outgoing_it : !graph.edge_set_iterator<["outgoing"]>]>})}

            %streamA3 = subop.nested_map %streamA2 [@outgoing::@set] (%arg0, %arg1){
                %edge_stream = graph.subop.scan_edge_set %arg1 : !graph.edge_set<[outgoing_it : !graph.edge_set_iterator<["outgoing"]>]> @edges::@ref({type = !graph.edge_ref<[edge_id : i64],[from : !graph.node_ref<[node_id1 : i64],[incoming1 : !graph.edge_set<[incoming_it1 : !graph.edge_set_iterator<["incoming"]>]>],[outgoing1 : !graph.edge_set<[outgoing_it1 : !graph.edge_set_iterator<["outgoing"]>]>],[property1 : i64]>],[to : !graph.node_ref<[node_id2 : i64],[incoming2 : !graph.edge_set<[incoming_it2 : !graph.edge_set_iterator<["incoming"]>]>],[outgoing2 : !graph.edge_set<[outgoing_it2 : !graph.edge_set_iterator<["outgoing"]>]>],[property2 : i64]>],[edge_prop : i64]>})
                tuples.return %edge_stream : !tuples.tuplestream
            }
            %streamA4 = subop.gather %streamA3 @edges::@ref {edge_id => @edges::@id({type = i64}), to => @edges::@toRef({type = !graph.node_ref<[node_id2 : i64],[incoming2 : !graph.edge_set<[incoming_it2 : !graph.edge_set_iterator<["incoming"]>]>],[outgoing2 : !graph.edge_set<[outgoing_it2 : !graph.edge_set_iterator<["outgoing"]>]>],[property2 : i64]>})}
            %streamA5 = subop.gather %streamA4 @edges::@toRef {node_id2 => @nodes::@id({type = i64})}
            subop.scatter %streamA5 @edges::@toRef { @edges::@id => property2 }

            // Stream B
            %streamB0 = graph.subop.scan_graph %g : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.edge_set_iterator<["all"]>]>]> @nodes::@set({type = !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]>}), @edges::@set({type = !graph.edge_set<[ex_it : !graph.edge_set_iterator<["all"]>]>})
            %streamB1 = subop.nested_map %streamB0 [@nodes::@set] (%arg0, %arg1){
                %node_stream = graph.subop.scan_node_set %arg1 : !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]> @nodes::@ref({type = !graph.node_ref<[node_id : i64],[incoming : !graph.edge_set<[incoming_it : !graph.edge_set_iterator<["incoming"]>]>],[outgoing : !graph.edge_set<[outgoing_it : !graph.edge_set_iterator<["outgoing"]>]>],[property : i64]>})
                tuples.return %node_stream : !tuples.tuplestream
            }
            %streamB2 = subop.gather %streamB1 @nodes::@ref {outgoing => @outgoing::@set({type = !graph.edge_set<[outgoing_it : !graph.edge_set_iterator<["outgoing"]>]>})}

            %streamB3 = subop.nested_map %streamB2 [@outgoing::@set] (%arg0, %arg1){
                %edge_stream = graph.subop.scan_edge_set %arg1 : !graph.edge_set<[outgoing_it : !graph.edge_set_iterator<["outgoing"]>]> @edges::@ref({type = !graph.edge_ref<[edge_id : i64],[from : !graph.node_ref<[node_id1 : i64],[incoming1 : !graph.edge_set<[incoming_it1 : !graph.edge_set_iterator<["incoming"]>]>],[outgoing1 : !graph.edge_set<[outgoing_it1 : !graph.edge_set_iterator<["outgoing"]>]>],[property1 : i64]>],[to : !graph.node_ref<[node_id2 : i64],[incoming2 : !graph.edge_set<[incoming_it2 : !graph.edge_set_iterator<["incoming"]>]>],[outgoing2 : !graph.edge_set<[outgoing_it2 : !graph.edge_set_iterator<["outgoing"]>]>],[property2 : i64]>],[edge_prop : i64]>})
                tuples.return %edge_stream : !tuples.tuplestream
            }
            %streamB4 = subop.gather %streamB3 @edges::@ref {edge_id => @edges::@id({type = i64}), to => @edges::@toRef({type = !graph.node_ref<[node_id2 : i64],[incoming2 : !graph.edge_set<[incoming_it2 : !graph.edge_set_iterator<["incoming"]>]>],[outgoing2 : !graph.edge_set<[outgoing_it2 : !graph.edge_set_iterator<["outgoing"]>]>],[property2 : i64]>}), edge_prop => @edges::@property({type = i64})}
            %streamB5 = subop.gather %streamB4 @edges::@toRef {node_id2 => @nodes::@id({type = i64}), property2 => @nodes::@property({type = i64})}

            %streamB6 = graph.node_count %streamB5, %g -> @graph::@numVertices({type = index}) : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.edge_set_iterator<["all"]>]>]>

            %0 = subop.create !subop.result_table<[eid_i64 : i64, eprop_i64 : i64, nid_i64 : i64, nprop_i64 : i64, ncount_index : index]>
            subop.materialize %streamB6 {@edges::@id => eid_i64, @edges::@property => eprop_i64, @nodes::@id => nid_i64, @nodes::@property => nprop_i64, @graph::@numVertices => ncount_index}, %0 : !subop.result_table<[eid_i64 : i64, eprop_i64 : i64, nid_i64 : i64, nprop_i64 : i64, ncount_index : index]>
            %res = subop.create_from ["eid", "eprop", "nid", "nprop", "ncount"] %0 : !subop.result_table<[eid_i64 : i64, eprop_i64 : i64, nid_i64 : i64, nprop_i64 : i64, ncount_index : index]> -> !subop.local_table<[eid_i64 : i64, eprop_i64 : i64, nid_i64 : i64, nprop_i64 : i64, ncount_index : index], ["eid", "eprop", "nid", "nprop", "ncount"]>
            subop.execution_group_return %res : !subop.local_table<[eid_i64 : i64, eprop_i64 : i64, nid_i64 : i64, nprop_i64 : i64, ncount_index : index], ["eid", "eprop", "nid", "nprop", "ncount"]>
        
        } -> !subop.table<[eid_i64n : i64, eprop_i64n : i64, nid_i64n : i64, nprop_i64n : i64, ncount_indexn : index]>
        subop.set_result 0 %subop_result : !subop.table<[eid_i64n : i64, eprop_i64n : i64, nid_i64n : i64, nprop_i64n : i64, ncount_indexn : index]>
        return
    }
}