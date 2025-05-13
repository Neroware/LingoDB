// ./build/mlir-db-opt test.snippets/graph/test.graph.mlir

module {
    func.func @main() {
    	%subop_result = subop.execution_group (){
            
            %g = graph.subop.create_graph !graph.graph<[vx : !graph.node_set<[vx_it : !graph.node_set_iterator]>],[ex : !graph.edge_set<[ex_it : !graph.edge_set_iterator]>]>
            %g_scan = graph.subop.scan_graph %g : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.node_set_iterator]>],[ex : !graph.edge_set<[ex_it : !graph.edge_set_iterator]>]> @nodes::@set({type = !graph.node_set<[vx_it : !graph.node_set_iterator]>})
            %vx = subop.nested_map %g_scan [@nodes::@set] (%arg0, %arg1){
                %node_stream = graph.subop.scan_node_set %arg1 : !graph.node_set<[vx_it : !graph.node_set_iterator]> @nodes::@iter({type = !graph.node_ref<[node_id : index],[incoming : !graph.edge_set<[incoming_it : !graph.edge_set_iterator]>],[outgoing : !graph.edge_set<[outgoing_it : !graph.edge_set_iterator]>],[property : i64]>})
                tuples.return %node_stream : !tuples.tuplestream
            }
            
            %g_scan0 = graph.subop.scan_graph %g : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.node_set_iterator]>],[ex : !graph.edge_set<[ex_it : !graph.edge_set_iterator]>]> @edges::@set({type = !graph.edge_set<[ex_it : !graph.edge_set_iterator]>})
            %ex = subop.nested_map %g_scan0 [@edges::@set] (%arg0, %arg1){
                %edge_stream = graph.subop.scan_edge_set %arg1 : !graph.edge_set<[ex_it : !graph.edge_set_iterator]> @edges::@iter({type = !graph.edge_ref<[edge_id : index],[from : !graph.node_ref<[node_id1 : index],[incoming1 : !graph.edge_set<[incoming_it1 : !graph.edge_set_iterator]>],[outgoing1 : !graph.edge_set<[outgoing_it1 : !graph.edge_set_iterator]>],[property1 : i64]>],[to : !graph.node_ref<[node_id2 : index],[incoming2 : !graph.edge_set<[incoming_it2 : !graph.edge_set_iterator]>],[outgoing2 : !graph.edge_set<[outgoing_it2 : !graph.edge_set_iterator]>],[property2 : i64]>],[edge_prop : i64]>})
                tuples.return %edge_stream : !tuples.tuplestream
            }

            %0 = subop.create !subop.result_table<[int64p0 : i64]>
            %res = subop.create_from ["int64"] %0 : !subop.result_table<[int64p0 : i64]> -> !subop.local_table<[int64p0 : i64], ["int64"]>
            subop.execution_group_return %res : !subop.local_table<[int64p0 : i64], ["int64"]>
        
        } -> !subop.table<[int64n0 : i64]>
        // subop.set_result 0 %subop_result : !subop.table<[int64n0 : i64]>
        return
    }
}