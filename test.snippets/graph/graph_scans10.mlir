// ./build/mlir-db-opt test.snippets/graph/graph_scans10.mlir

module {
    func.func @main() {
    	%subop_result = subop.execution_group (){
            
            %g = graph.subop.create_graph !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]>
            %g_scan = graph.subop.scan_graph %g : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]> @nodes::@set({type = !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>}), @edges::@set({type = !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>})
            %vx = subop.nested_map %g_scan [@edges::@set] (%arg0, %arg1){
                %node_stream = graph.subop.scan_edge_set %arg1 : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]> @edges::@ref({type = !graph.edge_ref<[edge_id : i64],[from : !graph.node_ref<[node_id1 : i64],[incoming1 : !graph.edge_set<[incoming_it1 : !graph.graph_set_iterator<["incoming"]>]>],[outgoing1 : !graph.edge_set<[outgoing_it1 : !graph.graph_set_iterator<["outgoing"]>]>],[property1 : i64]>],[to : !graph.node_ref<[node_id2 : i64],[incoming2 : !graph.edge_set<[incoming_it2 : !graph.graph_set_iterator<["incoming"]>]>],[outgoing2 : !graph.edge_set<[outgoing_it2 : !graph.graph_set_iterator<["outgoing"]>]>],[property2 : i64]>],[edge_prop : i64]>})
                tuples.return %node_stream : !tuples.tuplestream
            }
            %result_edges = subop.gather %vx @edges::@ref {edge_id => @edges::@id({type = i64})}
            %result_nodes0 = subop.gather %result_edges @edges::@ref {from => @nodes::@from({type = !graph.node_ref<[node_id1 : i64],[incoming1 : !graph.edge_set<[incoming_it1 : !graph.graph_set_iterator<["incoming"]>]>],[outgoing1 : !graph.edge_set<[outgoing_it1 : !graph.graph_set_iterator<["outgoing"]>]>],[property1 : i64]>})}
            %result_nodes1 = subop.gather %result_nodes0 @nodes::@from {node_id1 => @nodes::@id({type = i64})}

            %0 = subop.create !subop.result_table<[int64p0 : i64]>
            subop.materialize %result_nodes1 {@nodes::@id => int64p0}, %0 : !subop.result_table<[int64p0 : i64]>
            %res = subop.create_from ["int64"] %0 : !subop.result_table<[int64p0 : i64]> -> !subop.local_table<[int64p0 : i64], ["int64"]>
            subop.execution_group_return %res : !subop.local_table<[int64p0 : i64], ["int64"]>
        
        } -> !subop.table<[int64n0 : i64]>
        subop.set_result 0 %subop_result : !subop.table<[int64n0 : i64]>
        return
    }
}