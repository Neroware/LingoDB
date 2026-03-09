module {
    func.func @main() {
    	%subop_result = subop.execution_group (){
            
            %g = graph.subop.create_graph !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]> { testgraph = 0 : index }
            %g_scan = graph.subop.scan_graph %g : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]> @nodes::@set({type = !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>}), @edges::@set({type = !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>})
            %vx = subop.nested_map %g_scan [@nodes::@set] (%arg0, %arg1){
                %node_stream = graph.subop.scan_node_set %arg1 : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]> @nodes::@ref({type = !graph.node_ref<[node_id : i32],[incoming : !graph.edge_set<[incoming_it : !graph.graph_set_iterator<["incoming"]>]>],[outgoing : !graph.edge_set<[outgoing_it : !graph.graph_set_iterator<["outgoing"]>]>],[property : i64]>})
                tuples.return %node_stream : !tuples.tuplestream
            }
            %result_nodes = subop.gather %vx @nodes::@ref {node_id => @nodes::@id({type = i32})}
            %node_degrees = graph.node_degree %result_nodes, %g["all"] -> @graph::@degrees({type = i64}) : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]>

            %0 = subop.create !subop.result_table<[int32p0 : i32, int64p1 : i64]>
            subop.materialize %result_nodes {@nodes::@id => int32p0, @graph::@degrees => int64p1}, %0 : !subop.result_table<[int32p0 : i32, int64p1 : i64]>
            %res = subop.create_from ["int32", "int64"] %0 : !subop.result_table<[int32p0 : i32, int64p1 : i64]> -> !subop.local_table<[int32p0 : i32, int64p1 : i64], ["int32", "int64"]>
            subop.execution_group_return %res : !subop.local_table<[int32p0 : i32, int64p1 : i64], ["int32", "int64"]>
        
        } -> !subop.table<[int32n0 : i32, int64n1 : i64]>
        subop.set_result 0 %subop_result : !subop.table<[int32n0 : i32, int64n1 : i64]>
        return
    }
}