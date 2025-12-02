module {
    func.func @main() {
        %subop_result = subop.execution_group (){

            %graph = graph.subop.create_graph !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]>
            %graph_scan = graph.subop.scan_graph %graph : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]> @nodes::@set({type = !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>}), @edges::@set({type = !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>})
            %vertex_count = graph.node_count %graph_scan, %graph -> @graph::@numVertices({type = index}) : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]>
            %node_refs = subop.nested_map %vertex_count [@nodes::@set] (%arg0, %arg1){
                %node_stream = graph.subop.scan_node_set %arg1 : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]> @nodes::@ref({type = !graph.node_ref<[node_id_0 : i64],[incoming_0 : !graph.edge_set<[incoming_it_0 : !graph.graph_set_iterator<["incoming"]>]>],[outgoing_0 : !graph.edge_set<[outgoing_it_0: !graph.graph_set_iterator<["outgoing"]>]>],[rank_0 : f64, nextRank_0 : f64, l_0 : i32]>})
                tuples.return %node_stream : !tuples.tuplestream
            }
            %gather_outgoing = subop.gather %node_refs @nodes::@ref {outgoing_0 => @nodes::@outgoing({type = !graph.edge_set<[outgoing_it_0 : !graph.graph_set_iterator<["outgoing"]>]>})}
            %gather_rank = subop.gather %gather_outgoing @nodes::@ref { rank_0 => @nodes::@rank({type = f64}) }
            %gather_l = subop.gather %gather_rank @nodes::@ref { l_0 => @nodes::@l({type = i32}) }
            %outgoing_refs = subop.nested_map %gather_l [@nodes::@outgoing] (%arg0, %arg1){
                %edge_stream = graph.subop.scan_edge_set %arg1 : !graph.edge_set<[outgoing_it_0 : !graph.graph_set_iterator<["outgoing"]>]> @edges::@ref({type = !graph.edge_ref<[edge_id_0 : i64],[from_0: !graph.node_ref<[node_id_1 : i64],[incoming_1 : !graph.edge_set<[incoming_it_1 : !graph.graph_set_iterator<["incoming"]>]>],[outgoing_1 : !graph.edge_set<[outgoing_it_1 : !graph.graph_set_iterator<["outgoing"]>]>],[rank_1 : f64, nextRank_1 : f64, l_1 : i32]>],[to_0 : !graph.node_ref<[node_id_2 : i64],[incoming_2 : !graph.edge_set<[incoming_it_2 : !graph.graph_set_iterator<["incoming"]>]>],[outgoing_2 : !graph.edge_set<[outgoing_it_2 : !graph.graph_set_iterator<["outgoing"]>]>],[rank_2 : f64, nextRank_2 : f64, l_2 : i32]>],[eprop_0 : i64]>})
                subop.reduce %edge_stream @node::@ref [@graph::@numVertices] ["rank_0", "l_0"] ([%totalVertices],[%currRank, %currL]) {
                    tuples.return %currRank, %currL : f64, i32
                }
                tuples.return %edge_stream : !tuples.tuplestream
            }

            %result_table = subop.create !subop.result_table<[id0:i32, rank0 : f64, l0 :i32]>
            %local_table = subop.create_from ["id","rank","l"] %result_table : !subop.result_table<[id0:i32,rank0 : f64, l0 :i32]> -> !subop.local_table<[id0:i32,rank0 : f64, l0 :i32],["id","rank","l"]>
            subop.execution_group_return %local_table : !subop.local_table<[id0:i32,rank0 : f64, l0 :i32],["id","rank","l"]>
        } -> !subop.local_table<[id0:i32,rank0 : f64, l0 :i32],["id","rank","l"]>
        subop.set_result 0 %subop_result  : !subop.local_table<[id0:i32,rank0 : f64, l0 :i32],["id","rank","l"]>
        return
    }
}