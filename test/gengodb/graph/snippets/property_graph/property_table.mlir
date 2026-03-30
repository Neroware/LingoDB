module {
    func.func @main() {
    	%subop_result = subop.execution_group (){
            
            %g = graph.subop.create_builtin_graph {builtin = propertygraph} : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]>
            %g_scan = graph.subop.scan_graph %g : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>]> @nodes::@set({type = !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]>}), @edges::@set({type = !graph.edge_set<[ex_it : !graph.graph_set_iterator<["all"]>]>})
            %vx = subop.nested_map %g_scan [@nodes::@set] (%arg0, %arg1){
                %node_stream = graph.subop.scan_node_set %arg1 : !graph.node_set<[vx_it : !graph.graph_set_iterator<["all"]>]> @nodes::@ref({type = !graph.node_ref<[node_id : i32],[incoming : !graph.edge_set<[incoming_it : !graph.graph_set_iterator<["incoming"]>]>],[outgoing : !graph.edge_set<[outgoing_it : !graph.graph_set_iterator<["outgoing"]>]>],[property : !graph.property_set<[prop_it : !graph.graph_set_iterator<["node"]>]>]>})
                tuples.return %node_stream : !tuples.tuplestream
            }
            %vx_id = subop.gather %vx @nodes::@ref { node_id => @nodes::@id({type = i32}) }
            %px = subop.gather %vx_id @nodes::@ref { property => @props::@set({type = !graph.property_set<[prop_it : !graph.graph_set_iterator<["node"]>]>}) }
            %props = subop.nested_map %px [@props::@set] (%arg0, %arg1){
                %prop_stream0 = graph.subop.scan_property_set %arg1 : !graph.property_set<[prop_it : !graph.graph_set_iterator<["node"]>]> @props::@refs({type = !graph.property_ref})
                %prop_stream1 = graph.cast_property_ref %prop_stream0 @props::@refs -> @props::@refsI32({type = !graph.typed_property_ref<[property_i32 : i32]>})
                tuples.return %prop_stream1 : !tuples.tuplestream
            }

            %result_props = subop.gather %props @props::@refsI32 {property_i32 => @props::@result({type = i32})}

            %0 = subop.create !subop.result_table<[int32p0 : i32, int32p1 : i32]>
            subop.materialize %result_props {@nodes::@id => int32p0, @props::@result => int32p1}, %0 : !subop.result_table<[int32p0 : i32, int32p1 : i32]>
            %res = subop.create_from ["int32", "int32"] %0 : !subop.result_table<[int32p0 : i32, int32p1 : i32]> -> !subop.local_table<[int32p0 : i32, int32p1 : i32], ["int32", "int32"]>
            subop.execution_group_return %res : !subop.local_table<[int32p0 : i32, int32p1 : i32], ["int32", "int32"]>
        
        } -> !subop.table<[int32n0 : i32, int32n1 : i32]>
        subop.set_result 0 %subop_result : !subop.table<[int32n0 : i32, int32n1 : i32]>
        return
    }
}