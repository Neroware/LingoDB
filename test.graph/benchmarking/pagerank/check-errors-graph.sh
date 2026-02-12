#!/usr/bin/bash

rm -f *.log

readonly GRAPH_DIR="../../../test.graph/"
readonly RUN_MLIR="../../../build/run-mlir"

# Generate PageRank runtimes 
for i in $(seq 1 $1); do 
    $RUN_MLIR $GRAPH_DIR/test/pagerank.mlir | grep "1  | " >> pagerank_graph.log
done