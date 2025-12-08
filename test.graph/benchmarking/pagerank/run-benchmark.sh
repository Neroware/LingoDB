#!/usr/bin/bash

rm -f *.log

readonly GRAPH_DIR="../../../test.graph/"
readonly RUN_MLIR="../../../build/run-mlir"

# Generate PageRank runtimes 
for i in $(seq 1 $1); do 
    $RUN_MLIR $GRAPH_DIR/test/pagerank.mlir | grep pagerank.mlir >> pagerank_graph.log
    $RUN_MLIR ../../../test/lit/SubOp/pagerank.mlir | grep pagerank.mlir >> pagerank_classic.log
done

# Compile and run the C++ implementation
g++ -o $GRAPH_DIR/cpp/pagerank/pagerank $GRAPH_DIR/cpp/pagerank/pagerank.cpp
for i in $(seq 1 $1); do 
    $GRAPH_DIR/cpp/pagerank/pagerank | grep pagerank.cpp >> pagerank_cpp.log
done