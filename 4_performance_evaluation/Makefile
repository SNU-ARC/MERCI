CXX=g++-9
CXXFLAGS=-I. -g -O3 -W -std=c++17 -lstdc++fs -fopenmp  -march=native -ffast-math

all: eval_remap_only eval_baseline eval_merci
eval_baseline: src/eval_baseline.cc
	$(CXX) -o bin/eval_baseline $< $(CXXFLAGS)

eval_merci: src/eval_merci.cc
	$(CXX) -o bin/eval_merci $< $(CXXFLAGS)

eval_remap_only: src/eval_remap_only.cc
	$(CXX) -o bin/eval_remap_only $< $(CXXFLAGS)