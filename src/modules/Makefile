#++++++++++ USER defined var +++++++++++
IDIR = /home/rketka/gitlab/plumed2/src/
#+++++++++++++++++++++++++++++++++++++++

# Uncomment to include generic makefile modules for PLUME (DeepCV source code must be placed in PLUMED's src/ folder)
# include ../maketools/make.module

CXX = g++
# -g for debugging info
# -Wall for showing compiler warnings
CXXFLAGS = -g -Wall -fPIC -std=c++11 -O2 -I$(IDIR) 

.PHONY: all clean

all: hello

hello: 
	@echo "Automatic compilation of DeepCV lib for PLUMED"

deepcv.o:
	$(CXX) $(CXXFLAGS) -c -o deepcv.o deepcv.cpp

deepcv.so: deepcv.o
	$(CXX) -fPIC -shared -o deepcv.so deepcv.o

compile: deepcv.so
	@echo "Compilation is completed"

clean:
	rm -f deepcv.o deepcv.so

