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
	@echo "" 
	@echo "Automatic compilation of DeepCV lib for PLUMED."

object:
	$(CXX) $(CXXFLAGS) -c -o deepcv.o deepcv.cpp

dynamic: object
	$(CXX) -fPIC -shared -o libdeepcv.so deepcv.o
	@echo "Dynamic library libdeepcv.so is successfully built!" 

static: object
	ar crv libdeepcv.a deepcv.o
	@echo "Static library libdeepcv.a is successfully built!" 

compile: hello dynamic
	@echo "" 

clean:
	rm -f deepcv.o libdeepcv.so libdeepcv.a

