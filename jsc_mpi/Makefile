#make sure to set the MPICH_CXX and LD_LIBRARY_PATH to the path of your g++4.9.x compiler
CC=mpicxx
CCFLAGS=-lhdf5 -fopenmp -std=c++11 -Wall
CCDEBUG=-O1 -pg -g
CCRUN=-O3

SRC=*.cpp
OUT=dbscan

all:
	$(CC) $(CCRUN) $(CCFLAGS) $(SRC) -o $(OUT)

debug:
	$(CC) $(CCDEBUG) $(CCFLAGS) $(SRC) -o $(OUT)

clean:
	rm $(OUT)
