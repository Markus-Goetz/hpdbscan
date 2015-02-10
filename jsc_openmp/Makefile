CC=g++49
CCFLAGS=-fopenmp -std=c++11
CCDEBUG=-O1 -pg -g
CCRUN=-O3

SRC=rules.cpp points.cpp space.cpp hpdbscan.cpp main.cpp
OUT=dbscan
OBJ=dbscan.o
LIB=libdbscan.so

all:
	$(CC) $(CCRUN) $(CCFLAGS) $(SRC) -o $(OUT)
	
lib:
	$(CC) -c -fPIC $(CCRUN) $(CCFLAGS) main.cpp -o $(OBJ)
	$(CC) -shared -Wl,-soname,$(LIB) -lgomp -o $(LIB) $(OBJ)
	rm $(OBJ)

debug:
	$(CC) $(CCDEBUG) $(CCFLAGS) $(SRC) -o $(OUT)

clean:
	rm $(OUT)
