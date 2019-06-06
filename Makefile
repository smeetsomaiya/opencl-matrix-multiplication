all:
	g++ main.cpp -lOpenCL -o main.o -std=c++11 -Wall
clean:
	rm -r *.o
