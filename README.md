Computer Architecture II
Programming Assignment 4

Smeet Somaiya
1213422780

Files provided:
		main.cpp
		matrix_mul.cl

How to compile?
Run the Makefile with
		make all

This basically calls the g++ compiler with this command

		g++ main.cpp -lOpenCL -o main.o -std=c++11
This assumes that the OpenCl header is in the default path. For any other path, the command will have an additional argument, as follows

		g++ -I/path/to/headers/ main.cpp -lOpenCL -o main.o -std=c++11

Clean the build files
		make clean

How to run?
Run the following command in the terminal to run the compiled program
		./main.o
