all:

compile: compSeq compParalMPI compParalCuda compTest

compSeq:
	cd ./sequential;\
	 rm -f CMakeCache.txt;\
	 rm -f CMakeFiles;\
	 rm -f cmake_install.cmake;\
	 rm -f DisplayImage;\
	 rm -f Makefile;\
	 cmake .;\
	 make
	
compParalMPI:
	cd parallel;\
	 mpic++  parallel_MPI.cpp -g  -o parallel_MPI `pkg-config --libs opencv` -fopenmp
compParalCuda:
	cd parallel;\
	# mpic++  parallel_CUDA.cu -g  -o parallel_CUDA `pkg-config --libs opencv`
compTest:
	g++ -g TestAndAnalyze_SEQUENCIAL.cpp -o testAndAnalyse_SEQ
	g++ -g TestAndAnalyze_PARALELO.cpp -o testAndAnalyse_PAR
	g++ -g TestAndAnalyze_CUDA.cpp -o testAndAnalyse_CUDA
	