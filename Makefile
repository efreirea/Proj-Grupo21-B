all:

compile: compSeq compParalMPI compParalCuda compTest


compSeq:
	cd ./T2/sequential;\
	 rm -f CMakeCache.txt;\
	 rm -f CMakeFiles;\
	 rm -f cmake_install.cmake;\
	 rm -f DisplayImage;\
	 rm -f Makefile;\
	 cmake .;\
	 make
	
compParalMPI:
	cd ./T2/parallel;\
	 mpic++  parallel_MPI.cpp -g  -o parallel_MPI /usr/local/lib/libopencv_calib3d.so /usr/local/lib/libopencv_contrib.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_features2d.so /usr/local/lib/libopencv_flann.so /usr/local/lib/libopencv_gpu.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_legacy.so /usr/local/lib/libopencv_ml.so /usr/local/lib/libopencv_nonfree.so /usr/local/lib/libopencv_objdetect.so /usr/local/lib/libopencv_ocl.so /usr/local/lib/libopencv_photo.so /usr/local/lib/libopencv_stitching.so /usr/local/lib/libopencv_superres.so /usr/local/lib/libopencv_ts.a /usr/local/lib/libopencv_video.so /usr/local/lib/libopencv_videostab.so /usr/lib/x86_64-linux-gnu/libXext.so /usr/lib/x86_64-linux-gnu/libX11.so /usr/lib/x86_64-linux-gnu/libICE.so /usr/lib/x86_64-linux-gnu/libSM.so /usr/lib/libGL.so /usr/lib/x86_64-linux-gnu/libGLU.so ibs opencv -lm -fopenmp
compParalCuda:
	cd ./Proj/parallel;\
	nvcc parallel_CUDA.cu -o parallel_CUDA `pkg-config --libs opencv`
compTest:
	g++ -g TestAndAnalyze_SEQUENCIAL.cpp -o testAndAnalyse_SEQ
	g++ -g TestAndAnalyze_PARALELO.cpp -o testAndAnalyse_PAR
	g++ -g TestAndAnalyze_CUDA.cpp -o testAndAnalyse_CUDA
	