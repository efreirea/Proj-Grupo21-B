#ifndef _T2_H_
#define _T2_H_

//Definicao dos nomes dos executaveis gerados apos a compilacao.
//necessario pois a funcao buildCommandLine em analyzer usa tal informacao
#define PARALLEL_NAME "./parallel/parallel_MPI"
#define PARALLEL_NAME_CUDA "./parallel/parallel_CUDA"
#define SERIAL_NAME "./sequential/sequential"

#endif
