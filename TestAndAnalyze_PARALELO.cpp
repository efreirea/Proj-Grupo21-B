#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include "T2.h"



void buildCommandLine(char *commandline, const char* hostFileName,char *outputFileName,int threadNumber,const char *inputFileName,int tipoImagem);
void concatFileName(char *dest, const char *fileName, int thread,int bloco);
int main(int argc, char **argv){
	int k=0,i=0,j=0,t=0,b=0,count=0,NUM_VEZES=10,NUM_ARQUIVOS=0,execute=1,THREADS_NUM=1,BLOCOS_NUM=1;
	
	char auxIMGNAME[101];
	char filenames[10][101];
	int tipoImagens[10];
	char hostFileNames[10][101];
	char outputFilenames[10][101];
	char arquivosAnalise[100][101];
	char commandline[1000];
	double valoresDeTempo[20];
	int auxBLOCOS=0,auxTHREADS=0,threadsValues[10],blocosValues[10];
	double auxTIME=0.0,sum=0.0,var=0.0,dp=0.0,media=0.0; 
	FILE *fp=NULL; 

	/*
	usage:
		nome do arquivo de host
		n de threads diferentes que vamos usar => le os valores em si
		nome da imagem
		tipo da imagem
		n vezes

	*/
	scanf("%d %d\n",&NUM_ARQUIVOS,&NUM_VEZES);
	scanf("%d",&THREADS_NUM);
	for(i=0;i<THREADS_NUM;++i){
		scanf(" %d",&(threadsValues[i]));
		//printf("%d\n",threadsValues[i]);
	}
	scanf("\n");
	scanf("%d",&BLOCOS_NUM);
	for(i=0;i<BLOCOS_NUM;++i){
		scanf(" %d",&(blocosValues[i]));
		//printf("%d\n",threadsValues[i]);
	}
	scanf("\n");
	if(argc==2){
		if(strcmp(argv[1],"-nE")==0){
			execute=0;
		}
	}
	if(execute==1){
		/*Caso deseja-se que os metodos sejam executados, Leem-se os nomes das imagens
		(arazenando en filenames) e os arquivos em que serao armazenados os tempos
		(outputfilenames)
		*/
		for(i=0;i<NUM_ARQUIVOS;++i){
			scanf("%s\n",&(filenames[i][0]));
			scanf("%d\n",&(tipoImagens[i]));
			printf("%s\n",filenames[i]);
		}
		for(i=0;i<BLOCOS_NUM;++i){
			scanf("%s\n",&(hostFileNames[i][0]));
		}

		for(i=0;i<NUM_ARQUIVOS;++i){
			scanf("%s\n",&(outputFilenames[i][0]));
		}
	}else{
		/*Caso deseja-se apenas realizar as analises, leem-se os nomes dos arquivos 
		onde estao armazedanos os tempos
		*/
		for(i=0;i<NUM_ARQUIVOS;++i){
			scanf("%s\n",&(arquivosAnalise[i][0]));
			k++;
		}
	}

	
	if(execute==1){
		
		for(i=0;i<NUM_ARQUIVOS;++i){
			//executando paralelo n vezes
			for(b=0;b<BLOCOS_NUM;++b){
				for(t=0;t<THREADS_NUM;++t){
					/*Para a execucao com threads, padronizou-se o nome de arquivos de saida como:
					[NUM_THREAD]_[NOME_ARQUIVO_DE_SAIDA]
					*/
					concatFileName(&(arquivosAnalise[k++][0]),outputFilenames[i],threadsValues[t],blocosValues[b]);
					for(j=0;j<NUM_VEZES;++j){
						buildCommandLine(commandline,hostFileNames[b],arquivosAnalise[k-1],threadsValues[t],filenames[i],tipoImagens[i]);	
						system(commandline);
					}
				}
			}
		//	executando serial n vezes
			// strcpy(arquivosAnalise[k++],outputFilenames[i]);
			// for(j=0;j<NUM_VEZES;++j){
				
			// 	buildCommandLine(commandline,NULL,outputFilenames[i],-1,filenames[i],tipoImagens[i]);	
			// 	system(commandline);
			// }
		}
	}
	/*Processamento dos arquivos de saida dos programas SERIAL_NAME e PARALLEL_NAME para o calculo
	da media e desvio padrao
	*/
	// for(i=0;i<k;++i){
	// 	//Abertura do arquivo a ser processado
	// 	fp= fopen(arquivosAnalise[i],"r");
	// 	if(fp==NULL){
	// 		fprintf(stderr,"Erro ao abrir arquivo: ");
	// 		fprintf(stderr,"%s",arquivosAnalise[i]);
	// 		fprintf(stderr,"\n");
	// 	}else{
	// 		sum=0;
	// 		var=0;
	// 		count=0;
	// 		//leitura dos dados
	// 		while(fscanf(fp,"%s\t%d\t%d\t%lf\n",auxIMGNAME,&auxBLOCOS,&auxTHREADS,&auxTIME)!=EOF){
			
	// 			sum+=auxTIME;
	// 			valoresDeTempo[count++]=auxTIME;
				
	// 		}
	// 		fclose(fp);
	// 		//processamento
	// 		fp=NULL;
	// 		media=sum/(count*1.0);
	// 		for(j=0;j<count;++j){
	// 			var+=(valoresDeTempo[j]-media)*(valoresDeTempo[j]-media);
	// 		}
	// 		var/=(count*1.0);
	// 		dp=sqrt(var);

	// 		printf("%s\t%d\t%d\t%lf\t%lf\n",auxIMGNAME, auxBLOCOS,auxTHREADS,media,dp);
	// 	}		
	// }

	return EXIT_SUCCESS;
}
void concatFileName(char *dest, const char *fileName, int thread,int bloco){
	char strThread[10];
	char strBloco[10];
	
	strThread[0]='\0';
	strBloco[0]='\0';
	sprintf(strThread, "%d", thread);
	sprintf(strBloco, "%d", bloco);

	dest[0]='\0';
	strcat(dest,strBloco);
	strcat(dest,"_");
	strcat(dest,strThread);
	strcat(dest,"_");
	strcat(dest,fileName);
}
void buildCommandLine(char *commandline, const char* hostFileName,char *outputFileName,int threadNumber,const char *inputFileName,int tipoImagem){
	/**
	Funcao responsavel por contenar strings para construir a linha de comando necessaria para executar
	os programas SERIAL_NAME e PARALLEL_NAME. Para detalhes da sintaxe, consultar arquivo README
	*/
	char strThread[10];
	char strTipoImg[2];
	strThread[0]='\0';
	strTipoImg[0]='\0';
	commandline[0]='\0';
	sprintf(strThread, "%d", threadNumber);
	sprintf(strTipoImg, "%d", tipoImagem);
	
	if(threadNumber<0){
		
		strcat(commandline,SERIAL_NAME);
		strcat(commandline," ");
		strcat(commandline,inputFileName);
		strcat(commandline," ");
		strcat(commandline,strTipoImg);
	}else{
	//	strcat(commandline,"mpirun -np 17 --hostfile ");
		strcat(commandline,"mpirun -np 4 --hostfile ");
		strcat(commandline,hostFileName);
		strcat(commandline," ");
		strcat(commandline,PARALLEL_NAME);
		strcat(commandline," ");
		strcat(commandline,inputFileName);
		strcat(commandline," ");
		strcat(commandline,strTipoImg);
		strcat(commandline," ");
		strcat(commandline,strThread);

	}
	strcat(commandline," >> ");
	strcat(commandline,outputFileName);
	return;
}