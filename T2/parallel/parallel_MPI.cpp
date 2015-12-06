#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <cmath>

//#define NTHREADS 3
#define MASTER 0
#define TAG_QNTD_OFFSET 1
#define TAG_DADOS_INITIAL 2
#define TAG_DADOS_RESULT 3



using namespace std;
using namespace cv;

void smooth(unsigned char *img,unsigned char *output,int qntLinhas,int qntCols,int size_com_borda,int offset,int order);
void diffTimeSpec(struct timespec start,struct timespec end,struct timespec *temp);


int main(int argc,char** argv){
	int processId;
	int totalProcesses;
	char procName[MPI_MAX_PROCESSOR_NAME];
	int procNameLength;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &processId);
	MPI_Comm_size(MPI_COMM_WORLD,&totalProcesses);
	MPI_Get_processor_name(procName, &procNameLength);
	
	int NTHREADS = 3; // Inicializando com 3 Threads, mas sera alterado
	int NBLOCOS =  15; //Inicializando com 15 blocos, mas sera alterado
	//define a quantidade de blocos de acordo com o numero de processos gerados. Tal numero esta relacionado com a quantidade de ghosts no arquivo de host
	MPI_Comm_size(MPI_COMM_WORLD,&NBLOCOS);
	//elimina um noh, pois eh o master. EM nossa implementacao, o master nao realiza processamento da imagem
	NBLOCOS--;
	if(argc>=4){
		NTHREADS=atoi(argv[3]);
		//cout<<NTHREADS<<endl;
	}
	omp_set_dynamic(0);
	//seta o nro de threads que iremos utilizar. Este numero corresponde ao maximo de threads permitidas
	omp_set_num_threads(NTHREADS);

	
	
	int qntLines =0;
	int sendvInitial[3]; // buffer de envio. Contem tres inteiros utilizados para controle, que sao: quantidade
		//sendvnitial[0] -> quantidade de linhas
		//sendvnitial[1] -> Offset, ou seja, a parti de qual linha deve iniciar o proessamento
		//sendvnitial[2] -> Quantidade de colunas

	int recvInitial[3];  // buffer para receber os dados de sendvInitial.
	unsigned char **partitionedBlocksSend; //bufferes para envio dos dados em si, ou seja, os dados da imagem
	unsigned char *partitionedBlockRcv;    //bufferes para recebimento dos dados
	unsigned char *output; 
	int *offsets; // vetor que armazenara no MASTER os offsets de cada bloco
	int *qntLinhas; // vetor que armazenara no MASTER as quantiades de linha que cada bloco processara
	int border = 2; //relativo ao tamanho do kernel utilizado. Valor 2 deve ser utilizado para  kernel de tamanho  5
	//noh master que disponibiliza recursos para as outras
	if (processId == MASTER){
		//Abertura da imagem
		
		//condicao para que nao execute apenas no master
		if(NBLOCOS<=0){
			cout <<"E necessario pelo menos um processo alem do master"<<endl;
			return -1;

		}


		offsets = (int *) malloc(NBLOCOS*sizeof(int));
		qntLinhas = (int *) malloc(NBLOCOS*sizeof(int));
		partitionedBlocksSend = (unsigned char **) malloc(NBLOCOS*sizeof(unsigned char*));
		Mat image;
		if(argc!=4){
			printf("entrar com nome do arquivo, Tipo de imagem e numero de threads\n");
			return -1;
		}
		int tipoImagem = atoi(argv[2]);
		//abrindo a imagem
		if(tipoImagem == 1){ //caso seja RGB
			image=imread(argv[1],CV_LOAD_IMAGE_COLOR);
		}else if(tipoImagem==2){ //caso seja grayScale
			image=imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
		}
		
		if(!image.data){
			printf("No data read.\n");
			return -1;
		}
		//armazena as dimensoes originais (sem borda) das imagens
		int rowMat = image.rows;
		int colsMat = image.cols;
		int deptMat = image.depth();
		//cria a imagem de saida
		Mat outputMat(rowMat,colsMat,deptMat);
		

		struct timespec begin, end,result;
		double time_spent;	
		
		clock_gettime(CLOCK_REALTIME,&begin);
		
		int channelNum= image.channels();
		Mat channels[3];
		//inserindo borda na imagem
		copyMakeBorder(image,image,border,border,border,border,BORDER_CONSTANT,Scalar(0,0,0));
		Mat outB(image.rows-border*2,image.cols-border*2,image.depth());
		channels[0]=outB;
		if(channelNum==3){ //caso seja RGB, cria os canas adicionais
			Mat outG(image.rows-border*2,image.cols-border*2,image.depth());
			Mat outR(image.rows-border*2,image.cols-border*2,image.depth());
			channels[1]=outG;
			channels[2]=outR;
		}
		
		//divisao do trabalho
		sendvInitial[2]=image.cols; //tamanho da coluna com a bordacom a borda
		int blocosPorCanal[3]; //armazena quantos blocos cada canal possui
		int counterModBlocos = NBLOCOS%channelNum;
		int *vetorIdCanal = (int *)malloc(NBLOCOS*sizeof(int)); // vetor que indica qual canal cada bloco esta processando. usado para reconstruir a imagem a partir das saidas
		int it = 0 ;
		int maior = -1; // varaivel para calcular a maior quantidade de linha que os blocos estao processando, de modo a poder utilizar apra alocar memoria
		//percorre os canais
		for (int c=0;c<channelNum;c++){
			//realiza a divisao da imagem em blocos
			if(counterModBlocos>0){ //realiza a distribuicao das blocos restantes (caso a divisao nao seja exata)
				counterModBlocos--;
				blocosPorCanal[c]=floor(NBLOCOS/channelNum)+1;	
			}else{
				blocosPorCanal[c]=floor(NBLOCOS/channelNum);	
			}
			
			int counterMod = (image.rows-2*border)%blocosPorCanal[c]; //sem borda
			//realiza a divisao para gerar as quantidades de linhas em cada canal
			for(int i=0;i<blocosPorCanal[c];++i){
				vetorIdCanal[it]=c;	
				if(counterMod>0){//realiza a distribuicao das linhas restantes (caso a divisao nao seja exata)
					counterMod--;
					qntLines=floor((image.rows-2*border)/blocosPorCanal[c])+1;	
				}else{
					qntLines=floor((image.rows-2*border)/blocosPorCanal[c]);	
				}
				
				sendvInitial[0] = qntLines; //envia qnt de linhas
				qntLinhas[it]=qntLines;
				maior = max(maior,qntLines); //calcula a maior qnt de linha
				//calcula o offset que o determinado bloco deve ter, ou seja, a aprtir de que linha deve processar
				if(i<=(image.rows-2*border)%blocosPorCanal[c]){
					sendvInitial[1] = i*floor((image.rows-2*border)/blocosPorCanal[c]) +i;
				}else{
					sendvInitial[1] = i*floor((image.rows-2*border)/blocosPorCanal[c]) + (image.rows-2*border)%blocosPorCanal[c];
				}
				offsets[it]=sendvInitial[1];
				//preenche o vetor de dados. separa efetivamente cada bloco, preenhendo um vetor com pixel correspondenntes
				partitionedBlocksSend[it]=(unsigned char*)malloc(((qntLines)*(image.cols)+2*border*(image.cols))*sizeof(unsigned char));
				int p=0;

				for(int j=offsets[it]+border-border;j<offsets[it]+border+qntLines+border;j++){
					for(int k=0;k<image.cols;k++){
						if(channelNum==3){
							//sum+=img.at<Vec3b>(i+k,j+l).val[c]; 
							partitionedBlocksSend[it][p++]=image.at<Vec3b>(j,k).val[c];
						}else{
							partitionedBlocksSend[it][p++]=image.at<uchar>(j,k);	//TODO: verificar como pegar do canal especifico						
						}
					}

				}
				//envia os dados para os processos
				MPI_Send(sendvInitial, 3, MPI_INT, it+1,TAG_QNTD_OFFSET, MPI_COMM_WORLD);
				MPI_Send(partitionedBlocksSend[it], p, MPI_UNSIGNED_CHAR, (it)+1,TAG_DADOS_INITIAL, MPI_COMM_WORLD);
				free(partitionedBlocksSend[it]);
				it++;
			}
		}
	//	cout<<"it: "<<it<<endl;
		int colsReais = image.cols-2*border;
		int y=0;

		partitionedBlockRcv = (unsigned char *)malloc(maior*colsReais*sizeof(unsigned char)) ; //aloca em funcao do maior
		
		//espera o envio da resposta dos blocos
		for(int i=0;i<NBLOCOS;i++){
			//cout<< "Retorno do Recv: ";
			int size_sem_borda = qntLinhas[i]*(colsReais);
			y=0;
			MPI_Recv(partitionedBlockRcv, size_sem_borda, MPI_UNSIGNED_CHAR,i+1, TAG_DADOS_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//recebendo as partes processadas da imagem e montando a imagem final
			
			for(int l=offsets[i];l<qntLinhas[i]+offsets[i];l++){
				//cout<<"l: "<<l<<endl;
				for(int k=0;k<colsReais;k++){
				//	cout<<(int)partitionedBlockRcv[y]<< " ";
					if(channelNum==3){
				//		outputMat.at<Vec3b>(l,k).val[vetorIdCanal[i]]=partitionedBlockRcv[y++];
						channels[vetorIdCanal[i]].at<uchar>(l,k)=(uchar)partitionedBlockRcv[y++];
					//	cout<<vetorIdCanal[i]<<endl;
					}else{
						outputMat.at<uchar>(l,k)=partitionedBlockRcv[y++];		
					}
					//cout<<" k: "<<k;

				}
				//cout<<endl;
			}
			//cout<<endl;


		}
		if (channelNum==3){
			merge(channels,channelNum,outputMat);
		}	
		clock_gettime(CLOCK_REALTIME,&end);
		
		//Calculo tempo de execucao
		diffTimeSpec(begin, end, &result);
		//time_spent=((double) difftime(end.tv_sec,begin.tv_sec))+(result.tv_nsec*1.0/1000000000.0);
		time_spent=((double) result.tv_sec)+(result.tv_nsec*1.0/1000000000.0);
		
		// namedWindow("Orginal",WINDOW_NORMAL);
		// namedWindow("Resultado",WINDOW_NORMAL);
		// imshow("Original",image);
		// imshow("Resultado",outputMat);
		// waitKey(0);

	//	cout << "Nome imagem: "<< argv[1] <<endl;
		std::string inFileName(argv[1]);
		cout<<inFileName<<"\t"<<NBLOCOS<<"\t"<<NTHREADS<<"\t"<<time_spent<<endl;
		std::string outFileName  = inFileName.substr(0,inFileName.find_last_of("."));
		outFileName += ".ppm";
		imwrite(outFileName,outputMat);
		//imwrite("../canal0.ppm",channels[0]);
		//imwrite("../canal1.ppm",channels[1]);
		//imwrite("../canal2.ppm",channels[2]);
		//imwrite("OIEEEEEEEE.ppm",image);
		free(partitionedBlockRcv);
		free(partitionedBlocksSend);
		free(offsets);
		free(qntLinhas);
		free(vetorIdCanal);
		
	}
	if(processId>MASTER){ //Em cada outro processo que nao seja o master
		//recebe meta dados vindos do MASTER
		MPI_Recv(recvInitial, 3, MPI_INT,MASTER, TAG_QNTD_OFFSET, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//cout<<"RECEBEU - "<<"processID: "<<processId<<" qntLinhas :"<<recvInitial[0]<<" offset: "<<recvInitial[1]<<endl;
		
		int qntLinhasRv = recvInitial[0]; //sem borda
		int offsetRv = recvInitial[1]; //sem borda
		int qntcolsRv = recvInitial[2]; // com borda

		int size_com_borda = qntLinhasRv*qntcolsRv+2*border*qntcolsRv;
		int size_sem_borda = qntLinhasRv*(qntcolsRv-2*border);
		

		//OS metadados sao utilizados para alocar recursos para que, agora, receba o dado em si
		partitionedBlockRcv = (unsigned char *)malloc(size_com_borda*sizeof(unsigned char));
		MPI_Recv(partitionedBlockRcv, size_com_borda,
			MPI_UNSIGNED_CHAR,MASTER, TAG_DADOS_INITIAL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		output = (unsigned char*) malloc(size_sem_borda*sizeof(unsigned char));
		//realza o processamento
		smooth(partitionedBlockRcv,output,qntLinhasRv,qntcolsRv,size_com_borda,offsetRv,border);
		//unsigned char *img,unsigned char *output,int qntLinhas,int qntCols,int size_com_borda,int offset)
		MPI_Send(output, size_sem_borda, MPI_UNSIGNED_CHAR, MASTER, TAG_DADOS_RESULT, MPI_COMM_WORLD);

		free(output);
		free(partitionedBlockRcv);

	}


	MPI_Finalize();
}

void diffTimeSpec(struct timespec start,struct timespec end,struct timespec *temp)
{
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp->tv_sec = end.tv_sec-start.tv_sec-1;
		temp->tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp->tv_sec = end.tv_sec-start.tv_sec;
		temp->tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return;
}

//retorna a posicao no voetor correspondente a uma coordenada i,j fornecida
inline
int posCalc(int i, int j,int qntCols){
	return (i*qntCols+j);
}
//retorna a coordenada J correspondente a uma posicao no vtor
inline
int idxCalcJ(int pos,int qntCols){
	return (pos%qntCols);
}
//retorna a coordenada I correspondente a uma posicao no vtor
inline
int idxCalcI(int pos,int qntCols){
	return (floor(pos*1.0/qntCols));
}
//relaiza o calculo da media
inline
unsigned char calculaMedia(unsigned char *in,unsigned char *out,int qntLinhas,int qntCols,int p,int border){
	int sum=0;
	int i= idxCalcI(p,qntCols);
	int j= idxCalcJ(p,qntCols);
	for(int k=-border;k<=border;k++){
		for(int l=-border;l<=border;l++){
			sum+=in[posCalc(i+k,j+l,qntCols)] ;
		}
	}

	return (unsigned char)(sum/pow(border*2+1,2));	

}

void smooth(unsigned char *img,unsigned char *output,int qntLinhas,int qntCols,int size_com_borda,int offset,int border){
	//TODO: colocar bordas de acordo com necessidade
	int sum;
//	Mat imgWithBorder(img,Rect(border,border,img.rows,img.cols));
	int lastPos = posCalc(qntLinhas+2*border-1,qntCols-1,qntCols)+1; 
	int k=0;
	int auxi,auxj;
	unsigned char auxDebug;
	//cout<< "Retorno do smooth: ";
	
	#pragma omp parallel //shared(output)
	{
		#pragma omp for schedule(dynamic) private(auxi,auxj,auxDebug)
			//processa o vetor. Como a matriz foi colocada em formato de vetor, eh necessario um for so, mas fora utilizadas fucoes para converter os indces
			for(int i=posCalc(border,border,qntCols);i<lastPos;i++){
				auxi=idxCalcI(i,qntCols);
				auxj=idxCalcJ(i,qntCols);
				//cout<< auxi << " "<<auxj<<endl;
				if(auxi>=border && auxi<qntLinhas+2*border-border && auxj>=border && auxj<qntCols-border){ // se e um pixel valido da imagem
					auxDebug=calculaMedia(img,output,qntLinhas,qntCols,i,border);
					output[posCalc(auxi-border,auxj-border,qntCols-2*border)]=auxDebug;
		
				}

				
			}

	}


}