#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;




void cudaFatal(cudaError_t error)
{
	if (error != cudaSuccess) {
		fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

//retorna a posicao no voetor correspondente a uma coordenada i,j fornecida
__device__ int posCalc(int channel,int i, int j,int qntCols,int imgSize){
	return (imgSize*channel+i*qntCols+j);
}
//retorna o canal correspondente a uma posicao no vtor
__host__  int idxCalcC(int pos,int qntCols,int imgSize){
	return (floorf(pos*1.0/imgSize));
}


//retorna a coordenada J correspondente a uma posicao no vtor
__host__  int idxCalcJ(int pos,int qntCols,int imgSize){
	return (pos%qntCols);
}
//retorna a coordenada I correspondente a uma posicao no vtor
__host__ int idxCalcI(int pos,int qntCols,int imgSize,int channelNum){
	int qntLinhasPorCanal = (imgSize/qntCols);
	// return ((int)floor(pos*1.0/qntLinhasPorCanal))%qntLinhasPorCanal;

	return ((int)floor(pos*1.0/qntCols))%qntLinhasPorCanal;
}

__global__ void smooth(unsigned char *d_inImage,unsigned char *d_outImage,int *d_blocoQnt,int *d_blocoOffset,
	int imgSizeIn,int imgSizeOut,int imgLinOut,int imgColIn,int imgColOut,int channelNum,int border){

	int offsetBloco;
	int bId= blockIdx.x;
	int qntLinBloco;

	int offsetThread;
	int qntLinThread=0;
	int tId = threadIdx.x;

	int c=-1;

	
	// loop para detectar qual o primeiro bloco de cada canal
	for(int i=0;i<channelNum;i++){
		if(bId>=d_blocoOffset[i] && bId<d_blocoOffset[i+1]){
			c=i;
			break;
		}
	}
	//se eh um bloco ocioso, retorna
	if(c<0 || c>=channelNum){
		return;
	}
	//para a logica  dos calculos abaixo funcionar, o identificador do primeiro bloco de cada
	//canal deve ser zero, deslocando o valor de todos os blocos seguintes para proximo de zero
	//portanto, sutrai-se o primeiro bloco de todos os identificadores de bloco
	bId=bId-d_blocoOffset[c];
	
	//Nos condicionais abaixo, calculam-se os offsets das threads e dos bloco
	//a divisao segue a segunte logica: caso a divisao nao de exata, o restante eh distribuido
	//aos demais blocos, adicionando uma linha. Comeca-se a partir do primeiro bloco adicionando
	//uma linha ate se exaurirem os remanescentes


	//calclo da primeira linha do bloco
	if(bId<(imgLinOut%d_blocoQnt[c])){
		offsetBloco= bId*floorf(1.0*imgLinOut/d_blocoQnt[c]) + bId;
	}else{
		offsetBloco= bId*floorf(1.0*imgLinOut/d_blocoQnt[c]) + imgLinOut%d_blocoQnt[c];
	}

	//calculo das quantidades de llinha por bloco
	if(bId<imgLinOut%d_blocoQnt[c]){//se ainda existe remanescente a ser distribuido, adiciona nesse bloco
		qntLinBloco=ceilf(1.0*imgLinOut/d_blocoQnt[c]);
	}else{ //caso contrario, permanece com a quantidade orignal
		qntLinBloco=floorf(1.0*imgLinOut/d_blocoQnt[c]);
	}

	//claculo da primeira linha da thread, considerando que a primeira linha do bloco eh 0
	if(tId<qntLinBloco%blockDim.x){
		offsetThread=tId*floorf(1.0*qntLinBloco/blockDim.x)+tId;

	}else{
		offsetThread=tId*floorf(1.0*qntLinBloco/blockDim.x)+qntLinBloco%blockDim.x;
	}
	//calculo da quantidae de linhas que essa thread sera responsavel
	if(tId<qntLinBloco%blockDim.x){
		qntLinThread=ceilf(1.0*qntLinBloco/blockDim.x);
	}else{
		qntLinThread=floorf(1.0*qntLinBloco/blockDim.x);
	}
	
	//pegando o offset do primeiro bloco do canal
	// Quantidade total de linhas / qntde de blocos naquele canal
	int offsetPrimeiroBloco;
	if(d_blocoOffset[c]<(imgLinOut%d_blocoQnt[c])){
		offsetPrimeiroBloco= d_blocoOffset[c]*floorf(1.0*imgLinOut/d_blocoQnt[c]) + d_blocoOffset[c];
	}else{
		offsetPrimeiroBloco= d_blocoOffset[c]*floorf(1.0*imgLinOut/d_blocoQnt[c]) + imgLinOut%d_blocoQnt[c];
	}

	//calculando a primeira linha efetiva da thread, adicionando o offset do bloco
	int linStart=offsetThread+offsetBloco;
	
	float sum;
	
	//realiza o smooth
	for(int i=linStart;i<linStart+qntLinThread;++i){
		for(int j=0;j<imgColOut;++j){ //cada thread processa a linha toda, nao havendo divisao das colunas
			sum=0;
			for(int l=-border;l<=border;l++){
				for(int k=-border;k<=border;k++){
					sum+=d_inImage[posCalc(c,i+l+border,j+k+border,imgColIn,imgSizeIn)];
				}
			}
			sum=sum/(((2.0*border)+1)*((2.0*border)+1));
			d_outImage[posCalc(c,i,j,imgColOut,imgSizeOut)]=sum;
		}
	}
}


void diffTimeSpec(struct timespec start,struct timespec end,struct timespec *temp);


int main(int argc,char** argv){
	
	int NTHREADS = 3; // Inicializando com 3 Threads, mas sera alterado
	int NBLOCOS =  15; //Inicializando com 15 blocos, mas sera alterado
	
	
	
	//le a quantidade de threads e blocos a partir dos arguemntos
	if(argc>=5){
		NTHREADS=atoi(argv[3]);
		//cout<<NTHREADS<<endl;
		NBLOCOS=atoi(argv[4]);
	}
	
	
	int border = 2; //relativo ao tamanho do kernel utilizado. Valor 2 deve ser utilizado para  kernel de tamanho  5
	
	

	if(NBLOCOS<=0){
		cout <<"E necessario pelo menos um processo alem do master"<<endl;
		return -1;

	}

	Mat image;
	if(argc!=5){
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
	
	
	

	unsigned char *d_outImage,*d_inImage;//imagens de entrada e saida
	int *d_blocoOffset,*d_blocoQnt; //dados de controle. identificador do primeiro bloco de cada canal e a quantidade de blocos pro canal
	unsigned char *h_outImage,*h_inImage;
	int *h_blocoOffset,*h_blocoQnt;

	int sizeIn = image.rows*image.cols*channelNum*sizeof(unsigned char);
	int sizeOut = rowMat*colsMat*channelNum*sizeof(unsigned char);
	
	//alocacao de memoria
	h_inImage = (unsigned char*) malloc(sizeIn);
	h_outImage = (unsigned char*) malloc(sizeOut);
	h_blocoQnt = (int*) malloc((channelNum+1)*sizeof(int));
	h_blocoOffset = (int*) malloc((channelNum+1)*sizeof(int));
	

	cudaFatal(cudaMalloc(&d_inImage,sizeIn));
	cudaFatal(cudaMalloc(&d_outImage,sizeOut));
	cudaFatal(cudaMalloc(&d_blocoQnt,(channelNum+1)*sizeof(int)));
	cudaFatal(cudaMalloc(&d_blocoOffset,(channelNum+1)*sizeof(int)));
	

	//converrsao da imagem para um vetor unidimensional
	int k=0;
	for (int c=0;c<channelNum;c++){
		for(int i=0;i<image.rows;i++){
			for (int j = 0; j < image.cols; ++j){
				if(channelNum==3){
					h_inImage[k++] = image.at<Vec3b>(i,j).val[c];

				}else{
					h_inImage[k++] = image.at<uchar>(i,j);					
				}
			}
		}
	}
	
	//calculo de informacao de controle
	//necessaria para o kernel calcular em qual posicao da  imagem deve atuar
	//calculo do identificador do primeiro bloco de cada canal, bem como a quantidade de blocos por canal
	for(int i=0;i<channelNum;++i){
		if(i< NBLOCOS%channelNum){
			h_blocoOffset[i]=i*floor(1.0*NBLOCOS/channelNum) + i;
			h_blocoQnt[i]=ceil(1.0*NBLOCOS/channelNum);
		}else{
			h_blocoOffset[i]=i*floor(1.0*NBLOCOS/channelNum) + NBLOCOS%channelNum;
			h_blocoQnt[i]= floor(1.0*NBLOCOS/channelNum);
		}
		
	}
	
	int imgLinOut = outputMat.rows;
	h_blocoOffset[channelNum] = NBLOCOS;//seta uma posicao a mais no offset dos blocos para que seja possivel o calculo no kernel
	

	//transferencia de dados para a GPU
	cudaFatal(cudaMemcpy(d_inImage,h_inImage,sizeIn,cudaMemcpyHostToDevice));
	cudaFatal(cudaMemcpy(d_blocoOffset,h_blocoOffset,(channelNum+1)*sizeof(int),cudaMemcpyHostToDevice));
	cudaFatal(cudaMemcpy(d_blocoQnt,h_blocoQnt,(channelNum+1)*sizeof(int),cudaMemcpyHostToDevice));

	//calculos do tamanho da imagem
	int imgSizeIn = image.rows*image.cols;
	int imgSizeOut = rowMat*colsMat;
	int imgColIn = image.cols;
	int imgColOut = colsMat;

	dim3 griSize(NBLOCOS,1,1);//estabelece a quantidede de blocos
	dim3 blockSize(NTHREADS,1,1); //estabele a quantidade de threds
	smooth<<<griSize,blockSize>>>(d_inImage,d_outImage,d_blocoQnt,d_blocoOffset,imgSizeIn,imgSizeOut,imgLinOut,imgColIn,imgColOut,channelNum,border);
	cudaFatal(cudaThreadSynchronize());

	//REaliza a copia de volta para a CPU
	cudaFatal(cudaMemcpy(h_outImage,d_outImage,sizeOut,cudaMemcpyDeviceToHost));
	int auxC,auxI,auxJ;
	
	//retorna a saida para eu formato matricial
	k=0;
	for (int c=0;c<channelNum;c++){
		for(int i=0;i<rowMat;i++){
			for(int j=0;j<colsMat;j++){
				if(channelNum==3){
					channels[c].at<uchar>(i,j)=(uchar)h_outImage[k++];
				//	cout<<vetorIdCanal[i]<<endl;
				}else{
					outputMat.at<uchar>(i,j)=(uchar)h_outImage[k++];		
				}
			}

		}

	}
	//realiza o merge, uniao dos canais para criar a saida final
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

	free(h_outImage);
	free(h_inImage);

	cudaFatal(cudaFree(d_outImage));
	cudaFatal(cudaFree(d_inImage));
	
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
