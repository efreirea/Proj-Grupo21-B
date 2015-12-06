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
	int qntLinBloco; //talvez n precise

	int offsetThread;
	int qntLinThread=0;
	int tId = threadIdx.x;

	int c=-1;

	// int offsetCanal;
	//0
	//171
	//342
	//512
	// for(int i=0;i<channelNum;i++){
	// 	printf("%d ",d_blocoOffset[i]);
	// }
	
	for(int i=0;i<channelNum;i++){
		if(bId>=d_blocoOffset[i] && bId<d_blocoOffset[i+1]){
			c=i;
			break;
		}
	}
	if(c<0 || c>=channelNum){
		return;
	}
	// if(c==3){
	// 	c=2;
	// }
	bId=bId-d_blocoOffset[c];
	// Quantidade total de linhas / qntde de blocos naquele canal
	if(bId<(imgLinOut%d_blocoQnt[c])){
		offsetBloco= bId*floorf(1.0*imgLinOut/d_blocoQnt[c]) + bId;
	}else{
		offsetBloco= bId*floorf(1.0*imgLinOut/d_blocoQnt[c]) + imgLinOut%d_blocoQnt[c];
	}

	if(bId<imgLinOut%d_blocoQnt[c]){
		qntLinBloco=ceilf(1.0*imgLinOut/d_blocoQnt[c]);
	}else{
		qntLinBloco=floorf(1.0*imgLinOut/d_blocoQnt[c]);
	}
	// Quantidade total de linhas em um bloco / quantiade de threads naquele bloco
	if(tId<qntLinBloco%blockDim.x){
		offsetThread=tId*floorf(1.0*qntLinBloco/blockDim.x)+tId;

	}else{
		offsetThread=tId*floorf(1.0*qntLinBloco/blockDim.x)+qntLinBloco%blockDim.x;
	}
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

	// int linStart=offsetThread+offsetBloco;
	// int linStart=offsetThread+offsetBloco-offsetPrimeiroBloco;
	int linStart=offsetThread+offsetBloco;
	
	float sum;
	// printf("BlockID: %d ThreadID: %d C: %d offsetBloco: %d qntLinBloco:  %d offsetThread %d qntLinThread %d linstart: %d\n",bId,tId,c,offsetBloco,qntLinBloco,offsetThread,qntLinThread,linStart);

	for(int i=linStart;i<linStart+qntLinThread;++i){
		for(int j=0;j<imgColOut;++j){
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

// void smooth(unsigned char *img,unsigned char *output,int qntLinhas,int qntCols,int size_com_borda,int offset,int order);
void diffTimeSpec(struct timespec start,struct timespec end,struct timespec *temp);


int main(int argc,char** argv){
	
	int NTHREADS = 3; // Inicializando com 3 Threads, mas sera alterado
	int NBLOCOS =  15; //Inicializando com 15 blocos, mas sera alterado
	//define a quantidade de blocos de acordo com o numero de processos gerados. Tal numero esta relacionado com a quantidade de ghosts no arquivo de host
	
	//elimina um noh, pois eh o master. EM nossa implementacao, o master nao realiza processamento da imagem
	
	if(argc>=5){
		NTHREADS=atoi(argv[3]);
		//cout<<NTHREADS<<endl;
		NBLOCOS=atoi(argv[4]);
	}
	
	
	int border = 2; //relativo ao tamanho do kernel utilizado. Valor 2 deve ser utilizado para  kernel de tamanho  5
	//noh master que disponibiliza recursos para as outras
	

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
	
	//divisao do trabalho
	
	unsigned char *d_outImage,*d_inImage;
	int *d_blocoOffset,*d_blocoQnt;
	unsigned char *h_outImage,*h_inImage;
	int *h_blocoOffset,*h_blocoQnt;

	int sizeIn = image.rows*image.cols*channelNum*sizeof(unsigned char);
	int sizeOut = rowMat*colsMat*channelNum*sizeof(unsigned char);
	

	h_inImage = (unsigned char*) malloc(sizeIn);
	h_outImage = (unsigned char*) malloc(sizeOut);
	h_blocoQnt = (int*) malloc((channelNum+1)*sizeof(int));
	h_blocoOffset = (int*) malloc((channelNum+1)*sizeof(int));
	

	cudaFatal(cudaMalloc(&d_inImage,sizeIn));
	cudaFatal(cudaMalloc(&d_outImage,sizeOut));
	cudaFatal(cudaMalloc(&d_blocoQnt,(channelNum+1)*sizeof(int)));
	cudaFatal(cudaMalloc(&d_blocoOffset,(channelNum+1)*sizeof(int)));
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
	//512 / 3 =170
	//512 % 3 = 2 

	//0
	//171
	//342
	//512
	for(int i=0;i<channelNum;++i){
		if(i< NBLOCOS%channelNum){
			h_blocoOffset[i]=i*floor(1.0*NBLOCOS/channelNum) + i;
			h_blocoQnt[i]=ceil(1.0*NBLOCOS/channelNum);
		}else{
			h_blocoOffset[i]=i*floor(1.0*NBLOCOS/channelNum) + NBLOCOS%channelNum;
			h_blocoQnt[i]= floor(1.0*NBLOCOS/channelNum);
		}
		// printf("%d ",h_blocoOffset[i]);
	}
	// printf("NBLOCOS: %d NTHREADS: %d\n",NBLOCOS,NTHREADS);
	int imgLinOut = outputMat.rows;
	h_blocoOffset[channelNum] = NBLOCOS;
	// for(int i=0;i<channelNum+1;i++){
	// 	printf("%d ",h_blocoOffset[i]);
	// }
	// printf("\n");
	cudaFatal(cudaMemcpy(d_inImage,h_inImage,sizeIn,cudaMemcpyHostToDevice));
	cudaFatal(cudaMemcpy(d_blocoOffset,h_blocoOffset,(channelNum+1)*sizeof(int),cudaMemcpyHostToDevice));
	cudaFatal(cudaMemcpy(d_blocoQnt,h_blocoQnt,(channelNum+1)*sizeof(int),cudaMemcpyHostToDevice));


	int imgSizeIn = image.rows*image.cols;
	int imgSizeOut = rowMat*colsMat;
	int imgColIn = image.cols;
	int imgColOut = colsMat;

	dim3 griSize(NBLOCOS,1,1);
	dim3 blockSize(NTHREADS,1,1);
	smooth<<<griSize,blockSize>>>(d_inImage,d_outImage,d_blocoQnt,d_blocoOffset,imgSizeIn,imgSizeOut,imgLinOut,imgColIn,imgColOut,channelNum,border);
	cudaFatal(cudaThreadSynchronize());
// smooth(unsigned char *d_inImage,unsigned char *d_outImage,unsigned char *d_blocoQnt,unsigned char *d_blocoOffset,
	// int imgSizeIn,int imgSizeOut,int imgLinOut,int imgColIn,int imgColOut,int channelNum,int border)

	cudaFatal(cudaMemcpy(h_outImage,d_outImage,sizeOut,cudaMemcpyDeviceToHost));
	int auxC,auxI,auxJ;
	k=0;
	for (int c=0;c<channelNum;c++){
		for(int i=0;i<rowMat;i++){
			for(int j=0;j<colsMat;j++){
				if(channelNum==3){
	//		outputMat.at<Vec3b>(l,k).val[vetorIdCanal[i]]=partitionedBlockRcv[y++];
			
					channels[c].at<uchar>(i,j)=(uchar)h_outImage[k++];
				//	cout<<vetorIdCanal[i]<<endl;
				}else{
					outputMat.at<uchar>(i,j)=(uchar)h_outImage[k++];		
				}
			}

		}

	}


	// for(int k=0;k<(sizeOut/sizeof(unsigned char));k++){
	// 	auxC=idxCalcC(k,imgColOut,imgSizeOut);
	// 	auxI=idxCalcI(k,imgColOut,imgSizeOut,channelNum);
	// 	auxJ=idxCalcJ(k,imgColOut,imgSizeOut);
	// //	printf("C: %d I: %d J: %d\n",auxC,auxI,auxJ);
	// 	if(channelNum==3){
	// //		outputMat.at<Vec3b>(l,k).val[vetorIdCanal[i]]=partitionedBlockRcv[y++];
			
	// 		channels[auxC].at<uchar>(auxI,auxJ)=(uchar)h_outImage[k];
	// 	//	cout<<vetorIdCanal[i]<<endl;
	// 	}else{
	// 		outputMat.at<uchar>(auxI,auxJ)=(uchar)h_outImage[k];		
	// 	}
	// }

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


//relaiza o calculo da media
// inline
// unsigned char calculaMedia(unsigned char *in,unsigned char *out,int qntLinhas,int qntCols,int p,int border){
// 	int sum=0;
// 	int i= idxCalcI(p,qntCols);
// 	int j= idxCalcJ(p,qntCols);
// 	for(int k=-border;k<=border;k++){
// 		for(int l=-border;l<=border;l++){
// 			sum+=in[posCalc(i+k,j+l,qntCols)] ;
// 		}
// 	}

// 	return (unsigned char)(sum/pow(border*2+1,2));	

// }

// void smooth(unsigned char *img,unsigned char *output,int qntLinhas,int qntCols,int size_com_borda,int offset,int border){
// 	//TODO: colocar bordas de acordo com necessidade
// 	int sum;
// //	Mat imgWithBorder(img,Rect(border,border,img.rows,img.cols));
// 	int lastPos = posCalc(qntLinhas+2*border-1,qntCols-1,qntCols)+1; 
// 	int k=0;
// 	int auxi,auxj;
// 	unsigned char auxDebug;
// 	//cout<< "Retorno do smooth: ";
	
// 	#pragma omp parallel //shared(output)
// 	{
// 		#pragma omp for schedule(dynamic) private(auxi,auxj,auxDebug)
// 			//processa o vetor. Como a matriz foi colocada em formato de vetor, eh necessario um for so, mas fora utilizadas fucoes para converter os indces
// 			for(int i=posCalc(border,border,qntCols);i<lastPos;i++){
// 				auxi=idxCalcI(i,qntCols);
// 				auxj=idxCalcJ(i,qntCols);
// 				//cout<< auxi << " "<<auxj<<endl;
// 				if(auxi>=border && auxi<qntLinhas+2*border-border && auxj>=border && auxj<qntCols-border){ // se e um pixel valido da imagem
// 					auxDebug=calculaMedia(img,output,qntLinhas,qntCols,i,border);
// 					output[posCalc(auxi-border,auxj-border,qntCols-2*border)]=auxDebug;
		
// 				}

				
// 			}

// 	}


// }
