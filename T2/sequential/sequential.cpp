#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;
void smooth(Mat& img,Mat& output);
void diffTimeSpec(struct timespec start,struct timespec end,struct timespec *temp);
int main(int argc, char** argv){
	Mat image;
	if(argc!=3){
		printf("entrar com nome do arquivo e Tipo de imagem\n");
		return -1;
	}
	int tipoImagem = atoi(argv[2]);
	//abre a imagem de acordo com seu tipo (RGB ou grayScale)
	if(tipoImagem == 1){ //1 para imagens RGB
		image=imread(argv[1],CV_LOAD_IMAGE_COLOR);
	}else if(tipoImagem==2){ //2 para imagens grayScale
		image=imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
	}
	
	if(!image.data){
		printf("No data read.\n");
		return -1;
	}

	Mat output(image.rows,image.cols,image.depth());
	//printf("%d %d %d %d\n",CV_LOAD_IMAGE_COLOR,image.rows,image.cols,image.channels());
	struct timespec begin, end,result;
	double time_spent;	
	//funcao para coletar o tempo
	clock_gettime(CLOCK_REALTIME,&begin);
	//funcao que realiza o algoritmo de smooth
	smooth(image,output);
	//funcao para coletar o tempo
	clock_gettime(CLOCK_REALTIME,&end);
		
	//Calculo tempo de execucao
	diffTimeSpec(begin, end, &result);
	//time_spent=((double) difftime(end.tv_sec,begin.tv_sec))+(result.tv_nsec*1.0/1000000000.0);
	time_spent=((double) result.tv_sec)+(result.tv_nsec*1.0/1000000000.0);
	cout<<argv[1]<<"\t"<<-1<<"\t"<<-1<<"\t"<<time_spent<<endl;
	// namedWindow("DisplayImage",WINDOW_NORMAL);
	// namedWindow("DisplayImageOriginal",WINDOW_NORMAL);
	// imshow("DisplayImage",output);
	// imshow("DisplayImageOriginal",image);
	std::string inFileName(argv[1]);
	std::string outFileName  = inFileName.substr(0,inFileName.find_last_of("."));
	outFileName += ".ppm";
	imwrite(outFileName,output);
	// waitKey(0);
	return 0;

}


void smooth(Mat& img,Mat& output){
	int border = 2; //especifica o tamanho da borda adicional na imagem
	
	int sum;
	int channelNum= img.channels();
	Mat channels[3];
	//adiciona borda na imagem para tratamento de pixel de borda
	copyMakeBorder(img,img,border,border,border,border,BORDER_CONSTANT,Scalar(0,0,0));
	//matriz de saida
	Mat outB(img.rows-border*2,img.cols-border*2,img.depth());
	channels[0]=outB;
	//caso seja imagem RGB
	if(channelNum==3){
		Mat outG(img.rows-border*2,img.cols-border*2,img.depth());
		Mat outR(img.rows-border*2,img.cols-border*2,img.depth());
		channels[1]=outG;
		channels[2]=outR;
	}
	
	
	
//	Mat imgWithBorder(img,Rect(border,border,img.rows,img.cols));
	
	//percorre os canais (RGB = 3 e GrayScale = 1)
	for(int c=0;c<channelNum;c++){
		//percorre a imagem
		for(int i=border;i<img.rows-border;i++){
			for(int j=border;j<img.cols-border;j++){
					// Soma dos elementos do kernel
					sum=0;
					//realiza a media de uma matriz 5x5 substituindo o pixel central pela media calculada
					for(int k=-2;k<=2;k++){
						for(int l=-2;l<=2;l++){
							if(channelNum==3){ //verifica se eh RGB e realiza a somatoria
								sum+=img.at<Vec3b>(i+k,j+l).val[c]; 
							}else{ //caso seja grayScale e realiza a somatoria
								sum+=img.at<uchar>(i+k,j+l); 
							}
						}
					}
					//substitui o pixel pela media calculada
					channels[c].at<uchar>(i-border,j-border)=(uchar)(sum/25);	
			}
		}
	}
	//caso seja RGB realiza a juncao dos canais para gerar a imagem final
	if (channelNum==3){
		merge(channels,channelNum,output);
	}else{ //caso seja grayScale atrinui a matriz de saida a imagem final
		output=channels[0];
	}

	// namedWindow("AZUL",WINDOW_NORMAL);
	// namedWindow("VERDE",WINDOW_NORMAL);
	// namedWindow("VERMELHO",WINDOW_NORMAL);
	// imshow("AZUL",outB);
	// imshow("VERDE",outG);
	// imshow("VERMELHO",outR);
}

// funcao que calcula o tempo
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