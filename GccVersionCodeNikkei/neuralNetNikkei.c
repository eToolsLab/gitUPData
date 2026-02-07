/*
 * neuralNet.c
 *
 *  Created on: 2026/01/20
 *      Author: h
 */

#include "neuralNetNikkei.h"
#include "neuralNetWeights.h"

// Helper function: Sigmoid for scalar
static float Sigmoid_Scalar(float x)
{
	return 1.0f/(1.0f+expf(-x));
}


void Matrix_Print(const Matrix_t *mat)
{
	printf("[%s] Shape: (%d, %d)\n", mat->name, mat->rows, mat->cols);
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			// Mapping 2D coords to 1D index: [i][j] -> [i * cols + j]
			printf("%2.6f ", mat->data[i * mat->cols + j]);
		}
		printf("\r\n");
	}
	printf("\r\n");
}


void Matrix_Dot(const Matrix_t *x1, const Matrix_t *x2, Matrix_t *y)
{
	// Validation: x1 columns must match x2 rows
	if (x1->cols != x2->rows) {
		printf("Error: Matrix_ shape mismatch in Dot Product.\n");
		return;
	}

	for (int i = 0; i < x1->rows; i++)
	{
		for (int j = 0; j < x2->cols; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < x1->cols; k++)
			{
				// Dot Product accumulation
				sum += x1->data[i * x1->cols + k] * x2->data[k * x2->cols + j];
			}
			y->data[i * x2->cols + j] = sum;
		}
	}
	// Update output shape metadata
	y->rows = x1->rows;
	y->cols = x2->cols;
}


void Matrix_Add(const Matrix_t *x, const Matrix_t *bias, Matrix_t *y)
{
	for (int i = 0; i < x->rows; i++) {
		for (int j = 0; j < x->cols; j++) {
			// Broadcasting: The same bias[j] is added to every row i
			y->data[i * x->cols + j] = x->data[i * x->cols + j] + bias->data[j];
		}
	}
}

void Matrix_Sigmoid(Matrix_t *x, Matrix_t *y)
{
	int total_elements = x->rows * x->cols;
	for (int i = 0; i < total_elements; i++) {
		y->data[i] = Sigmoid_Scalar(x->data[i]);
	}
}


void Matrix_Softmax(Matrix_t *x)
{
	for (int i = 0; i < x->rows; i++) {
		// 1. Find max value for numerical stability (prevent overflow)
		float max_val = x->data[i * x->cols];
		for (int j = 1; j < x->cols; j++) {
			if (x->data[i * x->cols + j] > max_val) {
				max_val = x->data[i * x->cols + j];
			}
		}

		// 2. Compute exponentials and sum
		float sum = 0.0f;
		for (int j = 0; j < x->cols; j++) {
			// Subtract max_val inside exp
			float e = expf(x->data[i * x->cols + j] - max_val);
			x->data[i * x->cols + j] = e;
			sum += e;
		}

		// 3. Normalize
		for (int j = 0; j < x->cols; j++) {
			x->data[i * x->cols + j] /= sum;
		}
	}
}


void Matrix_IdentifyFunction(Matrix_t *x)
{
	for(int i = 0; i < x->rows; i++)
	{
		for(int j=0; j < x->cols; j++)
		{
			x->data[i*x->cols+j] = x->data[i*x->cols+j];
		}
	}
}


float standardizing(float val, float mean, float std)
{
	if(std==0.0f) return val;
	return (val - mean)/std;
}

void Matrix_Standardizing(Matrix_t *x1, float mean, float std)
{
	int i,j;

	for(i=0; i < x1->rows; i++)
	{
		for(j =0; j < x1->cols; j++)
		{
			x1->data[i*x1->cols+j] = standardizing(x1->data[i*x1->cols+j],mean,std);
		}
	}
}

float inverse_standardizing(float val, float mean, float std)
{
	if(std==0.0f) return val;
	return val*std + mean;
}


void Matrix_InverseStandardizing(Matrix_t *x1, float mean, float std)
{
	int i,j;

	for(i=0; i < x1->rows; i++)
	{
		for(j =0; j < x1->cols; j++)
		{
			x1->data[i*x1->cols+j] = inverse_standardizing(x1->data[i*x1->cols+j],mean,std);
		}
	}
}


//**********************
// Display operations
//**********************

void Matrix__CalerrorRatio(Matrix_t *y1, Matrix_t *y2, Matrix_t *diff, Matrix_t *ratio)
{
	int i,j;

	for(i=0; i < y1->rows; i++)
	{
		for(j =0; j < y1->cols; j++)
		{
			diff->data[i*y1->cols+j] = y2->data[i*y1->cols+j] - y1->data[i*y1->cols+j];
			ratio->data[i*y1->cols+j] = diff->data[i*y1->cols+j] / y1->data[i*y1->cols+j]*100;
		}
	}
}


void Matrix_SetDisplayData(Matrix_t *x1,Matrix_t *y1, Matrix_t *y2, Matrix_t *diff, Matrix_t *ratio)
{
	int i,j;

	for(i=0; i< DData.rows; i++)
	{
		for(j=0; j< DData.cols; j++)
		{
			switch(j)
			{
				case 0:
					DData.data[i*DData.cols+0] = x1->data[i];
					break;
				case 1:
					DData.data[i*DData.cols+1] = y1->data[i];
					break;
				case 2:
					DData.data[i*DData.cols+2] = y2->data[i];
					break;
				case 3:
					DData.data[i*DData.cols+3] = diff->data[i];
					break;
				case 4:
					DData.data[i*DData.cols+4] = ratio->data[i];
					break;
			}
		}
	}
}


void Matrix_FirstRunNN(void){
	//neuralnet XOR start-----------------------------
	  printf("********* Input Data Nikkei vs JPY=X ************ \r\n");
	  Matrix_Print(&Xtrain);
	  Matrix_Print(&W1);
	  Matrix_Print(&b1);
	  Matrix_Print(&W2);
	  Matrix_Print(&b2);

	  printf("----------standardization--------------\r\n");
	  Matrix_Standardizing(&Xtrain, MEAN_JPY, STD_JPY);
	  Matrix_Print(&Xtrain);

	  printf("----Input Layer-----\r\n");
	  Matrix_Dot(&Xtrain, &W1, &Y1out);
	  Matrix_Add(&Y1out,&b1,&Y1out);
	  Matrix_Print(&Y1out);

	  printf("----Sigmoid-----\r\n");
	  Matrix_Sigmoid(&Y1out,&Y1out);
	  Matrix_Print(&Y1out);

	  printf("----hidden Layer-----\r\n");
	  Matrix_Dot(&Y1out, &W2, &Y2out);
	  Matrix_Add(&Y2out, &b2, &Y2out);
	  Matrix_Print(&Y2out);

	  printf("----Result-----\r\n");
	  printf("real:JPY=X\r\n");
	  Matrix_InverseStandardizing(&Xtrain,MEAN_JPY,STD_JPY);
	  Matrix_Print(&Xtrain);
	  printf("predict:Nikkei n225\r\n");
	  Matrix_InverseStandardizing(&Y2out,MEAN_NIKKEI,STD_NIKKEI);
	  Matrix_Print(&Y2out);
	  printf("real:Nikkei n225\r\n");
	  Matrix_Print(&Ttrain);
	  printf("----Display All data & Error Rate ---- \r\n");
	  Matrix__CalerrorRatio(&Ttrain,&Y2out, &Diff, &Ratio);
	  Matrix_SetDisplayData(&Xtrain, &Ttrain, &Y2out, &Diff, &Ratio);
	  printf(" JPY=X         real   predict       diff        error rate\r\n");
	  Matrix_Print(&DData);
}

void Matrix_InputRunNN(float jpy)
{
	Xtrain.data[0]=jpy;
	Xtrain.rows=1;
	Xtrain.cols=1;
	//printf("----------standardization--------------\r\n");
	Matrix_Standardizing(&Xtrain, MEAN_JPY, STD_JPY);
	Matrix_Dot(&Xtrain, &W1, &Y1out);
	Matrix_Add(&Y1out,&b1,&Y1out);
	Matrix_Sigmoid(&Y1out,&Y1out);
	//printf("----hidden Layer-----\r\n");
	Matrix_Dot(&Y1out, &W2, &Y2out);
	Matrix_Add(&Y2out, &b2, &Y2out);
	//printf("----Result-----\r\n");
	Matrix_InverseStandardizing(&Xtrain,MEAN_JPY,STD_JPY);
	Matrix_InverseStandardizing(&Y2out,MEAN_NIKKEI,STD_NIKKEI);
	printf("USD/JPY:%4.3f\r\n", jpy);
	printf("n255:%10.5f\r\n", Y2out.data[0]);
}






















