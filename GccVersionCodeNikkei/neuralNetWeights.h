/**
 * @file unidivion_weights.h
 * @brief Model parameter declarations for the uniDivion Micro-Inference Engine.
 * This file defines the interface for pre-trained weights, biases, and
 * calculation buffers.
 * @author uniDivion
 */

#ifndef UNIDIVION_WEIGHTS_H_
#define UNIDIVION_WEIGHTS_H_

#include "neuralNetNikkei.h"

//Nikkei JPY mean standard deviation
extern const float MEAN_JPY;
extern const float STD_JPY;
extern const float MEAN_NIKKEI;
extern const float STD_NIKKEI;

//Matrix data-----------------------
//Weight bias realData:Ttrain
extern Matrix_t Xtrain,W1,b1,W2,b2,Ttrain;
//Calculation Buffer 
extern Matrix_t Y1out,Y2out;
//display buffer
extern Matrix_t Diff, Ratio;
//display baffer
extern Matrix_t DData;

//Matrix data flatter 1D array-------
//input data
extern float val_Xtrain[];
//Weight1
extern const float val_W1[];
//bias1
extern const float val_b1[];
//Weight2
extern const float val_W2[];
//bias2
extern const float val_b2[];
//real data 
extern float val_Ttrain[];
//Calculation buffer array
extern float buffer_Y1out[100];
extern float buffer_Y2out[100];
//display buffer array
extern float val_Diff[15];
extern float val_Ratio[15];
//display array
extern float val_DData[10][5];


/* 答え合わせ用データ　
 * Iteration 9001: Loss=0.79083
       JPY=X         実際の株価     予測株価        誤差        誤差率(%)
[[  147.8179932 36119.921875  36589.3897893   469.4679143     1.2997479]
 [  136.3950043 28143.9707031 31868.5847517  3724.6140486    13.2341456]
 [  132.9909973 27501.859375  30747.588071   3245.728696     11.8018518]
 [  144.5399933 38211.5117188 35149.1887167 -3062.3230021    -8.0141373]
 [  148.128006  35963.2695312 36725.2581958   761.9886645     2.118797 ]
 [  115.6350021 27460.4003906 27560.3462808    99.9458902     0.3639637]
 [  130.3509979 26213.640625  30001.2329799  3787.5923549    14.4489368]
 [  145.7140045 31450.7597656 35663.7529375  4212.9931718    13.3955211]
 [  148.2960052 38707.640625  36798.761107  -1908.879518     -4.9315316]
 [  159.8609924 38804.6484375 41181.2333086  2376.5848711     6.124485 ]]
iterNum  9000
 */


#endif /* INC_NEURALNETWEIGHT_H_ */
