/*
 * neuralNetWeight.c
 *
 *  Created on: 2026/01/20
 *      Author: h
 */

#include "neuralNetWeights.h"



/*******************************************
 Neural Network
 * Input layer, 1 node
 * Hidden layer 1, 5 nodes
 * Output layer, 1 node
********************************************/

// Input data
Matrix_t Xtrain = { "Xtrain",10,1,(float*)val_Xtrain };

// Weight W1
Matrix_t W1={ "W1",1,5,(float*)val_W1 };
Matrix_t b1={ "b1",5,1,(float*)val_b1 };

// Weight W2
Matrix_t W2={ "W2",5,1,(float*)val_W2 };
Matrix_t b2={ "b2",1,1,(float*)val_b2 };

// Correct data Ttrain
Matrix_t Ttrain ={ "Ttrain",10,1,(float*)val_Ttrain };

// Calculation buffer
float buffer_Y1out[100];
float buffer_Y2out[100];
// Calculation buffer 1 Y1out
Matrix_t Y1out={ "Y1out",4,4,(float*)buffer_Y1out };

// Calculation buffer 2 Y2out
Matrix_t Y2out={ "Y2out",4,2,(float*)buffer_Y2out };

// Display buffer
float val_Diff[15];
float val_Ratio[15];
float val_DData[10][5];
Matrix_t Diff={ "Diff",15,1,(float*)val_Diff };

Matrix_t Ratio={ "Ratio",15,1,(float*)val_Ratio };

Matrix_t DData={ "DData",10,5,(float*)val_DData };

//**************************************
// Nikkei vs JPY=X learned weight and bias data
//**************************************

const float MEAN_JPY    = 144.07161228f;
const float STD_JPY     = 10.06022906f;
const float MEAN_NIKKEI = 34977.46915665f;
const float STD_NIKKEI  = 6931.12944101f;

// input data JPY=X
// Calculate index when using: X[i][j] -> X_flat[i * 2 + j]
// val_Xtrain shape: (10, 1)
float val_Xtrain[] = {
    147.81799316f,
    136.39500427f,
    132.99099731f,
    144.53999329f,
    148.12800598f,
    115.63500214f,
    130.35099792f,
    145.71400452f,
    148.29600525f,
    159.86099243f
};

// Layer 1 (Input -> Hidden)
// Calculate index when using: W[i][j] -> W_flat[i * 2 + j]
// val_W1 shape: (1, 5)
const float val_W1[] = {
    0.33833527f,
    0.40261043f,
    1.00266723f,
    0.82899118f,
    1.18707913f
};

// Calculate index when using: W[i][j] -> W_flat[i * 2 + j]
// val_b1 shape: (5,)
const float val_b1[] = {
    0.07524294f,
    0.29752470f,
    0.28947074f,
    -0.17233699f,
    -0.72011154f
};

// Layer 2 (Hidden -> Output)
// Calculate index when using: W[i][j] -> W_flat[i * 2 + j]
// val_W2 shape: (5, 1)
const float val_W2[] = {
    0.07909354f,
    -0.05351116f,
    0.94874739f,
    0.85483691f,
    0.84306110f
};

// Calculate index when using: W[i][j] -> W_flat[i * 2 + j]
// val_b2 shape: (1,)
const float val_b2[] = {
    -1.22420727f
};

// real N224 data
// Calculate index when using: X[i][j] -> X_flat[i * 2 + j]
// val_Ttrain shape: (10, 1)
float val_Ttrain[] = {
    36119.92187500f,
    28143.97070312f,
    27501.85937500f,
    38211.51171875f,
    35963.26953125f,
    27460.40039062f,
    26213.64062500f,
    31450.75976562f,
    38707.64062500f,
    38804.64843750f
};

/*
 *Iteration 9001: Loss=0.79083
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
*/
