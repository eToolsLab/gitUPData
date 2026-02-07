/**
 * @file neuralNetNikkei.h
 * @brief Light-weight Matrix_ Operation Library for Embedded Edge AI.
 * Part of the uniDivion Micro-Inference Suite.
 * @author uniDivion
 * @version 1.0.0
 */

#ifndef INC_NEURALNETNIKKEI_H_
#define INC_NEURALNETNIKKEI_H_

#include <stdio.h>  // printf用
#include <math.h>   // expf用
#include <stdbool.h>

/**
 * @struct Matrix__t
 * @brief Data structure representing a flattened 2D Matrix_.
 * Designed for memory-constrained embedded systems.
 */
typedef struct {
    char name[16];      // Debug label for the Matrix_
    int rows;           // Number of rows (Shape[0])
    int cols;           // Number of columns (Shape[1])
    float *data;        // Pointer to the flattened 1D data array
} Matrix_t;

/* --- Matrix_ Operations --- */

/**
 * @brief Prints Matrix_ elements to stdout for debugging.
 * @param mat Pointer to the Matrix_ structure.
 */
extern void Matrix_Print(const Matrix_t *mat);

/**
 * @brief Computes the dot product of two matrices (y = x1 . x2).
 * @param x1 Pointer to the first input matrix (Left operand).
 * @param x2 Pointer to the second input matrix (Right operand).
 * @param y  Pointer to the output matrix (Result buffer).
 */
extern void Matrix_Dot(const Matrix_t *x1, const Matrix_t *x2, Matrix_t *y);

/**
 * @brief Performs element-wise addition with broadcasting (y = x + bias).
 * Adds the bias vector to each row of the input matrix.
 * @param x    Pointer to the input matrix.
 * @param bias Pointer to the bias vector.
 * @param y    Pointer to the output matrix.
 */
extern void Matrix_Add(const Matrix_t *x, const Matrix_t *bias, Matrix_t *y);

/* --- Activation Functions --- */

/**
 * @brief Applies the Sigmoid activation function to a matrix element-wise.
 * @param x Pointer to the input matrix.
 * @param y Pointer to the output matrix (Can be same as input for in-place).
 */
extern void Matrix_Sigmoid(Matrix_t *x, Matrix_t *y);

/**
 * @brief Applies the Softmax function to a matrix (typically the output layer).
 * Includes numerical stability handling (max subtraction).
 * @param x Pointer to the target matrix (Modified in-place).
 */
extern void Matrix_Softmax(Matrix_t *x);

/**
 * @fn void Matrix_IdentifyFunction(Matrix_t*)
 * @brief
 *
 * @param x
 */
extern void Matrix_IdentifyFunction(Matrix_t *x);

// Standardization functions
/**
 * @fn float standardizing(float, float, float)
 * @brief Appiles the standardizing function to a scalar
 *
 * @param val : value
 * @param mean:
 * @param std : standard deviation
 * @return
 */
extern float standardizing(float val, float mean, float std);

/**
 * @fn void Matrix_Standardizing(Matrix_t*, float, float)
 * @brief Applies the standardizing function to a matrix
 *
 * @param x: matrix
 * @param mean
 * @param std: standard deviation
 */
extern void Matrix_Standardizing(Matrix_t *x,float mean, float std);

/**
 * @fn float inverse_standardizing(float, float, float)
 * @brief Applies the inverse standardizing to scalar.
 *
 * @param val :value
 * @param mean
 * @param std :standard deviation
 * @return
 */
extern float inverse_standardizing(float val, float mean, float std);

/**
 * @fn void Matrix_InverseStandardizing(Matrix_t*, float, float)
 * @brief
 *
 * @param x : matrix
 * @param mean
 * @param std :standard deviation
 */
extern void Matrix_InverseStandardizing(Matrix_t *x1, float mean, float std);

// Display operation functions

/**
 * @fn void Matrix__CalerrorRatio(Matrix_Items*, Matrix_Items*, Matrix_Items*, Matrix_Items*)
 * @brief Calculation of error and error ratio
 *
 * @param y1     :Real data
 * @param y2     :Predicted value
 * @param diff   :Error buffer
 * @param ratio  :Error ratio buffer
 */
extern void Matrix__CalerrorRatio(Matrix_t *y1, Matrix_t *y2, Matrix_t *diff, Matrix_t *ratio);

/**
 * @fn void Matrix_SetDisplayData(Matrix_Items*, Matrix_Items*, Matrix_Items*, Matrix_Items*, Matrix_Items*)
 * @brief  Store data in the display buffer
 *
 * @param x1    Input data (n,1)
 * @param y1    real Raw data (n,1)
 * @param y2    predict Inference data (n,1)
 * @param diff  Error (n,1)
 * @param ratio Error ratio (n,1)
 */
extern void Matrix_SetDisplayData(Matrix_t *x1,Matrix_t *y1, Matrix_t *y2, Matrix_t *diff, Matrix_t *ratio);

// Execute neural network inference
/**
 * @brief Executes the complete neural network inference flow.
 * Topology: Input -> Hidden(Sigmoid) -> Output.
 */
extern void Matrix_FirstRunNN(void);

/**
 * @fn void Matrix_InputRunNN(float)
 * @brief Executes the  complete neural network inference flow.
 *         input data, predicting  nikkei price.
 * @param jpy
 */
extern void Matrix_InputRunNN(float jpy);
#endif /* INC_NEURALNETNIKKEI_H_ */
