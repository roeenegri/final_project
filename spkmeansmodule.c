#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h> 
#include <stdlib.h> /*memo alloc*/
#include <math.h> /*math ops*/
#include <string.h> /*string ops*/
#include <ctype.h>
#include "spkmeans.h"

/* function declaration */

static PyObject* mainC_Py(PyObject *self, PyObject *args);


/* constants */

double EPSILON = 0.001;


/* pythonC function */
static PyMethodDef _capiMethods[] = {
    {"fit", (PyCFunction)mainC_Py, METH_VARARGS, PyDoc_STR("calc the k meams")},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    _capiMethods
};


PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    PyObject *m;
    m = PyModule_Create(&_moduledef);
    if(!m){
        return NULL;
    }
    return m;
}

/* main function */

static PyObject *
mainC_Py(PyObject *self, PyObject *args){
    PyObject *kFirstVectorsIndexPy;
    PyObject *ret;
    char *file_name;
    char* goal;
    int* k_first_vectors_index;
    int k, n, max_iter, dimension;
    int cnt ; /*iteration counter of the while-loop*/
    double **cents; /*array of centroids*/
    double **prev_cents; /*array of previos centroids (for later calculations)*/
    double **data_list;
    int i;
    int val;
    int *num_of_vectors;
    double **sum_of_vectors;
    int j;
    double **return_matrix;
    double **adjacency_matrix;
    double **lnorm_matrix;
    double **ddg_matrix;
    double **identity_matrix;
    double *eigen_values;


    if (!PyArg_ParseTuple(args, "ssOiii", &goal, &file_name, &kFirstVectorsIndexPy, &k, &n, &dimension)){
        terminate_with_error();
        return NULL;
    }

    /* place for eigen_values */
    eigen_values = calloc (n, sizeof(double));

    /* place for data_list */
    data_list = (double **) calloc (n, sizeof(double *));
    if(!data_list){
        terminate_with_error();
    }
    for (i = 0; i < n; i++) {
        data_list[i] = (double *) calloc (dimension, sizeof(double));
        if(!data_list[i]){
            terminate_with_error();
        }
    }

    /* place for return_matrix */
    return_matrix = (double **) calloc (n, sizeof(double *));
    if(!return_matrix){
        terminate_with_error();
    }
    for (i = 0; i < n; i++) {
        return_matrix[i] = (double *) calloc (n, sizeof(double));
        if(!return_matrix[i]){
            terminate_with_error();
        }
    }
    /* place for adjacency_matrix */
    adjacency_matrix = (double **) calloc (n, sizeof(double *));
    if(!adjacency_matrix){
        terminate_with_error();
    }
    for (i = 0; i < n; i++) {
        adjacency_matrix[i] = (double *) calloc (n, sizeof(double));
        if(!adjacency_matrix[i]){
            terminate_with_error();
        }
    }
    /* place for ddg_matrix */
    ddg_matrix = (double **) calloc (n, sizeof(double *));
    if(!ddg_matrix){
        terminate_with_error();
    }
    for (i = 0; i < n; i++) {
        ddg_matrix[i] = (double *) calloc (n, sizeof(double));
        if(!ddg_matrix[i]){
            terminate_with_error();
        }
    }
    /* place for lnorm_matrix */
    lnorm_matrix = (double **) calloc (n, sizeof(double *));
    if(!lnorm_matrix){
        terminate_with_error();
    }
    for (i = 0; i < n; i++) {
        lnorm_matrix[i] = (double *) calloc (n, sizeof(double));
        if(!lnorm_matrix[i]){
            terminate_with_error();
        }
    }

    /* place for identity_matrix */
    identity_matrix = (double **) calloc (n, sizeof(double *));
    if(!identity_matrix){
        terminate_with_error();
    }
    for (i = 0; i < n; i++) {
        identity_matrix[i] = (double *) calloc (n, sizeof(double));
        if(!identity_matrix[i]){
            terminate_with_error();
        }
    }

    if(k == 0){
        create_adjacency_matrix(adjacency_matrix, data_list, dimension, n);
         create_L_norm_matrix(identity_matrix, ddg_matrix, lnorm_matrix, adjacency_matrix, n);
         jacobi_algorithm(return_matrix, lnorm_matrix, eigen_values, n);
         k = calculate_k_from_eigengap(eigen_values, n);
    }
    k_first_vectors_index = (int*) calloc(k, sizeof(int));
    if(!k_first_vectors_index){
        terminate_with_error();
        return NULL;
    }

    for(i=0; i < k; i++){
        PyObject *item = PyList_GetItem(kFirstVectorsIndexPy, i); 
        k_first_vectors_index[i] = PyLong_AsLong(item);
    }


    cnt = 0;



    prev_cents = calloc(k, sizeof(double*));
    if(!prev_cents){
        terminate_with_error();
        return NULL;
    }
    for(i = 0; i < k; ++i){
        prev_cents[i] = calloc(dimension, sizeof(double));
    }
    
    val = extract_vectors(file_name, data_list, n, dimension); //put all the data from the file to the datalist
    if(val == -2){
        terminate_with_error();
        return NULL;
    }
    if(strcmp(goal, "spk")){

        /*algorithm*/

        /*initialization of centroid list: with first k vectors*/
        max_iter = 300;

        cents = extract_k_first_vectors(data_list, k , dimension); 

        do {
        
            int current_xi_position = -1;
            int clust_num = -1;
        
            sum_of_vectors = calloc(k, sizeof(double*));
            if(!sum_of_vectors){
                terminate_with_error();
                return NULL;
            }

            for(i = 0; i < k; ++i){
                sum_of_vectors[i] = calloc(dimension, sizeof(double));
            } 

            num_of_vectors = calloc(k, sizeof(int));

            if(!num_of_vectors){
                terminate_with_error();
                return NULL;
            }

            /* prev_cents = cents; we save the current cents to compare the with new ones later: make prev point to cents,,,, could be a pointer problem here*/ 

            for (i=0; i<k; i++) {
                for (j=0; j<dimension; j++) {
                    prev_cents[i][j] = cents[i][j];
                }
            }


            for (current_xi_position = 0; current_xi_position < n; current_xi_position ++) {
                
                clust_num = matching_cluster(data_list[current_xi_position], cents, k, dimension); 
                sum_two_vectors(sum_of_vectors, data_list[current_xi_position], dimension, clust_num);
                num_of_vectors[clust_num] += 1;
            }

            calc_centroids(sum_of_vectors, num_of_vectors, cents, k, dimension);

            cnt = cnt+1;

        } while ((cnt < max_iter) && (check_centroid_diff(prev_cents, cents, k, dimension) == 0));

        ret = PyList_New(k);
        for (i = 0;i < k; i++){
            PyObject *vector_py = PyList_New(dimension);
            for(j = 0;j < dimension; j++){
                PyList_SetItem(vector_py, j, PyFloat_FromDouble(cents[i][j]));
            }
            PyList_SetItem(ret,i, vector_py);
        }
        /* free memory */
        
        for(i = 0; i<n; i++){
            free(data_list[i]);
        }
        free(data_list);
        free(k_first_vectors_index);

        for(i = 0; i<k; i++){
            free(prev_cents[i]);
        }
        free(prev_cents);

        for(i=0; i<k; i++){
            free(cents[i]);
        }
        free(cents);
        free(num_of_vectors);

        for(i=0; i<k; i++){
            free(sum_of_vectors[i]);
        }
        free(sum_of_vectors);
        
        return ret;
    }

    if(!strcmp(goal, "wam")){
        create_adjacency_matrix(return_matrix, data_list, dimension, n);
    }
    else if(!strcmp(goal, "ddg")){
        create_adjacency_matrix(adjacency_matrix, data_list, dimension, n);
        create_diagonal_degree_matrix(return_matrix, adjacency_matrix, n, 0);
    }
    else if(!strcmp(goal, "lnorm")){
         create_adjacency_matrix(adjacency_matrix, data_list, dimension, n);
         create_L_norm_matrix(identity_matrix, ddg_matrix, return_matrix, adjacency_matrix, n);
    }
    else if(!strcmp(goal, "jacobi")){
         create_adjacency_matrix(adjacency_matrix, data_list, dimension, n);
         create_L_norm_matrix(identity_matrix, ddg_matrix, lnorm_matrix, adjacency_matrix, n);
         jacobi_algorithm(return_matrix, lnorm_matrix, eigen_values, n);
    }
    else{
        return_matrix = NULL;
        terminate_invalid_input();
    }
    print_square_matrix(return_matrix, n);
    memory_free(return_matrix, n);
    memory_free(adjacency_matrix, n);
    memory_free(lnorm_matrix, n);
    memory_free(ddg_matrix, n);
    memory_free(identity_matrix, n);
    memory_free(data_list, n);
    ret = PyList_New(n);
    for (i = 0;i < n; i++){
        PyObject *vector_py = PyList_New(n);
        for(j = 0;j < n; j++){
            PyList_SetItem(vector_py, j, PyFloat_FromDouble(return_matrix[n][n]));
        }
        PyList_SetItem(ret, i, vector_py);
    }
    return ret;
}