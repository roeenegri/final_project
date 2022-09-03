#include <stdio.h> 
#include <stdlib.h> /*memo alloc*/
#include <math.h> /*math ops*/
#include <string.h> /*string ops*/
#include "spkmeans.h"

/* constants */

/* MAKE SURE: verify successful calloc, free memory, exit(terminate...) and not just terminate */

int main(int argc, char* argv[]){
    char *file_name;
    char *goal;
    int n, dimension;
    double **data_list;
    int i;
    double **return_matrix;
    double **adjacency_matrix;
    double **lnorm_matrix;
    double **ddg_matrix;
    double **identity_matrix;
    double *eigen_values;


    if(argc < 3){
        exit(terminate_invalid_input()); 
    }
    goal = argv[1];
    file_name = argv[2];
    dimension = find_dimension(file_name);
    n = find_num_of_vectors(file_name);

    /* place for data_list */
    data_list = (double **) calloc (n, sizeof(double *));
    if(!data_list){
        exit(terminate_with_error());
        
    }
    for (i = 0; i < n; i++) {
        data_list[i] = (double *) calloc (dimension, sizeof(double));
        if(!data_list[i]){
            exit(terminate_with_error());
        }
    }

    /* place for return_matrix */
    return_matrix = (double **) calloc (n, sizeof(double *));
    if(!return_matrix){
        exit(terminate_with_error());
    }
    for (i = 0; i < n; i++) {
        return_matrix[i] = (double *) calloc (n, sizeof(double));
        if(!return_matrix[i]){
            exit(terminate_with_error());
        }
    }
    /* place for adjacency_matrix */
    adjacency_matrix = (double **) calloc (n, sizeof(double *));
    if(!adjacency_matrix){
        exit(terminate_with_error());
    }
    for (i = 0; i < n; i++) {
        adjacency_matrix[i] = (double *) calloc (n, sizeof(double));
        if(!adjacency_matrix[i]){
            exit(terminate_with_error());
        }
    }
    /* place for ddg_matrix */
    ddg_matrix = (double **) calloc (n, sizeof(double *));
    if(!ddg_matrix){
        exit(terminate_with_error());
    }
    for (i = 0; i < n; i++) {
        ddg_matrix[i] = (double *) calloc (n, sizeof(double));
        if(!ddg_matrix[i]){
            exit(terminate_with_error());
        }
    }
    /* place for lnorm_matrix */
    lnorm_matrix = (double **) calloc (n, sizeof(double *));
    if(!lnorm_matrix){
        exit(terminate_with_error());
    }
    for (i = 0; i < n; i++) {
        lnorm_matrix[i] = (double *) calloc (n, sizeof(double));
        if(!lnorm_matrix[i]){
            exit(terminate_with_error());
        }
    }

    /* place for identity_matrix */
    identity_matrix = (double **) calloc (n, sizeof(double *));
    if(!identity_matrix){
        exit(terminate_with_error());
    }
    for (i = 0; i < n; i++) {
        identity_matrix[i] = (double *) calloc (n, sizeof(double));
        if(!identity_matrix[i]){
            exit(terminate_with_error());
        }
    }
    /* place for return_matrix */
    eigen_values = (double *) calloc (n, sizeof(double));
    extract_vectors(file_name, data_list, n, dimension);
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
        exit(terminate_invalid_input());
    }
    print_square_matrix(return_matrix, n);
    memory_free(return_matrix, n);
    memory_free(adjacency_matrix, n);
    memory_free(lnorm_matrix, n);
    memory_free(ddg_matrix, n);
    memory_free(identity_matrix, n);
    memory_free(data_list, n);
    return 0;
}

/* implementation of helper functions */

double** extract_k_first_vectors(double **data_list, int k, int dimention) {
    int i;
    int j;
    double **centroids;
    centroids = calloc(k, sizeof(double *));
    for(i = 0; i < k; ++i){
        centroids[i] = calloc(dimention, sizeof(double));
        if (centroids[i] == NULL) {
            exit(terminate_with_error());
        }
    }
    for(i = 0; i < k; ++i){
        for(j = 0; j < dimention; ++j){
            centroids[i][j] = data_list[i][j];
        }
    }

    return centroids;
    
    /*extract_first_k_vectors(data_list *, arr *)):
    input: data_list *
    output: will update the double-array arr with first k vectors of the input file*/
}

int check_centroid_diff(double **prv_cents, double **cents, int k, int dimension) { 
    
    int i = 0;
    double diff;

    for(i = 0; i < k; i++){
        diff = calc_norm_form(prv_cents[i], cents[i], dimension, 1);

        if(diff > 0.001){
            return 0;
        }
    }  

    return 1;
   
    /*check_centroid_diff (prev_cents, curr_cents):
    input: 2 lists of previous centroids and current centroids
    output: 1 if ALL of them hasn't changed more than epsilon, else 0*/
}

int find_dimension(char* input_file){
    char c;
    int dimension = 0;
    FILE *f = fopen(input_file, "r");
    if(f == NULL){
        exit(terminate_invalid_input());
    }

    c = fgetc(f);

    while ((c != '\n') && (c != EOF)) {
        if(c == ','){
            dimension += 1;
        }
        c = fgetc(f);
    }

    fclose(f);

    dimension += 1;

    return dimension;
}

int find_num_of_vectors(char* input_file){
            
    FILE *f;
    int count = 0; 
    char c; 
 
    f = fopen(input_file, "r");
  
    if (f == NULL){
        return -1;
    }
  
    c = fgetc(f);

    while (c != EOF) {
        if (c == '\n') {
            count = count + 1;
        }
    
        c = fgetc(f);
    }

    fclose(f);

    return count;
}

void calc_centroids(double **sum_of_vectors,int *num_of_vectors,double **cents, int k, int dimension){

    int i;
    int j;
    for(i = 0; i < k; ++i){
        for(j = 0; j < dimension; ++j){

            cents[i][j] = sum_of_vectors[i][j] / num_of_vectors[i];

        }
    }

    return;
    
    /*input: two arrays: sum_of_vectors, num_of_vectors
    output: void (just updates the centroid value in cents list)*/
}

int matching_cluster(double *current_xi,double **cents, int k, int dimension) {

    int i;
    double curr;
    double min_diff;
    int min_centro;
    
    i = 0;

    min_diff = calc_norm_form(current_xi, cents[0], dimension, 0);
    min_centro = 0;

    for(i = 1; i < k; i++){
        curr = calc_norm_form(current_xi, cents[i], dimension, 0);
        if(curr < min_diff){
            min_diff = curr;
            min_centro = i;
        }
    }

    return min_centro;

    /*input: current vector xi, and the centroid list (data_list)
    output: (int) the number of the new cluster to assign the vector to: the closest centroid to the vector*/
}

double calc_norm_form(double *vec1, double *vec2, int dimension, int with_sqrt) {

    /* calculate the norm of the difference vector */

    double sum;
    int i;
    double num1;
    double num2;

    sum = 0;
    i = 0;

    for(i = 0; i < dimension; i++){
        num1 = vec1[i];
        num2 = vec2[i];
        sum += pow((num1-num2), 2.0);

    }
    
    if (with_sqrt == 1) {
        
        sum = pow(sum, 0.5);

    }

    return sum;
}

int extract_vectors (char *filename, double **data_list, int n, int dimension) {
        
    /*input: filename, a pointer to the data (vector) list
    output: number of vectors
    function will extract all the vectors from txt file to a list of vectors, and return their number*/

    int i = 0;
    double number = 0;
    int j = 0;

    FILE* f = fopen(filename, "r");
    if (f == NULL) {
        return -1; /*invalid input will be printed outside the function call*/
    
    }

    for(i = 0; i < n; i++){
        while(fscanf(f, "%lf", &number) != EOF && (j < dimension)) {
                if ((number != ',') && (number != '\n')){
                data_list[i][j] = number;
                j++;
                number = fgetc(f);
                }
        }
        j = 0;
    }

    if (data_list == NULL) {
        return -2;
    } 

    fclose(f);
    return 0;

     /* i (divided by) dimension =  number of vectors, this is what needs to be returned*/
    /* MISSING: we need to also return the dimension outside somehow */
}

int terminate_with_error () {
        
    /* terminates the program: prints the general error message and returns 1 */

    printf("An Error Has Occurred");
    return 1;
}

int terminate_invalid_input () {
    
    /* terminates the program: prints the invalid input message and returns 1 */

    printf("Invalid Input!");
    return 1;
}

void sum_two_vectors (double **sum_of_vectors, double *new_vector, int dimension, int cluster_number) {

    int i;
    i = 0;
    /*input: a list of vectors, a vector, the dimension and the index of the list to the specific sum-vector we want to add the vector to.
    output: void
    the function takes a vector in the list that represents a sum, and addes a new vector to the sum (the change is done in place)*/

    for (i = 0; i < dimension; i++) {

        sum_of_vectors[cluster_number][i] += new_vector[i];
    }

}

void create_adjacency_matrix(double **adjacency_matrix, double **data_list, int dimension, int num_of_vectors) {
    int i;
    int j;

    for(i = 0; i < num_of_vectors; ++i){
        for(j = 0; j < num_of_vectors; ++j){
            if (i == j) { /* we do not allow self loops */
                adjacency_matrix[i][j] = (double) 0;
            }

            else {
                adjacency_matrix[i][j] = exp(((-1)*calc_norm_form(data_list[i], data_list[j], dimension, 1))/2);
            }
        }
    }    
}

void create_diagonal_degree_matrix(double **diagonal_degree_matrix, double **adjacency_matrix, int num_of_vectors, int pow_minus_half) {
    int i;
    int j;

    for(i = 0; i < num_of_vectors; ++i){
        for(j = 0; j < num_of_vectors; ++j) {
            if (i==j) {
                if (pow_minus_half == 0) {
                    diagonal_degree_matrix[i][j] = sum_of_row(adjacency_matrix, i, num_of_vectors); /* Regular D matrix */
                }
                else { /* (pow_minus_half == 1) */
                    diagonal_degree_matrix[i][j] = (1/sqrt(sum_of_row(adjacency_matrix, i, num_of_vectors))); /* D^(-1/2) */
                }
            }

            else {
                diagonal_degree_matrix[i][j] = 0;
            }
        }
    }    
}

double sum_of_row(double** adjacency_matrix, int row_number, int num_of_vectors){
    int col;
    double sum;

    sum = 0;

    for(col = 0; col < num_of_vectors; ++col){
        sum = sum + adjacency_matrix[row_number][col];
    }

    return sum;
    
}

void matrix_multiplication_same_dimensions(double **result_matrix, double** matrix_1, double** matrix_2, int row_col_number) {
    int i;
    int j;
    int k;
    double** tmp_matrix;

    tmp_matrix = calloc(row_col_number, sizeof(double *));
    if (tmp_matrix == NULL) {
        exit(terminate_with_error());
    }

    for(i = 0; i < row_col_number; ++i){
        tmp_matrix[i] = calloc(row_col_number, sizeof(double));
        if (tmp_matrix[i] == NULL) {
            exit(terminate_with_error());
        }
    }

    for (i = 0; i < row_col_number; i++) {
        for (j = 0; j < row_col_number; j++) {
            tmp_matrix[i][j] = 0;
 
            for (k = 0; k < row_col_number; k++) {
                tmp_matrix[i][j] += matrix_1[i][k] * matrix_2[k][j];
            }
        }
    }
    for (i = 0; i < row_col_number; i++) {
        for (j = 0; j < row_col_number; j++) {
            result_matrix[i][j] = tmp_matrix[i][j]; 
        }
    }
}

void create_identity_matrix (double** identity_matrix, int row_col_num) {
    int i;
    int j;

    /* Assignment of values */
    for (i=0; i<row_col_num; i++) {
        for (j=0; j<row_col_num; j++) {
            if (i==j) {
                identity_matrix[i][j] = 1.0;
            }
            else { 
                identity_matrix[i][j] = 0.0;
            }
        }
    }
}


void create_L_norm_matrix (double **identity_matrix, double **ddg_matrix, double **L_norm_matrix, double** adjacency_matrix, int num_of_vectors) {

    create_diagonal_degree_matrix(L_norm_matrix, adjacency_matrix, num_of_vectors, 1);
    matrix_multiplication_same_dimensions (L_norm_matrix, L_norm_matrix, adjacency_matrix, num_of_vectors);
    create_diagonal_degree_matrix(ddg_matrix, adjacency_matrix, num_of_vectors, 1);
    matrix_multiplication_same_dimensions(L_norm_matrix, L_norm_matrix, ddg_matrix, num_of_vectors);
    create_identity_matrix(identity_matrix, num_of_vectors);
    matrix_subtraction(L_norm_matrix, identity_matrix, L_norm_matrix, num_of_vectors);

}

void print_square_matrix (double** matrix, int row_col_size) {

    int i;
    int j;

    for (i=0; i<row_col_size; ++i) {
        for (j=0; j<row_col_size; ++j) {

            if (j == (row_col_size-1)) {
                printf("%.4f\n", matrix[i][j]);
                fflush( stdout );
            }
            
            else {
                printf("%.4f\t", matrix[i][j]);
                fflush( stdout );
            }
            
        }
    }
    printf("***********************\n");

}


void matrix_subtraction (double **result_matrix, double** matrix_1, double** matrix_2, int row_col_number) {
    int i;
    int j;
    double** tmp_matrix;

    tmp_matrix = calloc(row_col_number, sizeof(double *));
    if (tmp_matrix == NULL) {
        exit(terminate_with_error());
    }

    for(i = 0; i < row_col_number; ++i){
        tmp_matrix[i] = calloc(row_col_number, sizeof(double));
        if (tmp_matrix[i] == NULL) {
            exit(terminate_with_error());
        }
    }

    for (i = 0; i < row_col_number; i++) {
        for (j = 0; j < row_col_number; j++) {
            tmp_matrix[i][j] = matrix_1[i][j]-matrix_2[i][j];
        }
    }
    for (i = 0; i < row_col_number; i++) {
        for (j = 0; j < row_col_number; j++) {
            result_matrix[i][j] = tmp_matrix[i][j];
        }
    }
}

void jacobi_algorithm(double **result_matrix, double **matrix_A, double *eigen_values, int dimension){
    double **matrix_P;
    double **matrix_V;
    double **copy_matrix_V;
    double **matrix_A_tag;
    int i;
    int j;
    int iteration;
    int *ij;
    int i_max;
    int j_max;
    double t;
    double s;
    double c;
    /* set up */
    /* place for matrix P */
    matrix_P = (double **) calloc (dimension, sizeof(double *));
    if(!matrix_P){
        exit(terminate_with_error());
    }
    for (i = 0; i < dimension; i++) {
        matrix_P[i] = (double *) calloc (dimension, sizeof(double));
           if(!matrix_P[i]){
        exit(terminate_with_error());
    }
    }

    /* build matrix_V to be I_matrix */
    matrix_V = (double **) calloc (dimension, sizeof(double *));
    if(!matrix_V){
        exit(terminate_with_error());
    }
    for (i = 0; i < dimension; i++) {
        matrix_V[i] = (double *) calloc (dimension, sizeof(double));
        if(!matrix_V[i]){
            exit(terminate_with_error());
        }
    }
    build_I_matrix(matrix_V, dimension);

   /* place for copy_matrix_V*/
    copy_matrix_V = (double **) calloc (dimension, sizeof(double *));
    if(!copy_matrix_V){
       exit(terminate_with_error());
    }
    for (i = 0; i < dimension; i++) {
        copy_matrix_V[i] = (double *) calloc (dimension, sizeof(double));
        if(!copy_matrix_V[i]){
            exit(terminate_with_error());
        }
    }

    /* place for matrix_A_tag */
    matrix_A_tag = (double **) calloc (dimension, sizeof(double *));
    if(!matrix_A_tag){
        exit(terminate_with_error());
    }
    for (i = 0; i < dimension; i++) {
        matrix_A_tag[i] = (double *) calloc (dimension, sizeof(double));
        if(!matrix_A_tag[i]){
            exit(terminate_with_error());
        }
    }

    copy_matrix(matrix_A, matrix_A_tag, dimension);
    /* end of set up */

    /*the algorithm */
    iteration = 0;
    while(iteration <= 100){
        ij = calc_Aij(matrix_A, dimension);
        i_max = ij[0];
        j_max = ij[1];
        t = calc_t(i_max, j_max, matrix_A);
        c = calc_c(t);
        s = calc_s(t, c);
        produce_matrix_p(matrix_P, dimension, i_max, j_max, c, s);
        print_square_matrix(matrix_P, dimension);
        produce_matrix_A(matrix_A, matrix_A_tag, c, s, dimension, i_max, j_max);
        copy_matrix(matrix_V, copy_matrix_V, dimension);
        matrix_multiplication_same_dimensions(matrix_V, copy_matrix_V, matrix_P, dimension);
        iteration += 1;
        if((is_diag(matrix_A, matrix_A_tag, dimension) == 1)){
            break;
        }
        copy_matrix(matrix_A, matrix_A_tag, dimension);
    }
    for(i = 0; i < dimension; i++){
        for(j = 0; j < dimension; j++){
            result_matrix[i][j] = matrix_V[i][j];
        }
    }
    calc_eigen_values(matrix_A_tag, eigen_values, dimension);

}

void produce_matrix_A(double **matrix_A, double **matrix_A_tag, double c, double s, int dimension, int i_max, int j_max) {
    int r;
    /* calc A[i][i] and A[j][j] */
    matrix_A[i_max][i_max] = (c * c * matrix_A_tag[i_max][i_max]) + (s * s * matrix_A_tag[j_max][j_max]) - 2 * s * c * matrix_A_tag[i_max][j_max];
    matrix_A[j_max][j_max] = (s * s * matrix_A_tag[i_max][i_max]) + (c * c * matrix_A_tag[j_max][j_max]) + 2*s*c*matrix_A_tag[i_max][j_max];
    /* defined A[i][j], A[j][i] = 0 */
    matrix_A[i_max][j_max] = 0;
    matrix_A[j_max][i_max] = 0;
    for( r = 0; r < dimension; r++ ){
        if(r != i_max && r != j_max){
            matrix_A[r][i_max] = (c * matrix_A_tag[r][i_max]) - (s * matrix_A_tag[r][j_max]);
            matrix_A[r][j_max] = (c * matrix_A_tag[r][j_max]) + (s * matrix_A_tag[r][i_max]);
        }
    }
}

void produce_matrix_p(double **matrix,
    int dimension,
    int i_max,
    int j_max,
    double c,
    double s)
    {
    build_I_matrix(matrix, dimension);
    matrix[i_max][i_max] = c;
    matrix[j_max][j_max] = c;
    matrix[i_max][j_max] = s;
    matrix[j_max][i_max] = -s;
}

double off(double **matrix, int dimension){
    double sum;
    int i, j;
    sum = 0;
    for (i = 0; i < dimension; i++){
        for (j = 0; j < dimension; j++) {
            if (i!=j) {
                sum += ((matrix[i][j])*(matrix[i][j]));
            }
        }
    }
    return sum;
}

int is_diag(double **matrix_A, double **matrix_old, int dimension){
    float EPSILON;
    double off_A;
    double off_old;
    EPSILON = pow(10,-5);
    off_A = off(matrix_A, dimension);
    off_old = off(matrix_old, dimension); 
    return ((off_old - off_A) <= EPSILON);
}

void copy_matrix (double **from, double **to, int dimention) {
    int j;
    int i;
    for (i = 0; i < dimention; i++) {
        for (j = 0; j < dimention; j++) {
            to[i][j] = from[i][j];
        }
    }
}

int* calc_Aij(double **matrix_A, int dimention){
    int i;
    int j;
    int j_max;
    int i_max;
    int *ret;
    double max_val;
    max_val = 0;
    for (i = 0; i < dimention; i++){
        for (j=i + 1; j < dimention; j++) {
            if (fabs(matrix_A[i][j]) > max_val) {
                max_val = fabs(matrix_A[i][j]);
                i_max = i;
                j_max = j;
            }
        }
    }
    ret = (int*) malloc(sizeof(int) * 2);
    ret[0] = i_max;
    ret[1] = j_max;
    return ret;
}

double calc_t(int i_max, int j_max, double **matrix_A){
    double theta;
    double sign_theta;
    double t;
    theta = ((matrix_A[j_max][j_max]) - (matrix_A[i_max][i_max])) / (2 * matrix_A[i_max][j_max]);
    if (theta >= 0) {
        sign_theta = 1;
    } else {
        sign_theta = -1;
    }   
    t = sign_theta / (fabs(theta) + (sqrt((theta * theta) + 1)));
    return t;
}

double calc_c(double t){
    double c;
    c = 1 / (sqrt(t * t + 1));
    return c;
}

double calc_s(double t, double c){
    return c * t;
}

void build_I_matrix(double **mat, int dimension){
    int i;
    int j;
    for (i = 0; i < dimension; i++){
        for (j = 0; j < dimension; j++){
            if (i == j) {
                mat[i][j] = 1;
            } else {
                mat[i][j] = 0;
            }
        }
    }
}

void multiply_matrixes(double **ret_matrix, double **mat1, double **mat2, int dimension) {
    int i, j, k;
  
    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            ret_matrix[i][j] = 0;
            for (k = 0; k < dimension; k++) {
                ret_matrix[i][j] += mat1[i][k] * mat2[k][j];
            }
         }
     }
}

void memory_free(double **matrix, int n){
    int i;
    for (i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int calculate_k_from_eigengap (double* eigenvalues, int num_of_eigenvalues) {
    
    int i;
    int k;
    int max_search_index;
    double max_diff;
    double temp_diff;

    k=0;
    max_search_index = (num_of_eigenvalues / 2);
    max_diff=0;

    for (i=0; i<max_search_index; ++i) {
        
        temp_diff = eigenvalues[i] - eigenvalues[i+1]; /* assuming the eigenvalues are decreasingly ordered */
        if (temp_diff > max_diff) {
            max_diff = temp_diff;
            k = i;
        }
    }

    return k+1; /* since the loop has i = 0,1,... and the requirement is i = 1,2, ... */
    
}

void calc_eigen_values(double ** matrix_A_tag, double *eigen_values, int dimension){
    int i;
    for(i = 0; i < dimension; i++){
        eigen_values[i] = matrix_A_tag[i][i];
    }
}
