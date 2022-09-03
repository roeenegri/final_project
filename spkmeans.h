#ifndef FINAL_CLEAN_2_SPKMEANS_H
#define FINAL_CLEAN_2_SPKMEANS_H

void produce_matrix_A(double **matrix_A,
    double **matrix_A_tag,
    double c,
    double s,
    int dimension,
    int i_max,
    int j_max);
void produce_matrix_p(double **matrix,
    int dimension,
    int i_max,
    int j_max,
    double c,
    double s);
double off(double **matrix, int dimension);
int is_diag(double **matrix_A, double **matrix_old, int dimension);
void copy_matrix (double **from, double **to, int dimention);
int* calc_Aij(double **matrix_A, int dimention);
double calc_t(int i_max, int j_max, double **matrix_A);
double calc_c(double t);
double calc_s(double t, double c);
void build_I_matrix(double **mat, int dimension);
void multiply_matrixes(double **ret_matrix, double **mat1, double **mat2, int dimension);
double** extract_k_first_vectors(double**, int, int);
int check_centroid_diff(double**, double**, int, int);
void calc_centroids(double**, int *, double **, int, int);
int matching_cluster(double*, double **, int, int);
double calc_norm_form(double*, double *, int, int);
int extract_vectors (char*, double**, int, int);
int terminate_with_error (void);
int terminate_invalid_input (void);
void sum_two_vectors (double **, double *, int, int);
int find_dimension(char*);
int find_num_of_vectors(char*);
void create_adjacency_matrix(double**, double**, int, int);
void create_diagonal_degree_matrix(double**, double**, int, int);
double sum_of_row(double**, int, int);
void matrix_multiplication_same_dimensions(double**, double**, double**, int);
void create_identity_matrix (double**, int);
void create_L_norm_matrix (double**, double**, double**, double**, int);
void print_square_matrix (double**, int);
void matrix_subtraction (double**, double**, double**, int);
void jacobi_algorithm(double **result_matrix, double **matrix_A, double * eigen_values, int dimension);
void memory_free(double**, int);
int calculate_k_from_eigengap (double*, int);
void calc_eigen_values(double**, double*, int);
void transpose_matrix(double **result_matrix, double **matrix, int n);

#endif
