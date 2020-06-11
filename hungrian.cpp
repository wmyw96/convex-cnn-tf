#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <cstdlib>

using namespace std;

#define MAXN 2222

class adj_mat{
public:
    int n;
    float mat[MAXN][MAXN];
    adj_mat(int _n){
        n = _n;
        memset(mat, 0, sizeof mat);
    }
    adj_mat(adj_mat* b){
        n = b->n;
        memset(mat, 0, sizeof mat);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                mat[i][j] = b->mat[i][j];
    }
    void fill(float value){
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                mat[i][j] = value;
    }
    float get_row_min(int i){
        float tmp = 1e9;
        for (int j = 0; j < n; ++j)
            if (mat[i][j] < tmp){
                tmp = mat[i][j];
            } 
        return tmp;
    }
    void print(){
        for (int i = 0; i < n; ++i){
            for (int j = 0; j < n; ++j)
                printf("%.4f ", mat[i][j]);
            printf("\n");
        }
    }
};

class indicator{
public:
    int n;
    int mat[MAXN][MAXN];
    indicator(int _n){
        n = _n;
        memset(mat, 0, sizeof mat);
    }
    indicator(indicator* b){
        n = b->n;
        memset(mat, 0, sizeof mat);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                mat[i][j] = b->mat[i][j];
    }
    void fill(int value){
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                mat[i][j] = value;
    }
    int mark_row(int i){
        for (int j = 0; j < n; ++j)
            if (mat[i][j] == 1)
                return j;
        return -1;
    }
    int mark_col(int j){
        for (int i = 0; i < n; ++i)
            if (mat[i][j] == 1)
                return i;
        return -1;
    }
    void print(){
        for (int i = 0; i < n; ++i){
            for (int j = 0; j < n; ++j)
                printf("%d", mat[i][j]);
            printf("\n");
        }
    }
};

void read_matrix(const char* filename, adj_mat* cost, int n){
    FILE *fo = fopen(filename, "rt+");
    float tmp;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j){
            fscanf(fo, "%f", &tmp);
            cost->mat[i][j] = tmp;
        }
    fclose(fo);
}

void find_matching(adj_mat* cost, indicator* &result){
    int n = cost->n;
    indicator* stars = new indicator(n);
    stars->fill(0.0);
    indicator* primes = new indicator(n);
    primes->fill(0.0);

    int row_cov[n + 1];
    int col_cov[n + 1];
    memset(row_cov, 0, sizeof row_cov);
    memset(col_cov, 0, sizeof col_cov);

    // step 1: reduce matrix
    for (int i = 0; i < n; ++i){
        float row_min = cost->get_row_min(i);
        for (int j = 0; j < n; ++j)
            cost->mat[i][j] -= row_min;
    }
    //printf("step 1: reduce matrix solved successfully...\n");
    //cost->print();

    // step 2: find zero
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (cost->mat[i][j] < 1e-9 && !row_cov[i] && !col_cov[j]){
                stars->mat[i][j] = 1;
                row_cov[i] = 1;
                col_cov[j] = 1;
            }
    //printf("step 2: find zero solved successfully...\n");
    //stars->print();
    memset(row_cov, 0, sizeof row_cov);
    memset(col_cov, 0, sizeof col_cov);

    while (true){
        // step 3
        step3:
        //printf("step 3: check matched pairs ...\n");
        int matched = 0;
        for (int j = 0; j < n; ++j){
            int idx = stars->mark_col(j);
            if (idx > -1){
                col_cov[j] = 1;
                matched ++;
            }
        }
        //printf("Total %d matches.\n", matched);
        if (matched == n){
            result = new indicator(stars);
            return;
        }

        //printf("step 4: current cost matrix:\n");
        //cost->print();
        // step 4
        step4:
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j){
                if (cost->mat[i][j] < 1e-9 && !row_cov[i] && !col_cov[j]){
                    primes->mat[i][j] = 1;
                    int col = stars->mark_row(i);
                    //printf("find minimum at (%d,%d), current cost mat:\n", i, j);
                    //cost->print();
                    if (col < 0){
                        // step 5
                        // swap_stars_and_primes
                        int prime_row = i;
                        int prime_col = j;
                        bool done = false;
                        while (!done){
                            int star_row_in_col = stars->mark_col(prime_col);
                            if (star_row_in_col < 0){
                                primes->mat[prime_row][prime_col] = 0;
                                stars->mat[prime_row][prime_col] = 1;
                                done = true;
                            }
                            else{
                                int prime_col_in_row = primes->mark_row(star_row_in_col);
                                primes->mat[prime_row][prime_col] = 0;
                                stars->mat[prime_row][prime_col] = 1;
                                stars->mat[star_row_in_col][prime_col] = 0;
                                prime_row = star_row_in_col;
                                prime_col = prime_col_in_row;
                            }
                        }
                        primes->fill(0);

                        memset(col_cov, 0, sizeof col_cov);
                        memset(row_cov, 0, sizeof row_cov);
                        goto step3;
                    }
                    else{
                        row_cov[i] = 1;
                        col_cov[col] = 0;
                    }
                }
            }
        //printf("step 4 finished: \n");
        //printf("row: ");
        //for (int i = 0; i < n; ++i)
        //    printf(i + 1 < n ? "%d " : "%d\n", row_cov[i]);
        //printf("col: ");
        //for (int i = 0; i < n; ++i)
        //    printf(i + 1 < n ? "%d " : "%d\n", col_cov[i]);

        // step 6
        float min_value = 1e9;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (!row_cov[i] && !col_cov[j] && cost->mat[i][j] < min_value){
                    min_value = cost->mat[i][j];
                }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (!row_cov[i] && !col_cov[j]){
                    cost->mat[i][j] -= min_value;
                }
                else if (row_cov[i] && col_cov[j]){
                    cost->mat[i][j] += min_value;
                }
        goto step4;
    }
}

int main(int argc, char** argv){
    int n = atoi(argv[1]);
    clock_t start, end;
    float elapsedTime;
    start = clock();
    
    printf("run hungarian algorithm (n = %d)", n);
    adj_mat* cost_mat = new adj_mat(n);
    //printf("build successfully..\n");
    read_matrix("mat.in", cost_mat, n);
    //printf("read matrix successfully..\n");
    adj_mat* cost = new adj_mat(cost_mat);

    indicator* result = new indicator(n);
    //cost->print();
    find_matching(cost, result);
    //result->print();

    float ans = 0.0;
    for (int i = 0; i < n; ++i){
        ans += cost_mat->mat[i][result->mark_row(i)];
    }
    FILE *f1 = fopen("mat_perm.out", "w");
    for (int i = 0; i < n; ++i)
        fprintf(f1, i + 1 < n ? "%d " : "%d\n", result->mark_row(i));
    fclose(f1);
    FILE *f2 = fopen("mat_val.out", "w");
    fprintf(f2, "%.4f\n", ans);
    fclose(f2);
    end = clock();
    elapsedTime = (float)(end - start) / CLOCKS_PER_SEC;
    cout << ": total time =  " << elapsedTime << " s" << endl;
}

