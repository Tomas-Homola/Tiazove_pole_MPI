#define _USE_MATH_DEFINES

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

//#define R 6378000.0
//#define D 200000.0
//#define N 8102

#define GM 398600.5
#define N 160002
#define D 50000.0
#define R 6378000.0
// Homola_MPI.cpp

using std::cos;
using std::sin;
using std::sqrt;

int main(int argc, char** argv)
{
    int nprocs = 0;
    int myRank = 0;
    // Initialize MPI environment
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // get num of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank); // assign ranks

    double* B = new double[N] {0.0};
    double* L = new double[N] {0.0};
    double Brad = 0.0, Lrad = 0.0, H = 0.0, u2n2 = 0.0;
    double temp = 0.0;

    // suradnice bodov X_i
    double* X_x  = new double[N] {0.0};
    double* X_y  = new double[N] {0.0};
    double* X_z  = new double[N] {0.0};
    double xNorm = 0.0;
    
    // suradnice bodov s_j
    double* s_x = new double[N] {0.0};
    double* s_y = new double[N] {0.0};
    double* s_z = new double[N] {0.0};
    
    // suradnicce normal v x_i
    double* n_x = new double[N] {0.0};
    double* n_y = new double[N] {0.0};
    double* n_z = new double[N] {0.0};
    
    // g vektor
    double* g = new double[N] {0.0};

    // r vector
    double r_x = 0.0;
    double r_y = 0.0;
    double r_z = 0.0;
    double rNorm = 0.0;
    double rNorm3 = 0.0;

    // dot product of vector r with normal n[i]
    double Kij = 0.0;

    // load data, only process 0
    if (myRank == 0)
    {
        //printf("Data reading started\n");
        FILE* file = nullptr;
        // BL-160002,dat
        file = fopen("BL-160002.dat", "r");
        if (file == nullptr)
        {
            delete[] B;
            delete[] L;
            delete[] g;
            delete[] X_x;
            delete[] X_y;
            delete[] X_z;
            delete[] s_x;
            delete[] s_y;
            delete[] s_z;
            delete[] n_x;
            delete[] n_y;
            delete[] n_z;

            printf("file did not open\n");
            return -1;
        }
        
        for (int i = 0; i < N; i++)
        {
            int result = fscanf(file, "%lf %lf %lf %lf %lf", &B[i], &L[i], &H, &g[i], &u2n2);
            g[i] = -g[i] * 0.00001;

            //g[i] = u2n2;
            Brad = B[i] * M_PI / 180.0;
            Lrad = L[i] * M_PI / 180.0;

            // pridat (R + H)
            X_x[i] = (R + H) * cos(Brad) * cos(Lrad);
            X_y[i] = (R + H) * cos(Brad) * sin(Lrad);
            X_z[i] = (R + H) * sin(Brad);

            // pridat (R + H -D)
            s_x[i] = (R + H -D) * cos(Brad) * cos(Lrad);
            s_y[i] = (R + H -D) * cos(Brad) * sin(Lrad);
            s_z[i] = (R + H -D) * sin(Brad);

            xNorm = sqrt(X_x[i] * X_x[i] + X_y[i] * X_y[i] + X_z[i] * X_z[i]);
            n_x[i] = -X_x[i] / xNorm;
            n_y[i] = -X_y[i] / xNorm;
            n_z[i] = -X_z[i] / xNorm;

            //if (i < 10)
              //  printf("g[%d]: %.5lf\n", i, g[i]);
            //    printf("X[%d] = (%.2lf, %.2lf, %.2lf)\n", i, X_x[i], X_y[i], X_z[i]);
        }

        fclose(file);
    }

    //if (myRank == 0)
    //    printf("Data loading done\n");
    
    // broadcast loaded data to all processes
    MPI_Bcast(X_x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(X_y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(X_z, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(s_x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(s_y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(s_z, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_z, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(g,   N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // MPI variables
    int nlocal = (N / nprocs) + 1; // pre vsetky procesy rovnake, posledny bude naddimenzovany
    int nlast = N - nlocal * (nprocs - 1);

    if (myRank == nprocs - 1)
        printf("P%02d-> nlocal: %d\n", myRank, nlast);
    else
        printf("P%02d-> nlocal: %d\n", myRank, nlocal);

    // vytvorenie matice systemu rovnic
    double* Alocal = new double[nlocal * N] {0.0};
    int ij = -1;
    int iGlobal = -1;

    if (myRank == nprocs - 1) // if last process
    {
        for (int i = 0; i < nlast; i++) // from 0 to nlast - 1
        {
            // compute iGlobal index
            iGlobal = i + nlocal * myRank;

            for (int j = 0; j < N; j++) // from 0 to N-1
            {
                // compute vector r & its norm
                r_x = X_x[iGlobal] - s_x[j];
                r_y = X_y[iGlobal] - s_y[j];
                r_z = X_z[iGlobal] - s_z[j];

                rNorm = sqrt(r_x * r_x + r_y * r_y + r_z * r_z);

                rNorm3 = rNorm * rNorm * rNorm;

                // dot product of vector r and normal n_i
                Kij = r_x * n_x[iGlobal] + r_y * n_y[iGlobal] + r_z * n_z[iGlobal];
                //if (i == j && i < 10)
                //    printf("Kij: %.4lf\n", Kij);

                // compute 
                ij = i * N + j;
                Alocal[ij] = (1.0 / (4.0 * M_PI * rNorm3)) * Kij;
            }
        }
    }
    else
    {
        for (int i = 0; i < nlocal; i++) // from 0 to nlocal-1
        {
            // compute iGlobal index
            iGlobal = i + nlocal * myRank;

            for (int j = 0; j < N; j++) // from 0 to N-1
            {
                // compute vector r & its norm
                r_x = X_x[iGlobal] - s_x[j];
                r_y = X_y[iGlobal] - s_y[j];
                r_z = X_z[iGlobal] - s_z[j];

                rNorm = sqrt(r_x * r_x + r_y * r_y + r_z * r_z);

                rNorm3 = rNorm * rNorm * rNorm;

                // dot product of vector r and normal n_i
                Kij = r_x * n_x[iGlobal] + r_y * n_y[iGlobal] + r_z * n_z[iGlobal];
                
                // compute 
                ij = i * N + j;
                Alocal[ij] = (1.0 / (4.0 * M_PI * rNorm3)) * Kij;
            }
        }
    }

    //########## BCGS linear solver ##########//

    double* sol = new double[nlocal * nprocs]; // [nlocal * nprocs]; vektor x^0 -> na ukladanie riesenia systemu
    double* r_hat = new double[nlocal * nprocs]; // [nlocal * nprocs]; vektor \tilde{r} = b - A.x^0;
    double* r = new double[nlocal * nprocs]; // [nlocal * nprocs]; vektor pre rezidua
    double* p = new double[nlocal * nprocs]; // [nlocal * nprocs]; pomocny vektor na update riesenia
    double* v = new double[nlocal * nprocs]; // [nlocal * nprocs]; pomocny vektor na update riesenia
    double* vlocal = new double[nlocal] {0.0}; // [nlocal]; local vektor v na parcialne vysledky, potom MPI_Allgather do v
    double* s = new double[nlocal * nprocs]; // [nlocal * nprocs]; pomocny vektor na update riesenia
    double* t = new double[nlocal * nprocs]; // [nlocal * nprocs]; pomocny vektor na update riesenia
    double* tlocal = new double[nlocal]{0.0}; // [nlocal]; local vektor t na parcialne vysledky, potom MPI_Allgather do t

    double beta = 0.0;
    double rhoNew = 1.0;
    double rhoOld = 0.0;
    double alpha = 1.0;
    double omega = 1.0;

    double tempDot = 0.0;
    double tempDot2 = 0.0;
    double sNorm = 0.0;

    int MAX_ITER = 500;
    double TOL = 1.0E-6;
    int iter = 1;

    double rezNorm = 0.0;
    for (int i = 0; i < nlocal * nprocs; i++) // set all to zero
    {
        sol[i] = 0.0;
        p[i] = 0.0; // = 0
        v[i] = 0.0; // = 0
        s[i] = 0.0;
        t[i] = 0.0;
        r[i] = 0.0;
        r_hat[i] = 0.0;
    }

    for (int i = 0; i < N; i++) // N <= nlocal * nprocs
    {
        r[i] = g[i];
        r_hat[i] = g[i];
        rezNorm += r[i] * r[i];
    }

    if (myRank == 0)
        printf("||r0||: %.10lf\n", sqrt(rezNorm));

    rezNorm = 0.0;

    // START BCGS
    do
    {
        rhoOld = rhoNew; // save previous rho_{i-2}
        rhoNew = 0.0; // compute new rho_{i-1}
        for (int i = 0; i < N; i++) // dot(r_hat, r), N <= nlocal*nprocs
            rhoNew += r_hat[i] * r[i];

        if (rhoNew == 0.0)
            return -1;

        if (iter == 1)
        {
            //printf("iter 1 setup\n");
            for (int i = 0; i < N; i++)
                p[i] = r[i];
        }
        else
        {
        beta = (rhoNew / rhoOld) * (alpha / omega);
        for (int i = 0; i < N; i++) // update vector p^(i)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // PARALELNE MATICA.VEKTOR compute vector v = A.p
        if (myRank == nprocs - 1)
        {
            for (int i = 0; i < nlast; i++)
            {
                vlocal[i] = 0.0;

                for (int j = 0; j < N; j++)
                {
                    ij = i * N + j;
                    vlocal[i] += Alocal[ij] * p[j];
                }
            }
        }
        else
        {
            for (int i = 0; i < nlocal; i++)
            {
                vlocal[i] = 0.0;

                for (int j = 0; j < N; j++)
                {
                    ij = i * N + j;
                    vlocal[i] += Alocal[ij] * p[j];
                }
            }
        }

        // gather all partial solutions to vector "v"
        MPI_Allgather(vlocal, nlocal, MPI_DOUBLE, v, nlocal, MPI_DOUBLE, MPI_COMM_WORLD);

        // compute alpha
        tempDot = 0.0;
        for (int i = 0; i < N; i++)
            tempDot += r_hat[i] * v[i];

        alpha = rhoNew / tempDot;

        // compute vektor s
        sNorm = 0.0;
        for (int i = 0; i < N; i++)
        {
            s[i] = r[i] - alpha * v[i];
            sNorm += s[i] * s[i];
        }

        sNorm = sqrt(sNorm);
        if (sNorm < TOL) // check if ||s|| is small enough
        {
            for (int i = 0; i < N; i++) // update solution x
                sol[i] = sol[i] + alpha * p[i];

            if (myRank == 0)
                printf("BCGS stop:   ||s||(= %.10lf) is small enough, iter: %3d\n", sNorm, iter);

            break;
        }

        // PARALELNE MATICA.VEKTOR compute vector t = A.s
        if (myRank == nprocs - 1)
        {
            for (int i = 0; i < nlast; i++)
            {
                tlocal[i] = 0.0;

                for (int j = 0; j < N; j++)
                {
                    ij = i * N + j;
                    tlocal[i] += Alocal[ij] * s[j];
                }
            }
        }
        else
        {
            for (int i = 0; i < nlocal; i++)
            {
                tlocal[i] = 0.0;

                for (int j = 0; j < N; j++)
                {
                    ij = i * N + j;
                    tlocal[i] += Alocal[ij] * s[j];
                }
            }
        }

        // gather all partial solutions to vector "t"
        MPI_Allgather(tlocal, nlocal, MPI_DOUBLE, t, nlocal, MPI_DOUBLE, MPI_COMM_WORLD);

        // compute omega
        tempDot = 0.0; tempDot2 = 0.0;
        for (int i = 0; i < N; i++)
        {
            tempDot += t[i] * s[i];
            tempDot2 += t[i] * t[i];
        }
        omega = tempDot / tempDot2;

        rezNorm = 0.0;
        for (int i = 0; i < N; i++)
        {
            sol[i] = sol[i] + alpha * p[i] + omega * s[i]; // update solution x
            r[i] = s[i] - omega * t[i]; // compute new residuum vector
            rezNorm += r[i] * r[i]; // compute residuum norm
        }

        rezNorm = sqrt(rezNorm);
        if (myRank == 0)
            printf("iter: %3d    ||r||: %.10lf\n", iter, rezNorm);

        if (rezNorm < TOL)
        {
            if (myRank == 0)
                printf("BCGS stop iter: ||r|| is small enough\n");

            break;
        }

        iter++;

    } while ((iter < MAX_ITER) && (rezNorm > TOL));

    delete[] r_hat;
    delete[] r;
    delete[] p;
    delete[] v;
    delete[] vlocal;
    delete[] s;
    delete[] t;
    delete[] tlocal;
    delete[] Alocal;

    //########## EXPORT DATA ##########//
    double* u = new double[nlocal * nprocs] {0.0}; // [nlocal * nprocs]
    double* ulocal = new double[nlocal] {0.0}; // [nlocal]
    double Gij = 0.0;

    //// compute solution u
    if (myRank == nprocs - 1) // last process
    {
        for (int i = 0; i < nlast; i++)
        {
            // compute iGlobal index
            iGlobal = i + nlocal * myRank;
            
            for (int j = 0; j < N; j++) // N <= nlocal * nprocs
            {
                r_x = X_x[iGlobal] - s_x[j];
                r_y = X_y[iGlobal] - s_y[j];
                r_z = X_z[iGlobal] - s_z[j];

                rNorm = sqrt(r_x * r_x + r_y * r_y + r_z * r_z);

                Gij = 1.0 / (4.0 * M_PI * rNorm);

                ulocal[i] += sol[j] * Gij;
            }
        }
    }
    else // other processes
    {
        for (int i = 0; i < nlocal; i++)
        {
            // compute iGlobal index
            iGlobal = i + nlocal * myRank;

            for (int j = 0; j < N; j++) // N <= nlocal * nprocs
            {
                r_x = X_x[iGlobal] - s_x[j];
                r_y = X_y[iGlobal] - s_y[j];
                r_z = X_z[iGlobal] - s_z[j];

                rNorm = sqrt(r_x * r_x + r_y * r_y + r_z * r_z);

                Gij = 1.0 / (4.0 * M_PI * rNorm);

                ulocal[i] += sol[j] * Gij;
            }
        }
    }

    // gather local solutions u at process P0
    MPI_Allgather(ulocal, nlocal, MPI_DOUBLE, u, nlocal, MPI_DOUBLE, MPI_COMM_WORLD);

    // release unnecessary memory
    delete[] ulocal;
    delete[] sol;
    delete[] X_x;
    delete[] X_y;
    delete[] X_z;
    delete[] s_x;
    delete[] s_y;
    delete[] s_z;
    delete[] n_x;
    delete[] n_y;
    delete[] n_z;
    delete[] g;

    if (myRank == 0)
    {
        FILE* file = nullptr;
        file = fopen("outCorrect_MPI.dat", "w");
        if (file == nullptr)
        {
            delete[] B;
            delete[] L;
            delete[] u;

            printf("data export failed\n");
            return -2;
        }
    
        for (int i = 0; i < N; i++)
        {
            fprintf(file, "%.5lf\t%.5lf\t%.5lf\n", B[i], L[i], u[i]);
        }
        
        fclose(file);
    }
    
    // release memory
    delete[] B;
    delete[] L;
    delete[] u;

    MPI_Finalize();
}