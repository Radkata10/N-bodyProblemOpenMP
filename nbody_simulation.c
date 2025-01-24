#include <stdio.h>     
#include <stdlib.h>   
#include <math.h>     
#include <omp.h>      


#define N 100  // Number of particles
#define G 6.67430e-11  // Gravitational constant
#define TIME_STEP 0.01  // Time step
#define STEPS 5  // Number of time steps

double masses[N];
double pos[N][3], vel[N][3], forces[N][3];

void initialize_particles() {
    for (int i = 0; i < N; i++) {
        masses[i] = 1.0;  
        for (int j = 0; j < 3; j++) {
            pos[i][j] = rand() / (double)RAND_MAX;  
            vel[i][j] = 0.0;  
            forces[i][j] = 0.0;  
        }
    }
}

void compute_forces() {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double rij[3], distance_squared = 0.0, distance, force;

            // Calculate distance vector and squared distance
            for (int k = 0; k < 3; k++) {
                rij[k] = pos[j][k] - pos[i][k];
                distance_squared += rij[k] * rij[k];
            }

            distance = sqrt(distance_squared);
            if (distance > 1e-5) {  
                force = (G * masses[i] * masses[j]) / (distance_squared * distance);

                // Update forces
                for (int k = 0; k < 3; k++) {
                    double f = force * rij[k];
                    #pragma omp atomic
                    forces[i][k] += f;

                    #pragma omp atomic
                    forces[j][k] -= f;
                }
            }
        }
    }
}

void update_particles() {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 3; k++) {
            // Update velocity
            vel[i][k] += (forces[i][k] / masses[i]) * TIME_STEP;

            // Update position
            pos[i][k] += vel[i][k] * TIME_STEP;

            // Reset forces for the next step
            forces[i][k] = 0.0;
        }
    }
}

int main() {
    initialize_particles();

    for (int step = 0; step < STEPS; step++) {
        compute_forces();
        update_particles();
    }

    
    for (int i = 0; i < N; i++) {
        printf("Particle %d: Position = (%f, %f, %f)\n", i, pos[i][0], pos[i][1], pos[i][2]);
    }

    return 0;
}
