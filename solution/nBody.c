#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SOFTENING 1e-9f

typedef struct{
	float x;
	float y;
	float z;
	float vx;
	float vy;
	float vz;
}Body;                                                                 


MPI_Datatype mpi_body_type; // nuovo tipo di dato da utilizzare

// MPI
int world_rank;            
int world_size;               


int nBodies; // numero di corpi totali
int nIters; //Numero di iterazioni
int printFreq; //frequenza di stampa
float dt; // timestamp

int nBodies; //numero corpi
int bodyProc; // numero medio di corpi da assegnare agli slave
int remaind; //particelle rimanenti da assegnare agli slave quando le particelle non sono in numero ideale
int nIters; //numero di iterazioni
int first; //prima particella assegnata allo slave
int last; //ultima particella assegnata allo slave
int myBodies; // numero corpi da assegnare ad ogni slave

Body *bodies;  // insieme di corpi

//funzioni
void randomizeBodies();
void bodyForce();
void updatePositions();
void gatherPointToPoint(int i); //funzione gather point to point
void updateAllSlave(); // funzione che sincronizza le particelle di tutti i processori
void printBodies(double t);
void calculateRange(); //Calcola il range da assegnare ad ogni processore

int main(int argc, char* argv[]) {
	
    double time;                                /* Tempo corrente */
    double start_time;                          /* Tempo di inizio esecuzione */
    double end_time;                            /* Tempo di fine esecuzione */
	
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
	
    if (argc == 4){
		nBodies = strtol(argv[1], NULL, 10);
		nIters = strtol(argv[2], NULL, 10);
		printFreq = strtol(argv[3], NULL, 10);
		dt = 0.1f;
	}
	else{
		printf("Inserire\n");
		printf("1) Numero corpi\n");
		printf("2) Numero iterazioni\n");
		printf("3) Frequenza di stampa\n");
	}
	if (nBodies <= 0 || nIters <= 0){               /* Esce se vengono inseriti parametri non accettabili */
		MPI_Finalize();
		exit(0);
	}
	
	bodyProc = nBodies/world_size;
	remaind = nBodies % world_size;
	calculateRange();
	
    printf("Sono il processo %d e analizzerò %d particelle [%d-%d]\n",world_rank,myBodies,first,last);
	
	bodies = malloc(nBodies * sizeof(Body));
	
    // tipo di dato contiguo
    MPI_Type_contiguous(6, MPI_FLOAT, &mpi_body_type);
    MPI_Type_commit(&mpi_body_type);
	
	randomizeBodies(); 
	
	start_time = MPI_Wtime();
	
	if (world_rank == 0){
		printBodies(0.0); 
	}
	
	int i;
    for(i = 1; i <= nIters; i++){
		time = i * dt;
		bodyForce();                           
		updatePositions();                  
		updateAllSlave();   
		if((i % printFreq) == 0 && world_rank == 0){
			printBodies(time);
		}
	}
	end_time = MPI_Wtime();
	
	if(world_rank == 0){
		printf("Tempo trascorso = %f secondi\n", end_time - start_time);
	}
    /* Dealloco la memoria */
    MPI_Type_free(&mpi_body_type);
    free(bodies);
    /* --- */
	
    MPI_Finalize();             /* Termino MPI */
    return 0;
}

void calculateRange(){
	if (remaind == 0) {
		myBodies = bodyProc;
		first = world_rank * bodyProc;
		last = ((world_rank + 1) * bodyProc)-1;
		} else if (remaind > 0) {
		if(world_rank < remaind) {
			myBodies = bodyProc + 1;
			first = world_rank * bodyProc;
			last = ((world_rank + 1 ) * bodyProc);
			}else if(world_rank >= remaind) {
			myBodies = bodyProc;
			first = world_rank * bodyProc + remaind;
			last = ((world_rank + 1) * bodyProc) + remaind - 1;
		}
	}
}
void randomizeBodies(){
    int i;	
	
    for(i = first; i <= last; i++){
		bodies[i].x = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
		bodies[i].y = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
		bodies[i].z = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
		bodies[i].vx = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
		bodies[i].vy = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
		bodies[i].vz = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
	}
    updateAllSlave();
}

void bodyForce(){
	
	int i;
	for (i = first; i <= last; i++){
		float Fx = 0.0f;
		float Fy = 0.0f;
		float Fz = 0.0f;
		int j;
		for (j = 0; j < nBodies; j++){
			float dx = bodies[j].x - bodies[i].x;
			float dy = bodies[j].y - bodies[i].y;
			float dz = bodies[j].z - bodies[i].z;
			float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
			float invDist = 1.0f/sqrtf(distSqr);
			float invDist3 = invDist * invDist * invDist;
			
			Fx += dx * invDist3;
			Fy += dy * invDist3;
			Fz += dz * invDist3;
		}
		
		bodies[i].vx += dt * Fx;
		bodies[i].vy += dt * Fy;
		bodies[i].vz += dt * Fz;
	}
}

void updatePositions(){
	int i;
	for (i = first; i <= last; i++){
		bodies[i].x += bodies[i].vx * dt;
		bodies[i].y += bodies[i].vy * dt;
		bodies[i].y += bodies[i].vz * dt;
	}
}

void printBodies(double time){
	
	int b;
	printf("Tempo corrente:%.2f\n", time);
	
	for(b = 0; b < nBodies; b++){
		printf("Body:%3d ", b);
		printf("| x:%10.3e ", bodies[b].x);
		printf("| y:%10.3e ", bodies[b].y);
		printf("| z:%10.3e ", bodies[b].z);
		printf("| vx:%10.3e ", bodies[b].vx);
		printf("| vy:%10.3e\n ", bodies[b].vy);
		printf("| z:%10.3e\n ", bodies[b].vz);
		printf("\n");
	}
}

void updateAllSlave(){
	
	int i;
	
	for(i = 0; i < world_size; i++){
        gatherPointToPoint(i);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
}

void gatherPointToPoint(int root){
	if (world_rank != root){
		MPI_Send(bodies + first, myBodies, mpi_body_type, root, 0, MPI_COMM_WORLD);
	}
	else{
		int i;
		for(i = 0; i < world_size; i++){
			if(world_rank != i){
				int start;
				int receive;
				if(i < remaind) {
					receive = bodyProc + 1;
					start= receive * i; 
					} else {
					receive = bodyProc;
					start= receive * i + remaind;
				}
				MPI_Recv(bodies + start, receive, mpi_body_type, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
	}
}

