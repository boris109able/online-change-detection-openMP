#include <stdio.h>
#include <time.h>
#include <string.h>
#include <random>//http://www.cplusplus.com/reference/random/normal_distribution/
#include <math.h>
#include <omp.h>//include openMP header
#include <chrono>

//B = 1.5:0.1:2.5 total 11 points
const int arraySize = 40;
const double shift = 0;//B = shift+step*i
const double step = 0.1;
const double epsilon = 0.5;
const long T = 100000;
const long runTime = 100000;
const int p = 20;
const int s = 4;
const char *fileloc = "F:/visual studio projects/onlinesimulation_openMP/result_openMP.txt";
const char *sep = "########################################################################\n";


void eachError(double *B, double *error);
void eachDelay(double *B, double *delay);

int main()
{

	double B[arraySize] = { 0 };
	for (int i = 0; i < arraySize; i++) {
		B[i] = shift + step * i;
	}

	double error[arraySize] = { 0 };
	double delay[arraySize] = { 0 };
	double errorTime;
	double delayTime;

	/*eachError(B, error);
	for (int i = 0; i < arraySize; i++) {
		printf("error[%d]: %f\n", i, *(error + i));
	}
	getchar();*/
	clock_t t1, t2;
	//write to file result.txt in the same folder as the project
	errno_t err;
	FILE* result;
	//write simulation information and header
	err = fopen_s(&result, fileloc, "a+");
	if (err == 0) {
		fprintf(result, sep);
		fprintf(result, "Simulation Result for Error\n");
		time_t ltime;
		char timebuf[26];
		time(&ltime);
		errno_t errtime;
		errtime = ctime_s(timebuf, 26, &ltime);
		if (errtime)
		{
			printf("ctime_s failed due to an invalid argument.");
			exit(1);
		}
		fprintf(result, "UNIX time and date:\t\t\t%s", timebuf);
		fclose(result);
	}
	else {
		printf("Error: file cannot be opened!");
	}
	//run simulation and write results
	t1 = clock();
	eachError(B, error);
	for (int i = 0; i < arraySize; i++) {
		printf("error[%d]: %f\n", i, *(error + i));
	}
	t2 = clock();
	errorTime = ((double)t2 - (double)t1) / CLOCKS_PER_SEC / runTime;
	t1 = clock();
	eachDelay(B, delay);
	for (int i = 0; i < arraySize; i++) {
		printf("delay[%d]: %f\n", i, *(delay + i));
	}
	t2 = clock();
	delayTime = ((double)t2 - (double)t1) / CLOCKS_PER_SEC / runTime;
	//write result to file
	err = fopen_s(&result, fileloc, "a+");
	if (err == 0) {
		for (int i = 0; i < arraySize; i++) {
			fprintf(result, "B: %f		", B[i]);
			fprintf(result, "error: %f		", error[i]);
			fprintf(result, "delay: %f\n", delay[i]);
		}
		fprintf(result, "errorTime: %f		", errorTime);
		fprintf(result, "delayTime: %f\n", delayTime);
		fclose(result);
	}
	else {
		printf("Error: file cannot be opened!");
	}
	//write result as vector
	err = fopen_s(&result, fileloc, "a+");
	if (err == 0)
	{
		fprintf(result, "B= [");
		for (int i = 0; i < arraySize - 1; i++) {
			fprintf(result, "%f ,", B[i]);
		}
		fprintf(result, "%f]\n", B[arraySize - 1]);


		fprintf(result, "error= [");
		for (int i = 0; i < arraySize - 1; i++) {
			fprintf(result, "%f ,", error[i]);
		}
		fprintf(result, "%f]\n", error[arraySize - 1]);

		fprintf(result, "delay= [");
		for (int i = 0; i < arraySize - 1; i++) {
			fprintf(result, "%f ,", delay[i]);
		}
		fprintf(result, "%f]\n", delay[arraySize - 1]);

		fprintf(result, "errorTime: [");
		fprintf(result, "%f]\n", errorTime);


		fprintf(result, "delayTime: [");
		fprintf(result, "%f]\n", delayTime);

		fprintf(result, sep);
		fclose(result);
		printf("Successfully Finished! Press any button to continue.\n");
		getchar();
		return 0;
	}
	else {
		printf("Error: file cannot be opened!");
		return 1;
	}
}

void eachError(double *B, double *error) {
	double bucketB[arraySize] = { 0 };	
	//long counter = 0;
#pragma omp parallel for schedule(dynamic) shared(bucketB)
	for (long run = 0; run < runTime; run++) {
		//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		//printf("*%d\n", seed); getchar();
		std::default_random_engine generator(run);
		std::normal_distribution<double> distribution(0, 1);
		double U[p];
		double C[p];
		int len = sizeof(double)*p;
		bool flag;
		int i;
		long t;
		double x[p], y;
		flag = false;
		int pbucket = 0;
		memset(U, 0, len);
		memset(C, 0, len);
		pbucket = 0;
		for (t = 0; t < T; t++) {
			y = distribution(generator);
			for (i = 0; i < p; i++) {
				x[i] = distribution(generator);
			}
			for (i = 0; i < p; i++) {
				U[i] += x[i] * y;
				C[i] = U[i] / pow(t + 1, (1 + epsilon) / 2);
				while (C[i] > B[pbucket]) {
#pragma omp atomic
					bucketB[pbucket] += 1;
					//#pragma omp flush(bucketB[pbucket])
					//printf("run: %d		bucketB: %f\n", run, bucketB[0]); //getchar();
					pbucket++;
					if (pbucket >= arraySize) {
						flag = true;
						break;
					}
				}
				if (flag) {
					break;
				}
			}
			if (flag) {
				break;
			}
		}
		if (run % 100 == 0) {
			printf("Processing(error): %d\n", run);
			//printf("counter: %d		bucketB: %f\n", counter, bucketB[0]);
		}
	}
	//printf("%d\n", bucketB[0]);
	for (int i = 0; i < arraySize; i++) {
		error[i] = bucketB[i] / runTime;
		//printf("error[%d]: %f\n", i, error[i]);
	}
	return;
}

void eachDelay(double *B, double *delay) {	
	double bucketB[arraySize] = { 0 };
#pragma omp parallel for schedule(dynamic) shared(bucketB)
	for (long run = 0; run < runTime; run++) {
		double U[p];
		double C[p];
		int len = sizeof(double)*p;
		bool flag;
		int i;
		long t;
		double x[p], y;
		double alpha[p] = { 0 };
		for (int i = 0; i < 2 * s; i++) {
			alpha[i] = 0.5;
		}	
		int pbucket = 0;
		std::default_random_engine generator(run);
		std::normal_distribution<double> distribution(0, 1);
		flag = false;
		memset(U, 0, len);
		memset(C, 0, len);
		pbucket = 0;
		for (t = 0; t < T; t++) {
			y = distribution(generator);
			for (i = 0; i < p; i++) {
				x[i] = distribution(generator);
			}//generate x
			for (i = 0; i < p; i++) {
				y += alpha[i] * x[i];
			}//generate y
			for (i = 0; i < p; i++) {
				U[i] += x[i] * y;
				C[i] = U[i] / pow(t + 1, (1 + epsilon) / 2);
				while (C[i] > B[pbucket]) {
#pragma omp atomic
					bucketB[pbucket] += (t + 1);
					pbucket++;
					if (pbucket >= arraySize) {
						flag = true;
						break;
					}
				}
				if (flag) {
					break;
				}
			}
			if (flag) {
				break;
			}
		}
		if (!flag) {
			for (i = pbucket; i < arraySize; i++) {
				bucketB[i] += T;
			}
		}
		if (run % 100 == 0) {
			printf("Processing(delay): %d\n", run);
		}
	}
	for (int i = 0; i < arraySize; i++) {
		delay[i] = bucketB[i] / runTime;
		//printf("delay[%d]: %f\n", i, delay[i]);
	}
	return;

}