#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

void readPnm(char * fileName, 
		int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

__global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height, 
		uint8_t * outPixels)
{
	// TODO
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	if (ix < width && iy <height)
	{
		int i = iy* width +ix;
		outPixels[i] = inPixels[3*i]*0.299 + inPixels[3*i +1]*0.587 + inPixels[3*i +2]*0.114;
	}
}

void convertRgb2Gray(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels, 
		bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
		for (int r = 0; r < height; r++)
		{
			for (int c = 0; c < width; c++)
			{
				int i = r * width + c;
				uint8_t r = inPixels[i * 3];
				uint8_t g = inPixels[i * 3 + 1];
				uint8_t b = inPixels[i * 3 + 2];
				outPixels[i] = 0.299f * r + 0.587f * g + 0.114f * b;
			}
		}
	}
	else // use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// TODO: Allocate device memories
		uint8_t *d_inpixels, *d_outpixels;
		CHECK(cudaMalloc(&d_inpixels, width * height * 3 * sizeof(uint8_t)));
		CHECK(cudaMalloc(&d_outpixels, width * height * sizeof(uint8_t)));


		// TODO: Copy data to device memories
		
		CHECK(cudaMemcpy(d_inpixels,inPixels,width * height*3 * sizeof(uint8_t),cudaMemcpyHostToDevice));

		// TODO: Set grid size and call kernel (remember to check kernel error)
		dim3 gridSize((width-1)/blockSize.x + 1,(height-1)/blockSize.y + 1);

		convertRgb2GrayKernel<<<gridSize,blockSize>>>(d_inpixels,width,height,d_outpixels);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		
		// TODO: Copy result from device memories
		CHECK(cudaMemcpy(outPixels,d_outpixels, width*height*sizeof(uint8_t),cudaMemcpyDeviceToHost));

		

		// TODO: Free device memories
		cudaFree(d_outpixels);
		cudaFree(d_inpixels);

		// for (int i = 0; i < width*height*3; i++)
		// 	printf("%d\t",d_inpixels[i]);

	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}

float computeError(uint8_t * a1, uint8_t * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
		err += abs((int)a1[i] - (int)a2[i]);
	err /= n;
	
	return err;
}

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

int main(int argc, char ** argv)
{	
	if (argc != 3 && argc != 5)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input RGB image file
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Convert RGB to grayscale not using device
	uint8_t * correctOutPixels= (uint8_t *)malloc(width * height);
	convertRgb2Gray(inPixels, width, height, correctOutPixels);

	// Convert RGB to grayscale using device
	uint8_t * outPixels= (uint8_t *)malloc(width * height);
	dim3 blockSize(32, 32); // Default
	if (argc == 5)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	} 
	convertRgb2Gray(inPixels, width, height, outPixels, true, blockSize); 

	// Compute mean absolute error between host result and device result
	float err = computeError(outPixels, correctOutPixels, width * height);
	printf("Error between device result and host result: %f\n", err);

	// Write results to files
	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(correctOutPixels, 1, width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(outPixels, 1, width, height, concatStr(outFileNameBase, "_device.pnm"));

	// Free memories
	free(inPixels);
	free(outPixels);
}
