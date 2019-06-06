#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <string.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)
#define SIZE_1 512 // column/row size.
#define SIZE_2 1024 // column/row size.
#define SIZE_3 2048 // column/row size.

#define TILE_SIZE_1 8 // column/row size in a tile. It can only be 1,2,4,8,16 because of the limitation from the GPU
#define TILE_SIZE_2 16 // column/row size in a tile. It can only be 1,2,4,8,16 because of the limitation from the GPU

cl_uint ret_num_platforms = 0;

cl_uint getPlatformIndex(cl_platform_id* platforms, bool platform_type) {

    char* required_platform_subname = (char*) malloc(5);
    cl_uint selected_platform_index = 3; //Start at max
    if(platform_type) {
        strcpy(required_platform_subname, "CPU");
    } else {
        strcpy(required_platform_subname, "Graphics"); //Names as per CapsBasic
    }
    std::cout << "Reqd name = " << required_platform_subname << std::endl;
    for(cl_uint i = 0; i < ret_num_platforms; ++i)
    {
        // Get the length for the i-th platform name
        size_t platform_name_length = 0;
        clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_NAME,
            0,
            0,
            &platform_name_length
            );

        // Get the name itself for the i-th platform
        char* platform_name = new char[platform_name_length];
        clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_NAME,
            platform_name_length,
            platform_name,
            0
            );

        // decide if this i-th platform is what we are looking for
        // we select the first one matched skipping the next one if any
        if(
            strstr(platform_name, required_platform_subname) 
//&&            selected_platform_index == num_of_platforms // have not selected yet
            )
        {
            std::cout << " [Selected] " << i << std::endl;
            selected_platform_index = i;
            delete [] platform_name;
            return selected_platform_index;
            // return the first match
        }

//        cout << endl;
//        delete [] platform_name;
    }
    return -1;
}

// C version of matrix multiplcation. Use this function for result validation and execution time comaprison
void matrix_mul_sequence (int *A_mat,
                          int *B_mat,
                          int *C_mat,
			  size_t SIZE)
{
	for (size_t j=0; j<SIZE; j++) {
		for (size_t i=0; i<SIZE; i++)
			for (size_t k=0; k<SIZE; k++)
				C_mat[j*SIZE + i] += A_mat[j*SIZE + k] * B_mat[k*SIZE + i];
	}
}


int runForSize(size_t SIZE, size_t TILE_SIZE, bool platform_select) {

    cl_device_type platformType;

    if(platform_select) {
        platformType = CL_DEVICE_TYPE_CPU;
    } else {
        platformType = CL_DEVICE_TYPE_GPU;
    }


    std::cout << "Platform " << platform_select << " Matrix size " << SIZE << "x" << SIZE << " Tile size " << TILE_SIZE << std::endl;
    
    // A, B are input matrix, C is the output matrix for OpenCL, C_seq is the output matrix for reference implementation.
    int *A = new int[SIZE*SIZE];
    int *B = new int[SIZE*SIZE];
    int *C = new int[SIZE*SIZE];
    int *C_seq = new int[SIZE*SIZE];

    //Initialize matrix
    for(size_t j=0; j<SIZE; j++) {
		for(size_t i=0; i<SIZE; i++) {
			A[j*SIZE + i] = 1;
			B[j*SIZE + i] = i+1;
			C[j*SIZE + i] = 0;
			C_seq[j*SIZE + i] = 0;
		}
    }

	std::chrono::high_resolution_clock::time_point t1, t2;
	t1 = std::chrono::high_resolution_clock::now();
    matrix_mul_sequence(A, B, C_seq, SIZE);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Reference C matrix multiplication: "
		<< (float)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000
		<< " sec"
		<< std::endl;

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("matrix_mul.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp); //Why is this required?
    fclose( fp );


    //Init variables
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;

    //Get number of platforms
    cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);


    cl_platform_id* platform_id = new cl_platform_id[ret_num_platforms]; //List of platforms

    std::cout << "clGetPlatformIDs " << ret_num_platforms << std::endl;

    // Get platform and device information
    ret = clGetPlatformIDs(ret_num_platforms, platform_id, 0); //Returns the list of platforms found. Minimum of arg1 and arg3.

    std::cout << "clGetPlatformIDs List Ret = " << ret << std::endl;

    cl_uint selected_platform_index = getPlatformIndex(platform_id, platform_select);

    std::cout << "getPlatformIndex " << selected_platform_index << std::endl;
	

    cl_platform_id platformCPU = platform_id[selected_platform_index];

ret = clGetDeviceIDs(platformCPU, platformType, 1, &device_id, &ret_num_devices); //Returns the devices found
	std::cout << "clGetDeviceIDs " << ret << std::endl;
    // Create an OpenCL context
//An OpenCL context is created with one or more devices. Contexts are used by the OpenCL runtime for managing objects such as command-queues, memory, program and kernel objects and for executing kernels on one or more devices specified in the context.
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue with the capability of performance profiling for target device
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    // Create memory buffers on the device for each matrix
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE*SIZE*sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE*SIZE*sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE*SIZE*sizeof(int), NULL, &ret);

    // Copy the matrix A, B and C to each device memory counterpart
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, SIZE*SIZE*sizeof(int), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, SIZE*SIZE*sizeof(int), B, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, c_mem_obj, CL_TRUE, 0, SIZE*SIZE*sizeof(int), C, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build and compile the OpenCL kernel program
    std::string build_option = "-DTILE_SIZE=" + std::to_string(TILE_SIZE);
    ret = clBuildProgram(program, 1, &device_id, build_option.c_str(), NULL, NULL);
    if (ret == CL_BUILD_PROGRAM_FAILURE) { // If compile failed, print the error message
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *log = (char *) malloc(log_size);

		// Get the log and print it
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
	}

    // Create the OpenCL kernel
    cl_kernel kernel;
	kernel = clCreateKernel(program, "matrix_mul", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

    int dimention = 2; // In this example, We will use 2 dimention index
    size_t global_item_size[] = {SIZE, SIZE, 1};
    size_t local_item_size[] = {TILE_SIZE, TILE_SIZE, 1};

	cl_event perf_event;
	cl_ulong start, end;

	// Execute the OpenCL kernel
    ret = clEnqueueNDRangeKernel(command_queue, kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, &perf_event);
    // Capture performance event from target device. In this case the event is to retrive the execution time.
    ret = clWaitForEvents(1, &perf_event);
    ret = clGetEventProfilingInfo(perf_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    ret = clGetEventProfilingInfo(perf_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
	std::cout << "OpenCL matrix multiplication: " << (float)(end - start)/1000000000 << " sec" << std::endl;

    // Read the memory buffer C from the device into the local variable C
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, SIZE*SIZE*sizeof(int), C, 0, NULL, NULL);

	// Make sure all the command in the command queue has been executed
    ret = clFinish(command_queue);

    /**
    * Tiled kernel
    */
	
    kernel = clCreateKernel(program, "matrix_mul_tile", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

	// Execute the OpenCL kernel
    ret = clEnqueueNDRangeKernel(command_queue, kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, &perf_event);
    // Capture performance event from target device. In this case the event is to retrive the execution time.
    ret = clWaitForEvents(1, &perf_event);
    ret = clGetEventProfilingInfo(perf_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    ret = clGetEventProfilingInfo(perf_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
	std::cout << "OpenCL matrix tiled: " << (float)(end - start)/1000000000 << " sec" << std::endl;

    // Read the memory buffer C from the device into the local variable C
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, SIZE*SIZE*sizeof(int), C, 0, NULL, NULL);

	// Make sure all the command in the command queue has been executed
    ret = clFinish(command_queue);

   
    bool validate = true;
    for(size_t j=0; j<SIZE; j++) {
		for(size_t i=0; i<SIZE; i++) {
			if (C[j*SIZE + i] != C_seq[j*SIZE + i])
				validate = false;
		}
	}

	if (validate == false)
		std::cout << "The results are mismatched !!" << std::endl;

    // Clean up
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

	std::cout << "Press Enter to finish..." << std::endl;
	getchar();
	return 0;
}

int main(void)
{
    /**
     * For GPU
    */
    bool isCPU = false; //Set to true for CPU selection

    //Two tile sizes and normal kernel for a size of the matrix

    runForSize(SIZE_1, TILE_SIZE_1, isCPU); runForSize(SIZE_1, TILE_SIZE_2, isCPU); //For 512x512

    runForSize(SIZE_2, TILE_SIZE_1, isCPU); runForSize(SIZE_2, TILE_SIZE_2, isCPU); //For 1024x1024

    runForSize(SIZE_3, TILE_SIZE_1, isCPU); runForSize(SIZE_3, TILE_SIZE_2, isCPU); //For 2048x2048


    /**
     * For CPU
    */
    isCPU = true;

    runForSize(SIZE_1, TILE_SIZE_1, isCPU); runForSize(SIZE_1, TILE_SIZE_2, isCPU); //Two tile sizes and normal kernel for a size of the matrix

    runForSize(SIZE_2, TILE_SIZE_1, isCPU); runForSize(SIZE_2, TILE_SIZE_2, isCPU);

    runForSize(SIZE_3, TILE_SIZE_1, isCPU); runForSize(SIZE_3, TILE_SIZE_2, isCPU);

    return 0;
}
