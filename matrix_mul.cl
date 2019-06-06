__kernel void matrix_mul(__global int *A, __global int *B, __global int *C) {
    
    // Get the index of the current element
    int size = get_global_size(0);
    int i = get_global_id(0);
    int j = get_global_id(1);

    int acc = 0;

    if (i < size && j < size) {
    	for (int k=0; k<size; k++)
    		//acc += A[j*size + k] * B[i*size + k];
	    	acc += A[j*size + k] * B[k*size + i];

	    C[j*size + i] = acc;
	}
}

__kernel void matrix_mul_tile(__global int *A, __global int *B, __global int *C) {
    int size = get_global_size(0);
	const int l_i = get_local_id(0); // Local col ID (max: TILE_SIZE)
    const int l_j = get_local_id(1); // Local row ID (max: TILE_SIZE)
    const int g_i = get_global_id(0); // Global col ID of C (0..SIZE)
    const int g_j = get_global_id(1); // Global row ID of C (0..SIZE)
 
    // Local memory to fit a tile of TILE_SIZE*TILE_SIZE elements of A and B
    __local int Asub[TILE_SIZE][TILE_SIZE];
    __local int Bsub[TILE_SIZE][TILE_SIZE];
 
    // Initialise the accumulation register
    int acc = 0;
    int t_i;
    int t_j;
    // Loop over all tiles
    const int numTiles = size/TILE_SIZE;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        t_i = t*TILE_SIZE + l_i;
        t_j = t*TILE_SIZE + l_j;

        Asub[l_i][l_j] = A[g_j*size + t_i];
        Bsub[l_i][l_j] = B[t_j*size + g_i];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TILE_SIZE; k++) {
            acc += Asub[k][l_j] * Bsub[l_i][k];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[g_j*size + g_i] = acc;
   
}
