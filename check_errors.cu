#define wbCheck(stmt) do {\ 
    cudaError_t err = stmt;\ 
    if(err != cudaSuccess) { \
        wbLog(ERROR, "Failed to run stmt", #stmt); \
        wbLog(ERROR, "Got CUDA error ...", cudaGetErrorString(err)); \
        return -1; \
        } \
    } while(0)

// example usage
int check_error = wbCheck(cudaMalloc(...));