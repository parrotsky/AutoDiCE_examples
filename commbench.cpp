#include "net.h"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "benchmark.h"
#include "cpu.h"
#include "gpu.h"
#include <stdio.h>
#include <vector>
#include <mpi.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

int main(int argc, char** argv)
{
   int provided;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
   if (provided < MPI_THREAD_MULTIPLE) {
       fprintf(stderr, "xxx MPI does not provide needed thread support!\n");
       return -1;
   // Error - MPI does not provide needed threading level
   }
    // MPI::Init(argc, argv);
        // Check for at least one argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [int1] [int2] [int3]\n";
        return -1;
    }

    ncnn::Mat mat, output;
    int dim1, dim2, dim3;
    const int warmup_trials = 20;
    const int measured_trials = 100;
    double send_bandwidth = 0;
    double recv_bandwidth = 0;
    double total_bandwidth = 0;

    switch(argc) {
        case 2: // 1D
            std::stringstream(argv[1]) >> dim1;
            mat = ncnn::Mat(dim1);
            output = ncnn::Mat(dim1);
            break;
        case 3: // 2D
            std::stringstream(argv[1]) >> dim1;
            std::stringstream(argv[2]) >> dim2;
            mat = ncnn::Mat(dim1, dim2);
            output = ncnn::Mat(dim1, dim2);
            break;
        case 4: // 3D
            std::stringstream(argv[1]) >> dim1;
            std::stringstream(argv[2]) >> dim2;
            std::stringstream(argv[3]) >> dim3;
            mat = ncnn::Mat(dim1, dim2, dim3);
            output = ncnn::Mat(dim1, dim2, dim3);
            break;
        default:
            std::cerr << "Invalid number of dimensions. Please provide 1 to 3 dimensions.\n";
            return -1;
    }

    // Debug output
    // std::cout << "Mat constructed successfully." << mat.total() << std::endl;

    double data_in_MB = (sizeof(float) * mat.total()) / (1024.0 * 1024.0); // Amount of data transferred in MB

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Request requests[2];
    MPI_Status statuses[2];

    if (world_size < 2) {
        std::cerr << "World size must be greater than 1\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

        for (int t = 0; t < warmup_trials + measured_trials; ++t) {

    if (world_rank == 0) {
        // Process 0 sends the buffer to process 1
        MPI_Isend((float* )mat, mat.total(), MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &requests[0]);

        auto start = std::chrono::high_resolution_clock::now();

        // Wait for the send to complete
        MPI_Wait(&requests[0], &statuses[0]);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
      //  std::cout << "Send time: " << elapsed.count() << " s\n";

	  if (t >= warmup_trials) {
	      double bandwidth = data_in_MB / elapsed.count(); // in bytes per second
        send_bandwidth += bandwidth;
    }

    } else if (world_rank == 1) {
        // Process 1 receives the buffer from process 0
        MPI_Irecv((float* )mat, mat.total(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &requests[1]);

        auto start = std::chrono::high_resolution_clock::now();

        // Wait for the receive to complete
        MPI_Wait(&requests[1], &statuses[1]);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
    //    std::cout << "Receive time: " << elapsed.count() << " s\n";

	            // Skip the warmup trials
            if (t >= warmup_trials) {
		//double bandwidth = (sizeof(float) * mat.total()) / elapsed.count(); // in bytes per second
		//bandwidth /= (1024 * 1024); // convert to MB/s
	              double bandwidth = data_in_MB / elapsed.count(); // in bytes per second
                recv_bandwidth += bandwidth;
            }
    }
}

    // Calculate average bandwidth
if (world_rank == 0) {
    double average_bandwidth = send_bandwidth / measured_trials;
    double average_time = data_in_MB / average_bandwidth; // Average time in seconds
    std::cout << "Average send bandwidth: " << average_bandwidth << " MB/s"<< "; Average send time: " << average_time *1000.0 << " ms\n";
} else if (world_rank == 1) {
    double average_bandwidth = recv_bandwidth / measured_trials;
    double average_time = data_in_MB / average_bandwidth; // Average time in seconds
    std::cout << "Average recv bandwidth: " << average_bandwidth << " MB/s"<< "; Average recv time: " << average_time*1000.0 << " ms\n";
}


// ## if test benchmark for all reduce.
// ## enable the next part.

   total_bandwidth = 0;
   for (int t = 0; t < warmup_trials + measured_trials; ++t) {

        auto start = std::chrono::high_resolution_clock::now();

        // Perform the allreduce operation
        MPI_Allreduce((float* )mat, (float* )output, mat.total(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;

        // Skip the warmup trials
        if (t >= warmup_trials) {
            double bandwidth = data_in_MB / elapsed.count(); // in bytes per second
            total_bandwidth += bandwidth;
        }
    }

    // Calculate average bandwidth and average time
    double average_bandwidth = total_bandwidth / measured_trials;
    double average_time = data_in_MB / average_bandwidth; // Average time in seconds

    std::cout << "Average allreduce bandwidth: " << average_bandwidth << " MB/s" << "; Average allreduce time: " << average_time*1000.0 << " ms\n";


   g_blob_pool_allocator.set_size_compare_ratio(0.0f);
   g_workspace_pool_allocator.set_size_compare_ratio(0.5f);
   std::vector<float> cls_scores;
   // Finalize the MPI environment.
   MPI_Finalize();
   return 0;
}
