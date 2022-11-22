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


static int read_input(const char* imagepath, ncnn::Mat& in)
{
   cv::Mat m = cv::imread(imagepath, 1);
   if (m.empty())
   {
       fprintf(stderr, "cv::imread %s failed\n", imagepath);
       return -1;
   }
   // m: BGR format.
   in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_BGR, m.cols, m.rows, 224, 224);
   const float mean_vals[3] = {104.f, 117.f, 123.f};
   in.substract_mean_normalize(mean_vals, 0);
   return 0;
}


void model0_engine(const char* imagepath, ncnn::Mat& data_0, ncnn::Mat& fc6_1){
   MPI_Request requests[0];
   MPI_Status status[0];

   int irank = MPI::COMM_WORLD.Get_rank();
    read_input(imagepath, data_0);
     ncnn::Net nx09_gpu_model_0;
    nx09_gpu_model_0.opt.blob_allocator = &g_blob_pool_allocator;
    nx09_gpu_model_0.opt.workspace_allocator = &g_workspace_pool_allocator;
    nx09_gpu_model_0.opt.use_vulkan_compute = true;
    nx09_gpu_model_0.opt.use_winograd_convolution = true;
    nx09_gpu_model_0.opt.use_sgemm_convolution = true;
    nx09_gpu_model_0.opt.use_int8_inference = true;
    nx09_gpu_model_0.opt.use_fp16_packed = true;
    nx09_gpu_model_0.opt.use_fp16_storage = true;
    nx09_gpu_model_0.opt.use_fp16_arithmetic = true;
    nx09_gpu_model_0.opt.use_int8_storage = true;
    nx09_gpu_model_0.opt.use_int8_arithmetic = true;
    nx09_gpu_model_0.opt.use_packing_layout = true;
    nx09_gpu_model_0.opt.use_shader_pack8 = false;
    nx09_gpu_model_0.opt.use_image_storage = false;
    nx09_gpu_model_0.opt.num_threads = 1;
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(1);
    nx09_gpu_model_0.load_param("nx09_gpu.param");
    nx09_gpu_model_0.load_model("nx09_gpu.bin");

    ncnn::Extractor ex0 = nx09_gpu_model_0.create_extractor();

        ex0 = nx09_gpu_model_0.create_extractor();

        ex0.input("data_0", data_0);

       ex0.extract("fc6_1", fc6_1);
}

static int multi_classify(const char* imagepath, std::vector<float>& cls_scores)
{
   int irank = MPI::COMM_WORLD.Get_rank();
  if (irank == 0) {
       ncnn::Mat data_0;
       ncnn::Mat fc6_1;
 
      model0_engine(imagepath, data_0, fc6_1);
    cls_scores.resize(fc6_1.w);

    for (int j = 0; j < fc6_1.w; j++)
    {
        cls_scores[j] = fc6_1[j];
    }

   }
return 0;
}

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

    // Get the number of processes
    int world_size;
    world_size = MPI::COMM_WORLD.Get_size();

    // Get the rank of the process
    int world_rank;
    world_rank = MPI::COMM_WORLD.Get_rank();

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

   const char* imagepath = argv[1];
    if(world_rank==0) {
    ncnn::create_gpu_instance();
   }
   g_blob_pool_allocator.set_size_compare_ratio(0.0f);
   g_workspace_pool_allocator.set_size_compare_ratio(0.5f);
   std::vector<float> cls_scores;
   multi_classify(imagepath, cls_scores);
    if(world_rank==0) {
    ncnn::destroy_gpu_instance();
   }
   // Finalize the MPI environment.
   MPI::Finalize();
   return 0;
}
