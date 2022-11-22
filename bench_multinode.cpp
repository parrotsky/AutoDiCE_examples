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
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in Kb.
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = ' ';
    i = atoi(p);
    return i;
}

int getVirtual(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

int getPhysical(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}
static int multi_classify(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
int irank = MPI::COMM_WORLD.Get_rank();
int num_threads = ncnn::get_cpu_count();
std::string memory_pre = std::to_string(irank) + "_MEMORY.txt";
std::string perf_pre = std::to_string(irank) + "_PERFORMANCE.txt";
const char* memory = memory_pre.c_str();
const char* perf = perf_pre.c_str();
FILE* pm = fopen(memory, "wb");
FILE* pp = fopen(perf, "wb");
MPI_Request requests[2];
MPI_Status status[2];

if(irank==0){
    ncnn::Mat conv5_1(12, 12, 256);
    ncnn::Net nx01conv5_1;
    nx01conv5_1.opt.blob_allocator = &g_blob_pool_allocator;
    nx01conv5_1.opt.workspace_allocator = &g_workspace_pool_allocator;
    nx01conv5_1.opt.use_vulkan_compute = false;
    nx01conv5_1.opt.use_winograd_convolution = true;
    nx01conv5_1.opt.use_sgemm_convolution = true;
    nx01conv5_1.opt.use_int8_inference = true;
    nx01conv5_1.opt.use_fp16_packed = true;
    nx01conv5_1.opt.use_fp16_storage = true;
    nx01conv5_1.opt.use_fp16_arithmetic = true;
    nx01conv5_1.opt.use_int8_storage = true;
    nx01conv5_1.opt.use_int8_arithmetic = true;
    nx01conv5_1.opt.use_packing_layout = true;
    nx01conv5_1.opt.use_shader_pack8 = false;
    nx01conv5_1.opt.use_image_storage = false;
    nx01conv5_1.opt.num_threads = num_threads;
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);
    nx01conv5_1.load_param("nx01conv5_1.param");
    nx01conv5_1.load_model("nx01conv5_1.bin");

    ncnn::Extractor exconv5_1 = nx01conv5_1.create_extractor();

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);
    int g_warmup_loop_count = 2;
    int g_loop_count = 50;
// warm up
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        exconv5_1 = nx01conv5_1.create_extractor();

        exconv5_1.input("data_0", in);

        exconv5_1.extract("conv5_1", conv5_1);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        {
            exconv5_1 = nx01conv5_1.create_extractor();

            exconv5_1.input("data_0", in);

            exconv5_1.extract("conv5_1", conv5_1);
            MPI_Isend((float* )conv5_1, conv5_1.total(), MPI_FLOAT, 1, 
                0, MPI_COMM_WORLD, &requests[0]);

            MPI_Wait(&requests[0], &status[0]);
        }
        double end = ncnn::get_current_time();
        double time = end - start;
        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }
    time_avg /= g_loop_count;
    fprintf(stderr, "IRank: %d  min = %7.2f  max = %7.2f  avg = %7.2f\n", irank, time_min, time_max, time_avg);
    fprintf(pp, "%.2f", time_avg); // Unit: ms
    fprintf(pm, "%.2f", getPhysical()*1.0/1024); // Unit:MBytes
    std::cout <<"IRank: "<< irank << ", Physical Memory Usage (KB): "<<getPhysical()<< std::endl;
 }

if(irank==1){
    ncnn::Mat conv5_1(12, 12, 256);
    ncnn::Mat prob_1;
    ncnn::Net nx02prob_1;
    nx02prob_1.opt.blob_allocator = &g_blob_pool_allocator;
    nx02prob_1.opt.workspace_allocator = &g_workspace_pool_allocator;
    nx02prob_1.opt.use_vulkan_compute = false;
    nx02prob_1.opt.use_winograd_convolution = true;
    nx02prob_1.opt.use_sgemm_convolution = true;
    nx02prob_1.opt.use_int8_inference = true;
    nx02prob_1.opt.use_fp16_packed = true;
    nx02prob_1.opt.use_fp16_storage = true;
    nx02prob_1.opt.use_fp16_arithmetic = true;
    nx02prob_1.opt.use_int8_storage = true;
    nx02prob_1.opt.use_int8_arithmetic = true;
    nx02prob_1.opt.use_packing_layout = true;
    nx02prob_1.opt.use_shader_pack8 = false;
    nx02prob_1.opt.use_image_storage = false;
    nx02prob_1.opt.num_threads = num_threads;
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);
    nx02prob_1.load_param("nx02prob_1.param");
    nx02prob_1.load_model("nx02prob_1.bin");

    ncnn::Extractor exprob_1 = nx02prob_1.create_extractor();

    int g_warmup_loop_count = 2;
    int g_loop_count = 50;
// warm up
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        exprob_1 = nx02prob_1.create_extractor();

        exprob_1.input("conv5_1", conv5_1);
        exprob_1.extract("prob_1", prob_1);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        {
            MPI_Irecv((float* )conv5_1, conv5_1.total(), MPI_FLOAT, 0, 
                    0, MPI_COMM_WORLD, &requests[1]);

            exprob_1 = nx02prob_1.create_extractor();

            MPI_Wait(&requests[1], &status[1]);
            exprob_1.input("conv5_1", conv5_1);
            exprob_1.extract("prob_1", prob_1);
        }
        double end = ncnn::get_current_time();
        double time = end - start;
        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }
    time_avg /= g_loop_count;
    fprintf(stderr, "IRank: %d  min = %7.2f  max = %7.2f  avg = %7.2f\n", irank, time_min, time_max, time_avg);
    fprintf(pp, "%.2f", time_avg); // Unit: ms
    fprintf(pm, "%.2f", getPhysical()*1.0/1024); // Unit:MBytes
    std::cout <<"IRank: "<< irank << ", Physical Memory Usage (KB): "<<getPhysical()<< std::endl;
    cls_scores.resize(prob_1.w);

    for (int j = 0; j < prob_1.w; j++)
    {
        cls_scores[j] = prob_1[j];
    }

 }

fclose(pm);
fclose(pp);
return 0;

}

int main(int argc, char** argv)
{
    MPI::Init(argc, argv);

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

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    multi_classify(m, cls_scores);

    // Finalize the MPI environment.
    MPI::Finalize();

    return 0;
}
            
