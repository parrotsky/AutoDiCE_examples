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
static int load_labels(std::string path, std::vector<std::string>& labels)
{       
    FILE* fp = fopen(path.c_str(), "r");
    while (!feof(fp))
    {       
        char str[1024];
        fgets(str, 1024, fp); 
        std::string str_s(str);
        if (str_s.length() > 0)
        {   
            for (int i = 0; i < str_s.length(); i++)
            {   
                if (str_s[i] == ' ')
                {   
                    std::string strr = str_s.substr(i, str_s.length() - i - 1);
                    labels.push_back(strr);
                    i = str_s.length();
                }
            }
        }   
    }       
    return 0;
}  


//static int print_topk(const std::vector<float>& cls_scores, int topk)
static int print_topk(const std::vector<float>& cls_scores, int topk, std::vector<int>& index_result,
             std::vector<float>& score_result)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int>> vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());


    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
        index_result.push_back(index);
        score_result.push_back(score);
    }
    return 0;
}

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


void model0_engine(const char* imagepath, ncnn::Mat& data_0, ncnn::Mat& fc6_2){
   MPI_Request requests[1];
   MPI_Status status[1];

   int irank; MPI_Comm_rank(MPI_COMM_WORLD, &irank);
    read_input(imagepath, data_0);
     ncnn::Net lenovo_cpu0_model_0;
    lenovo_cpu0_model_0.opt.blob_allocator = &g_blob_pool_allocator;
    lenovo_cpu0_model_0.opt.workspace_allocator = &g_workspace_pool_allocator;
    lenovo_cpu0_model_0.opt.use_vulkan_compute = false;
    lenovo_cpu0_model_0.opt.use_winograd_convolution = true;
    lenovo_cpu0_model_0.opt.use_sgemm_convolution = true;
    lenovo_cpu0_model_0.opt.use_int8_inference = true;
    lenovo_cpu0_model_0.opt.use_fp16_packed = true;
    lenovo_cpu0_model_0.opt.use_fp16_storage = true;
    lenovo_cpu0_model_0.opt.use_fp16_arithmetic = true;
    lenovo_cpu0_model_0.opt.use_int8_storage = true;
    lenovo_cpu0_model_0.opt.use_int8_arithmetic = true;
    lenovo_cpu0_model_0.opt.use_packing_layout = true;
    lenovo_cpu0_model_0.opt.use_shader_pack8 = false;
    lenovo_cpu0_model_0.opt.use_image_storage = false;
    lenovo_cpu0_model_0.opt.num_threads = 1;
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(1);
    lenovo_cpu0_model_0.load_param("lenovo_cpu0.param");
    lenovo_cpu0_model_0.load_model("lenovo_cpu0.bin");

    ncnn::Extractor ex0 = lenovo_cpu0_model_0.create_extractor();

        ex0 = lenovo_cpu0_model_0.create_extractor();

        ex0.input("data_0", data_0);

       ex0.extract("fc6_2", fc6_2);
        MPI_Isend((float* )fc6_2, fc6_2.total(), MPI_FLOAT, 1, 
                0, MPI_COMM_WORLD, &requests[0]);

            MPI_Wait(&requests[0], &status[0]);
}


void model1_engine(int index, ncnn::Mat& fc6_2, ncnn::Mat& prob_1){
   MPI_Request requests[1];
   MPI_Status status[1];

   int irank; MPI_Comm_rank(MPI_COMM_WORLD, &irank);
     ncnn::Net lenovo_cpu1_model_1;
    lenovo_cpu1_model_1.opt.blob_allocator = &g_blob_pool_allocator;
    lenovo_cpu1_model_1.opt.workspace_allocator = &g_workspace_pool_allocator;
    lenovo_cpu1_model_1.opt.use_vulkan_compute = false;
    lenovo_cpu1_model_1.opt.use_winograd_convolution = true;
    lenovo_cpu1_model_1.opt.use_sgemm_convolution = true;
    lenovo_cpu1_model_1.opt.use_int8_inference = true;
    lenovo_cpu1_model_1.opt.use_fp16_packed = true;
    lenovo_cpu1_model_1.opt.use_fp16_storage = true;
    lenovo_cpu1_model_1.opt.use_fp16_arithmetic = true;
    lenovo_cpu1_model_1.opt.use_int8_storage = true;
    lenovo_cpu1_model_1.opt.use_int8_arithmetic = true;
    lenovo_cpu1_model_1.opt.use_packing_layout = true;
    lenovo_cpu1_model_1.opt.use_shader_pack8 = false;
    lenovo_cpu1_model_1.opt.use_image_storage = false;
    lenovo_cpu1_model_1.opt.num_threads = 1;
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(1);
    lenovo_cpu1_model_1.load_param("lenovo_cpu1.param");
    lenovo_cpu1_model_1.load_model("lenovo_cpu1.bin");

    ncnn::Extractor ex1 = lenovo_cpu1_model_1.create_extractor();

        MPI_Irecv((float* )fc6_2, fc6_2.total(), MPI_FLOAT, 0, 
                    0, MPI_COMM_WORLD, &requests[0]);

        ex1 = lenovo_cpu1_model_1.create_extractor();

        MPI_Wait(&requests[0], &status[0]);
        ex1.input("fc6_2", fc6_2);

       ex1.extract("prob_1", prob_1);
}

static int multi_classify(const char* imagepath, std::vector<float>& cls_scores)
{
   int irank; MPI_Comm_rank(MPI_COMM_WORLD, &irank);
  if (irank == 0) {
       ncnn::Mat data_0;
       ncnn::Mat fc6_2(4096);
 
      model0_engine(imagepath, data_0, fc6_2);
   }
  if (irank == 1) {
       ncnn::Mat fc6_2(4096);
       ncnn::Mat prob_1;
 
      model1_engine(0, fc6_2, prob_1);
    cls_scores.resize(prob_1.w);

    for (int j = 0; j < prob_1.w; j++)
    {
        cls_scores[j] = prob_1[j];
    }

std::vector<std::string> labels; 

load_labels("synset_words.txt", labels);

std::vector<int> index;

std::vector<float> score;

print_topk(cls_scores, 3, index, score);

   for (int i = 0; i < index.size(); i++)

   {

       fprintf(stderr, "%s \n", labels[index[i]].c_str());

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
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

   const char* imagepath = argv[1];
   g_blob_pool_allocator.set_size_compare_ratio(0.0f);
   g_workspace_pool_allocator.set_size_compare_ratio(0.5f);
   std::vector<float> cls_scores;
   multi_classify(imagepath, cls_scores);
   // Finalize the MPI environment.
   MPI_Finalize();
   return 0;
}
