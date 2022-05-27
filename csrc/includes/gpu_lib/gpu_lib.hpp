namespace gpu_lib {
#if defined DS_HAS_CUDA
    using namespace cuda;
#elif defined DS_HAS_ROCM
    using namespace rocm;
#endif
