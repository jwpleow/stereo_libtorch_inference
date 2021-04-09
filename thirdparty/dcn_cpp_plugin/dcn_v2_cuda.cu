#include "dcn_v2_im2col_cuda.h"
#include "dcn_v2.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <torch/script.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

//extern THCState *state;

//THCState *state;

// THCState *state = at::globalContext().thc_state;

THCState *state = at::globalContext().lazyInitCUDA();

//THCState *state = at::globalContext().getTHCState();

__global__ void createBatchGemmBuffer(const float **input_b, float **output_b,
                                      float **columns_b, const float **ones_b,
                                      const float **weight_b, const float **bias_b,
                                      float *input, float *output,
                                      float *columns, float *ones,
                                      float *weight, float *bias,
                                      const int input_stride, const int output_stride,
                                      const int columns_stride, const int ones_stride,
                                      const int num_batches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_batches)
    {
        input_b[idx] = input + idx * input_stride;
        output_b[idx] = output + idx * output_stride;
        columns_b[idx] = columns + idx * columns_stride;
        ones_b[idx] = ones + idx * ones_stride;
        // share weights and bias within a Mini-Batch
        weight_b[idx] = weight;
        bias_b[idx] = bias;
    }
}

at::Tensor
dcn_v2_cuda_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const at::Tensor &offset,
                    const at::Tensor &mask,
                    const int64_t kernel_h,
                    const int64_t kernel_w,
                    const int64_t stride_h,
                    const int64_t stride_w,
                    const int64_t pad_h,
                    const int64_t pad_w,
                    const int64_t dilation_h,
                    const int64_t dilation_w,
                    const int64_t deformable_group)
{
    using scalar_t = float;
    //THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_kernel,
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto ones = at::ones({batch, height_out, width_out}, input.options());
    auto columns = at::empty({batch, channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());
    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());

    int matrices_size = batch * sizeof(float *);

    auto input_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto output_b = static_cast<float **>(THCudaMalloc(state, matrices_size));
    auto columns_b = static_cast<float **>(THCudaMalloc(state, matrices_size));
    auto ones_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto weight_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto bias_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));

    const int block = 128;
    const int grid = (batch + block - 1) / block;

    createBatchGemmBuffer<<<grid, block, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        (const float**)input_b, output_b,
        columns_b, ones_b,
        weight_b, bias_b,
        input.data<scalar_t>(),
        output.data<scalar_t>(),
        columns.data<scalar_t>(),
        ones.data<scalar_t>(),
        weight.data<scalar_t>(),
        bias.data<scalar_t>(),
        channels * width * height,
        channels_out * width_out * height_out,
        channels * kernel_h * kernel_w * height_out * width_out,
        height_out * width_out,
        batch);

    long m_ = channels_out;
    long n_ = height_out * width_out;
    long k_ = 1;
    THCudaBlas_SgemmBatched(state,
                            't',
                            'n',
                            n_,
                            m_,
                            k_,
                            1.0f,
                            ones_b, k_,
                            bias_b, k_,
                            0.0f,
                            output_b, n_,
                            batch);

    modulated_deformable_im2col_cuda(at::cuda::getCurrentCUDAStream().stream(),
                                     input.data<scalar_t>(),
                                     offset.data<scalar_t>(),
                                     mask.data<scalar_t>(),
                                     batch, channels, height, width,
                                     height_out, width_out, kernel_h, kernel_w,
                                     pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                     deformable_group,
                                     columns.data<scalar_t>());

    long m = channels_out;
    long n = height_out * width_out;
    long k = channels * kernel_h * kernel_w;
    THCudaBlas_SgemmBatched(state,
                            'n',
                            'n',
                            n,
                            m,
                            k,
                            1.0f,
                            (const float **)columns_b, n,
                            weight_b, k,
                            1.0f,
                            output_b, n,
                            batch);

    THCudaFree(state, input_b);
    THCudaFree(state, output_b);
    THCudaFree(state, columns_b);
    THCudaFree(state, ones_b);
    THCudaFree(state, weight_b);
    THCudaFree(state, bias_b);
    return output;
}

