// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Work-group size parameters re-used from the 'copy' kernel
#ifndef COPY_DIMX
  #define COPY_DIMX 8      // Local workgroup size in the first dimension (w)
#endif
#ifndef COPY_DIMY
  #define COPY_DIMY 8      // Local workgroup size in the second dimension (h)
#endif

__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void Xmaxpool(const int channels, const int input_h, const int input_w,
              const int output_h, const int output_w,
              const int kernel_h, const int kernel_w,
              const int pad_h, const int pad_w,
              const int stride_h, const int stride_w,
              const int dilation_h, const int dilation_w,
              const __global real* restrict x_buffer, const int x_offset,
              __global real* y_buffer, const int y_offset)
{
  const int b_id = get_global_id(2); // batch
  const int h_id = get_global_id(0); // image width, max 'output_w'
  const int w_id = ((int)get_global_id(1)) % output_h; // image height, max 'output_h'
  const int c_id = ((int)get_global_id(1)) / output_h; // input channels

  if (h_id < output_h && w_id < output_w && c_id < channels) {
    real val = SMALLEST;
    for (int kh_id = 0; kh_id < kernel_h; ++kh_id) {
      const int h_index = kh_id * dilation_h + stride_h * h_id - pad_h;
      if (h_index >= 0 && h_index < input_h) {
        for (int kw_id = 0; kw_id < kernel_w; ++kw_id) {
          const int w_index = kw_id * dilation_w + stride_w * w_id - pad_w;
          if (w_index >= 0 && w_index < input_w) {
            const int input_index = ((b_id*channels + c_id)*input_h + h_index)*input_w + w_index;
            val = maxval(val, x_buffer[input_index + x_offset]);
          }
        }
      }
    }

    const int output_index = ((b_id*channels + c_id)*output_h + h_id)*output_w + w_id;
    y_buffer[output_index + y_offset] = val;
  }
}

__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void Xavgpool(const int channels, const int input_h, const int input_w,
              const int output_h, const int output_w,
              const int kernel_h, const int kernel_w,
              const int pad_h, const int pad_w,
              const int stride_h, const int stride_w,
              const int dilation_h, const int dilation_w,
              const int count_include_pad,
              const __global real* restrict x_buffer, const int x_offset,
              __global real* y_buffer, const int y_offset)
{
  const int b_id = get_global_id(2); // batch
  const int h_id = get_global_id(0); // image width, max 'output_w'
  const int w_id = ((int)get_global_id(1)) % output_h; // image height, max 'output_h'
  const int c_id = ((int)get_global_id(1)) / output_h; // input channels

  if (h_id < output_h && w_id < output_w && c_id < channels) {
    real sum = ZERO;
    int count = 0;

    for (int kh_id = 0; kh_id < kernel_h; ++kh_id) {
      const int h_index = kh_id * dilation_h + stride_h * h_id - pad_h;
      if (h_index >= 0 && h_index < input_h) {
        for (int kw_id = 0; kw_id < kernel_w; ++kw_id) {
          const int w_index = kw_id * dilation_w + stride_w * w_id - pad_w;
          if (w_index >= 0 && w_index < input_w) {
            const int input_index = ((b_id*channels + c_id)*input_h + h_index)*input_w + w_index;
            sum += x_buffer[input_index + x_offset];
            count++;
          }
        }
      }
    }

    real val = count_include_pad ? sum/(kernel_h*kernel_w) : sum/count;
    const int output_index = ((b_id*channels + c_id)*output_h + h_id)*output_w + w_id;
    y_buffer[output_index + y_offset] = val;
  }
}

__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void Xlppool(const int channels, const int input_h, const int input_w,
             const int output_h, const int output_w,
             const int kernel_h, const int kernel_w,
             const int pad_h, const int pad_w,
             const int stride_h, const int stride_w,
             const int dilation_h, const int dilation_w,
             const int p,
             const __global real* restrict x_buffer, const int x_offset,
             __global real* y_buffer, const int y_offset)
{
  const int b_id = get_global_id(2); // batch
  const int h_id = get_global_id(0); // image width, max 'output_w'
  const int w_id = ((int)get_global_id(1)) % output_h; // image height, max 'output_h'
  const int c_id = ((int)get_global_id(1)) / output_h; // input channels

  if (h_id < output_h && w_id < output_w && c_id < channels) {
    real val = ZERO;
    for (int kh_id = 0; kh_id < kernel_h; ++kh_id) {
      const int h_index = kh_id * dilation_h + stride_h * h_id - pad_h;
      if (h_index >= 0 && h_index < input_h) {
        for (int kw_id = 0; kw_id < kernel_w; ++kw_id) {
          const int w_index = kw_id * dilation_w + stride_w * w_id - pad_w;
          if (w_index >= 0 && w_index < input_w) {
            const int input_index = ((b_id*channels + c_id)*input_h + h_index)*input_w + w_index;
            real x = x_buffer[input_index + x_offset];
            val += pow(fabs(x), p);
          }
        }
      }
    }

    const int output_index = ((b_id*channels + c_id)*output_h + h_id)*output_w + w_id;
    y_buffer[output_index + y_offset] = pow(val, ONE/p);
  }
}

)" // End of the C++11 raw string literal
