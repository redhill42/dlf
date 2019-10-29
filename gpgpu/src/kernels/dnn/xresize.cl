CL_PROGRAM R"(

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xresize1d(
    const __global real* restrict xgm, const int x_offset, const int x_len,
    __global real* ygm, const int y_offset, const int y_len)
{
    const int batch = get_global_id(1);
    const int gid   = get_global_id(0);

    const float x = (gid + 0.5f)*x_len/y_len - 0.5f;
    const int   i = (int)floor(x);
    const float a = x == (float)i ? 1.f : x - i;

    xgm += x_offset + batch*x_len;
    ygm += y_offset + batch*y_len;

    real q0 = xgm[max(0, i)];
    real q1 = xgm[min(i+1, x_len-1)];
    ygm[gid] = q0 + a*(q1 - q0);
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xresize1dStrided(const int rank,
    const __global real* restrict xgm, const int x_offset, __constant int* x_shape,
    __global real* ygm, const int y_offset, __constant int* y_shape)
{
    const int batch = get_global_id(1);
    const int gid   = get_global_id(0);

    const int x_len = x_shape[rank-1];
    const int y_len = y_shape[rank-1];
    const int x_inc = x_shape[2*rank-1];
    const int y_inc = y_shape[2*rank-1];

    const float x = (gid + 0.5f)*x_len/y_len - 0.5f;
    const int   i = (int)floor(x);
    const float a = x == (float)i ? 1.f : x - i;

    xgm += x_offset + unravel(batch*x_len, rank, x_shape);
    ygm += y_offset + unravel(batch*y_len, rank, y_shape);

    real q0 = xgm[max(0, i) * x_inc];
    real q1 = xgm[min(i+1, x_len-1) * x_inc];
    ygm[gid * y_inc] = q0 + a*(q1 - q0);
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xresize2d(
    const __global real* restrict xgm, const int x_offset, const int input_w, const int input_h,
    __global real* ygm, const int y_offset, const int output_w, const int output_h)
{
    const int batch = get_global_id(1);
    const int gid   = get_global_id(0);

    const int idy = gid / output_w;
    const int idx = gid - idy*output_w;

    const float x = (idx + 0.5f)*input_w/output_w - 0.5f;
    const int   i = (int)floor(x);
    const float a = x == (float)i ? 1.f : x - i;

    const float y = (idy + 0.5f)*input_h/output_h - 0.5f;
    const int   j = (int)floor(y);
    const float b = y == (float)j ? 1.f : y - j;

    xgm += x_offset + batch * input_w * input_h;
    ygm += y_offset + batch * output_w * output_h;

    const int i0 = max(0, i);
    const int i1 = min(i+1, input_w-1);
    const int j0 = max(0, j) * input_w;
    const int j1 = min(j+1, input_h-1) * input_w;

    real q00 = xgm[i0 + j0];
    real q10 = xgm[i1 + j0];
    real q01 = xgm[i0 + j1];
    real q11 = xgm[i1 + j1];

    real r0 = q00 + a*(q10 - q00);
    real r1 = q01 + a*(q11 - q01);
    ygm[idy*output_w + idx] = r0 + b*(r1 - r0);
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xresize2dStrided(const int rank,
    const __global real* restrict xgm, const int x_offset, __constant int* x_shape,
    __global real* ygm, const int y_offset, __constant int* y_shape)
{
    const int batch         = get_global_id(1);
    const int gid           = get_global_id(0);

    const int input_w       = x_shape[rank-1];
    const int input_h       = x_shape[rank-2];
    const int output_w      = y_shape[rank-1];
    const int output_h      = y_shape[rank-2];

    const int input_inc     = x_shape[2*rank-1];
    const int input_stride  = x_shape[2*rank-2];
    const int output_inc    = y_shape[2*rank-1];
    const int output_stride = y_shape[2*rank-2];

    const int idy = gid / output_w;
    const int idx = gid - idy*output_w;

    const float x = (idx + 0.5f)*input_w/output_w - 0.5f;
    const int   i = (int)floor(x);
    const float a = x == (float)i ? 1.f : x - i;

    const float y = (idy + 0.5f)*input_h/output_h - 0.5f;
    const int   j = (int)floor(y);
    const float b = y == (float)j ? 1.f : y - j;

    xgm += x_offset + unravel(batch*input_w*input_h, rank, x_shape);
    ygm += y_offset + unravel(batch*output_w*output_h, rank, y_shape);

    const int i0 = max(0, i) * input_inc;
    const int i1 = min(i+1, input_w-1) * input_inc;
    const int j0 = max(0, j) * input_stride;
    const int j1 = min(j+1, input_h-1) * input_stride;

    real q00 = xgm[i0 + j0];
    real q10 = xgm[i1 + j0];
    real q01 = xgm[i0 + j1];
    real q11 = xgm[i1 + j1];

    real r0 = q00 + a*(q10 - q00);
    real r1 = q01 + a*(q11 - q01);
    ygm[idy*output_stride + idx*output_inc] = r0 + b*(r1 - r0);
}

)"
