CL_PROGRAM R"(

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xwhere(const int n, const int rank, __constant int* shape,
            const __global bool* restrict cgm, const int c_offset,
            const __global real* restrict xgm, const int x_offset,
            const __global real* restrict ygm, const int y_offset,
            __global real* zgm, const int z_offset)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    int c_id = c_offset, x_id = x_offset, y_id = y_offset;
    unravel3(id, &c_id, &x_id, &y_id, rank, shape);
    zgm[id + z_offset] = cgm[c_id] ? xgm[x_id] : ygm[y_id];
  }
}

)"
