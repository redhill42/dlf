CL_PROGRAM R"(

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xrange(const int n, const real_arg start_arg, const real_arg delta_arg,
            __global real* xgm, const int x_offset)
{
    const real start = GetRealArg(start_arg);
    const real delta = GetRealArg(delta_arg);
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
        xgm[id + x_offset] = id*delta + start;
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XrangeStrided(
    const int n, const real_arg start_arg, const real_arg delta_arg,
    const int rank, __constant int* shape,
    __global real* xgm, const int x_offset)
{
    const real start = GetRealArg(start_arg);
    const real delta = GetRealArg(delta_arg);
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
        xgm[unravel(id, rank, shape) + x_offset] = id*delta + start;
    }
}

)" // End of the C++11 raw string literal
