PROGRAM_STRING_DEBUG_INFO R"(

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xonehot(const int n, const int d, const int k,
             const __global real* restrict indices,
             const __global real* restrict values,
             __global real* output)
{
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
        int ls = id / k;
        int src_id = ls/d*k + id%k;
        int b = ls % d;

        int a = indices[src_id];
        if (a < 0) a += d;
        output[id] = (a == b) ? values[1] : values[0];
    }
}

)"
