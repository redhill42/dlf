#pragma once

#include <functional>
#include <unordered_set>
#include <complex>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstddef>
#include <cstdlib>
#include <cmath>

#include "utility.h"
#include "parallel.h"
#include "cxxblas.h"

#include "tensor/shape.h"
#include "tensor/xfn.h"
#include "tensor/host.h"
#include "tensor/device.h"
#include "tensor/traits.h"
#include "tensor/map.h"
#include "tensor/reorder.h"
#include "tensor/transform.h"
#include "tensor/reduce.h"
#include "tensor/pad.h"
#include "tensor/linalg.h"
#include "tensor/dnn.h"
#include "tensor/image.h"
