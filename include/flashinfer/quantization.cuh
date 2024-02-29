/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_QUANTIZATION_CUH_
#define FLASHINFER_QUANTIZATION_CUH_

#include "vec_dtypes.cuh"

namespace flashinfer {

enum class QuantizationMode {
  k8BitSym = 0U,
  k8BitAsym = 1U,
  k4BitSym = 2U,
  k4BitAsym = 3U,
};

template <typename T>
__device__ vec_t<T, > group_dequant

template <QuantizationMode qmode, typename T>
__global__ QuantizeKernel() {

}

template <QuantizationMode qmode>
__global__ DequantizeKernel() {
}

}  // namespace flashinfer

#endif  // FLASHINFER_QUANTIZATION_CUH_
