/*
 * Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdint.h>

// WARP_SIZE: compile-time constant used for shared-memory sizing.
// `warpSize` (HIP/CUDA device-runtime built-in) is not constexpr,
// so we keep a literal here. Guarded so the build can override
// it (e.g. -DWARP_SIZE=64) for future wider-wavefront targets.
#ifndef WARP_SIZE
#define WARP_SIZE       32
#endif
// Fixed at 128 (not WARP_SIZE*4): the ip1 kernels tile the Gv axis by the fixed
// NG_PER_BLOCK=32 below, so the block size must not scale with the wavefront --
// otherwise host launch (host pass WARP_SIZE=32) and device (per-arch WARP_SIZE)
// disagree, and a wave64 block of 256 doubles the static shared array past LDS.
#define FT_AO_THREADS   128
#define NG_PER_BLOCK    32

typedef struct {
    int li;
    int lj;
    int nfi;
    int nfj;
    int g_size;
    int stride_i;
    int stride_j;
    int iprim;
    int jprim;
    int npairs_ij;
    int ngrids;
    int *bas_ij_idx;
    double *grids;
    int *img_offsets; // offset AFTIntEnvVars.img_idx for each shell-pair
    int *img_idx; // indices of img_coords in each shell-pair
} AFTBoundsInfo;

typedef struct {
    int ngrids;
    // The effective basis pair Id = ish*nbas+jsh
    int *bas_ij_idx;
    // the bas_ij_idx offset for each blockIdx.x
    int *shl_pair_offsets;
    // the AO-pair offset (address) in the output tensor for each blockIdx.x
    int *ao_pair_loc;
    // offset AFTIntEnvVars.img_idx for each shell-pair
    int *img_offsets;
    // indices of img_coords in each shell-pair
    int *img_idx;
    // gout_stride for for each (li,lj) pattern
    int *gout_stride_lookup;
    double *grids;
} BDivAFTBoundsInfo;
