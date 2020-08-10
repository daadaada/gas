__global__
void sgemm_128x128_nt_cuda(float* A, float* B, float* C, int m, int n, int k){
  // C[y][x] = A[k][x] * B[k][y]
  float A_fragment[16];
  float B_fragment[16];

  float a[8];
  float b[8];
  float c[64];

  for(int i=0; i<64; ++i) { c[i] = 0.0f;}

  __shared__ float A_smem[32][128];
  __shared__ float B_smem[32][128];

  int warp_x = threadIdx.y % 2;
  int warp_y = threadIdx.y / 2;
  int lane_x = threadIdx.x % 8;
  int lane_y = threadIdx.x / 8;

  int x_g = blockIdx.x*128 + threadIdx.x*4;
  int y_g = blockIdx.y*128 + threadIdx.x*4;
  int k_g = threadIdx.y; // warpid

  for(int iter=0; iter<k/32; ++iter){
    for(int r=0; r<4; ++r){ // 4 rounds
      for(int i=0; i<4; ++i){ // float4
        A_fragment[r*4 +i] = A[(iter*32+r*8+k_g)*m + x_g+i];
        B_fragment[r*4 +i] = B[(iter*32+r*8+k_g)*n + y_g+i];}}

    for(int r=0; r<4; ++r){
      for(int i=0; i<4; ++i){
        A_smem[r*8 + threadIdx.y][threadIdx.x*4 +i] = A_fragment[r*4 + i];
        B_smem[r*8 + threadIdx.y][threadIdx.x*4 +i] = B_fragment[r*4 + i];}}

    __syncthreads();

    for(int k_local=0; k_local<32; ++k_local){ 
      for(int warp_tile_x=0; warp_tile_x<2; ++warp_tile_x){
        for(int i=0; i<4; ++i){
          a[warp_tile_x*4 + i] = A_smem[k_local][warp_x*64 + warp_tile_x*32 + lane_x*4 + i];}}
      for(int warp_tile_y=0; warp_tile_y<2; ++warp_tile_y){
        for(int i=0; i<4; ++i){
          b[warp_tile_y*4 + i] = B_smem[k_local][warp_y*32 + warp_tile_y*16 + lane_y*4 + i];}}

      // col-major
      for(int col=0; col<8; ++col){
        for(int row=0; row<8; ++row){
          c[col*8 + row] += a[row] * b[col];}}
    } // for k_local=0...31
    __syncthreads();
  } // end of main loop

  int c_x_base = blockIdx.x*128 + warp_x*64 + /*warp_tile_x*32*/ lane_x*4 /*+x*/;
  int c_y_base = blockIdx.y*128 + warp_y*32 + /*warp_tile_y*16*/ lane_y*4 /*+y*/;
  // store C to gmem
  for(int warp_tile_y=0; warp_tile_y<2; ++warp_tile_y){
    for(int y=0; y<4; ++y){
      for(int warp_tile_x=0; warp_tile_x<2; ++warp_tile_x){
        for(int x=0; x<4; ++x){
          C[(c_y_base+warp_tile_y*16+y)*m + (c_x_base+warp_tile_x*32+x)] = c[(warp_tile_y*4+y)*8 + warp_tile_x*4+x];}}}}
} // sgemm_128x128_nt
