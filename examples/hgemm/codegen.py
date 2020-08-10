head = """
.version 6.5
.target sm_75
.address_size 64

.visible .entry hgemm1688_256x256_tn(
  .param .u64 param_A,
  .param .u64 param_B,
  .param .u64 param_C,
  .param .u32 param_m,
  .param .u32 param_n,
  .param .u32 param_k
){
  .reg .u64 A, B, C;
  ld.param.u64 A, [param_A];
  ld.param.u64 B, [param_B];
  ld.param.u64 C, [param_C];

  .reg .u32 m, n, k;
  ld.param.u32 m, [param_m];
  ld.param.u32 n, [param_n];
  ld.param.u32 k, [param_k];


  .reg .u32 lane, warp;
  .reg .u32 bid_x, bid_y, bid_z;
  mov.u32 lane, %tid.x;
  mov.u32 warp, %tid.y;
  mov.u32 bid_x, %ctaid.x;
  mov.u32 bid_y, %ctaid.y;
  mov.u32 bid_z, %ctaid.z;

  .reg .u32 gridDim_x;
  mov.u32 gridDim_x, %nctaid.x;

  .reg .s32 blk_x, blk_y;
  mad.lo.s32 blk_x, bid_z, gridDim_x, bid_x;
  mov.s32 blk_y, bid_y;

  .reg .s32 lane_x, lane_y;
  and.b32 lane_x, lane, 3;
  shr.s32 lane_y, lane, 2;

  .reg .s32 loadA, loadB;
  // loadA = (blk_x*256 + lane_y + warp*8) * lda + lane_x*8; *sizeof(half); where lda = k
  mad.lo.s32 loadA, blk_x, 256, lane_y;
  mad.lo.s32 loadA, warp, 8, loadA;
  mad.lo.s32 loadA, loadA, k, 0;
  mad.lo.s32 loadA, lane_x, 8, loadA;

  // loadB = (blk_y*256 + lane_y + warp*8) * ldb + lane_x*8;
  mad.lo.s32 loadB, blk_y, 256, lane_y;
  mad.lo.s32 loadB, warp, 8, loadB;
  mad.lo.s32 loadB, loadB, k, 0;
  mad.lo.s32 loadB, lane_x, 8, loadB;

  .reg .s64 loadA_ptr, loadB_ptr;
  .reg .s32 loadA_stride, loadB_stride;
  mad.wide.s32 loadA_ptr, loadA, 2, A;
  mad.wide.s32 loadB_ptr, loadB, 2, B;
  mul.lo.s32 loadA_stride, k, 128;
  mul.lo.s32 loadB_stride, k, 128;

  .reg .b32 A_fragment<16>, B_fragment<16>;

  ld.global.nc.v4.b32 {A_fragment0, A_fragment1, A_fragment2, A_fragment3}, [loadA_ptr];
  mad.wide.s32 loadA_ptr, loadA_stride, 1, loadA_ptr;
  ld.global.nc.v4.b32 {A_fragment4, A_fragment5, A_fragment6, A_fragment7}, [loadA_ptr];
  mad.wide.s32 loadA_ptr, loadA_stride, 1, loadA_ptr;
  ld.global.nc.v4.b32 {A_fragment8, A_fragment9, A_fragment10, A_fragment11}, [loadA_ptr];
  mad.wide.s32 loadA_ptr, loadA_stride, 1, loadA_ptr;
  ld.global.nc.v4.b32 {A_fragment12, A_fragment13, A_fragment14, A_fragment15}, [loadA_ptr];

  ld.global.nc.v4.b32 {B_fragment0, B_fragment1, B_fragment2, B_fragment3}, [loadB_ptr];
  mad.wide.s32 loadB_ptr, loadB_stride, 1, loadB_ptr;
  ld.global.nc.v4.b32 {B_fragment4, B_fragment5, B_fragment6, B_fragment7}, [loadB_ptr];
  mad.wide.s32 loadB_ptr, loadB_stride, 1, loadB_ptr;
  ld.global.nc.v4.b32 {B_fragment8, B_fragment9, B_fragment10, B_fragment11}, [loadB_ptr];
  mad.wide.s32 loadB_ptr, loadB_stride, 1, loadB_ptr;
  ld.global.nc.v4.b32 {B_fragment12, B_fragment13, B_fragment14, B_fragment15}, [loadB_ptr];

  .reg .s32 storeAs, storeBs, lane_yd2;
  shr.s32 lane_yd2, lane_y, 1;
  // storeAs = (warp*8*36 + lane_y*32 + lane_y/2*8 + lane_x*8) * sizeof(half);
  mul.lo.s32 storeAs, warp, 288;
  mad.lo.s32 storeAs, lane_y, 32, storeAs;
  mad.lo.s32 storeAs, lane_yd2, 8, storeAs;
  mad.lo.s32 storeAs, lane_x, 8, storeAs;
  mul.lo.s32 storeAs, storeAs, 2; 
  // storeBs = storeAs + 256*36*2
  add.s32 storeBs, storeAs, 18432;

  .reg .s32 loadAs, loadBs, warp_x, warp_y;
  and.b32 warp_x, warp, 3;
  shr.s32 warp_y, warp, 2;
  // loadAs = A[warp_y*128+lane_y][lane_x*2]; (warp_y*128*36+lane_y*32+lane_y/2*8 + lane_x*2) * sizeof(half)
  mad.lo.s32 loadAs, warp_y, 4608, 0;
  mad.lo.s32 loadAs, lane_y, 32, loadAs;
  mad.lo.s32 loadAs, lane_yd2, 8, loadAs;
  mad.lo.s32 loadAs, lane_x, 2, loadAs;
  mad.lo.s32 loadAs, loadAs, 2, 0;  

  // loadBs = B[warp_x*64+lane_y][lane_x*2];
  mad.lo.s32 loadBs, warp_x, 2304, 0;
  mad.lo.s32 loadBs, lane_y, 32, loadBs;
  mad.lo.s32 loadBs, lane_yd2, 8, loadBs;
  mad.lo.s32 loadBs, lane_x, 2, loadBs;
  mad.lo.s32 loadBs, loadBs, 2, 0;
  add.s32 loadBs, loadBs, 18432;

  // storeCs = ((warp_y*32+lane_y)*264 + warp_x*64+lane_x*2) * 2
  .reg .s32 storeCs;
  mad.lo.s32 storeCs, warp_y, 32, lane_y;
  mad.lo.s32 storeCs, storeCs, 264, 0;
  mad.lo.s32 storeCs, warp_x, 64, storeCs;
  mad.lo.s32 storeCs, lane_x, 2, storeCs;
  mad.lo.s32 storeCs, storeCs, 2, 0;

  // loadCs = (warp*8*264 + lane*8)*2;
  .reg .s32 loadCs;
  mad.lo.s32 loadCs, warp, 2112, 0;
  mad.lo.s32 loadCs, lane, 8, loadCs;
  mad.lo.s32 loadCs, loadCs, 2, 0;

  // storeC = (blk_x*256 + warp_x*8 + warp_y*128)*n + blk_y*256 + lane*8
  .reg .s32 storeC;
  mad.lo.s32 storeC, blk_x, 256, 0;
  mad.lo.s32 storeC, warp_x, 8, storeC;
  mad.lo.s32 storeC, warp_y, 128, storeC;
  mad.lo.s32 storeC, storeC, n, 0;
  mad.lo.s32 storeC, blk_y, 256, storeC;
  mad.lo.s32 storeC, lane, 8, storeC;

  .reg .b32 c<128>;
  .reg .b32 a0_<16>;
  .reg .b32 a1_<16>;
  .reg .b32 b0_<8>;
  .reg .b32 b1_<8>;

  mov.u32 c0, 0;
  mov.u32 c1, 0;
  mov.u32 c2, 0;
  mov.u32 c3, 0;
  mov.u32 c4, 0;
  mov.u32 c5, 0;
  mov.u32 c6, 0;
  mov.u32 c7, 0;
  mov.u32 c8, 0;
  mov.u32 c9, 0;
  mov.u32 c10, 0;
  mov.u32 c11, 0;
  mov.u32 c12, 0;
  mov.u32 c13, 0;
  mov.u32 c14, 0;
  mov.u32 c15, 0;
  mov.u32 c16, 0;
  mov.u32 c17, 0;
  mov.u32 c18, 0;
  mov.u32 c19, 0;
  mov.u32 c20, 0;
  mov.u32 c21, 0;
  mov.u32 c22, 0;
  mov.u32 c23, 0;
  mov.u32 c24, 0;
  mov.u32 c25, 0;
  mov.u32 c26, 0;
  mov.u32 c27, 0;
  mov.u32 c28, 0;
  mov.u32 c29, 0;
  mov.u32 c30, 0;
  mov.u32 c31, 0;
  mov.u32 c32, 0;
  mov.u32 c33, 0;
  mov.u32 c34, 0;
  mov.u32 c35, 0;
  mov.u32 c36, 0;
  mov.u32 c37, 0;
  mov.u32 c38, 0;
  mov.u32 c39, 0;
  mov.u32 c40, 0;
  mov.u32 c41, 0;
  mov.u32 c42, 0;
  mov.u32 c43, 0;
  mov.u32 c44, 0;
  mov.u32 c45, 0;
  mov.u32 c46, 0;
  mov.u32 c47, 0;
  mov.u32 c48, 0;
  mov.u32 c49, 0;
  mov.u32 c50, 0;
  mov.u32 c51, 0;
  mov.u32 c52, 0;
  mov.u32 c53, 0;
  mov.u32 c54, 0;
  mov.u32 c55, 0;
  mov.u32 c56, 0;
  mov.u32 c57, 0;
  mov.u32 c58, 0;
  mov.u32 c59, 0;
  mov.u32 c60, 0;
  mov.u32 c61, 0;
  mov.u32 c62, 0;
  mov.u32 c63, 0;
  mov.u32 c64, 0;
  mov.u32 c65, 0;
  mov.u32 c66, 0;
  mov.u32 c67, 0;
  mov.u32 c68, 0;
  mov.u32 c69, 0;
  mov.u32 c70, 0;
  mov.u32 c71, 0;
  mov.u32 c72, 0;
  mov.u32 c73, 0;
  mov.u32 c74, 0;
  mov.u32 c75, 0;
  mov.u32 c76, 0;
  mov.u32 c77, 0;
  mov.u32 c78, 0;
  mov.u32 c79, 0;
  mov.u32 c80, 0;
  mov.u32 c81, 0;
  mov.u32 c82, 0;
  mov.u32 c83, 0;
  mov.u32 c84, 0;
  mov.u32 c85, 0;
  mov.u32 c86, 0;
  mov.u32 c87, 0;
  mov.u32 c88, 0;
  mov.u32 c89, 0;
  mov.u32 c90, 0;
  mov.u32 c91, 0;
  mov.u32 c92, 0;
  mov.u32 c93, 0;
  mov.u32 c94, 0;
  mov.u32 c95, 0;
  mov.u32 c96, 0;
  mov.u32 c97, 0;
  mov.u32 c98, 0;
  mov.u32 c99, 0;
  mov.u32 c100, 0;
  mov.u32 c101, 0;
  mov.u32 c102, 0;
  mov.u32 c103, 0;
  mov.u32 c104, 0;
  mov.u32 c105, 0;
  mov.u32 c106, 0;
  mov.u32 c107, 0;
  mov.u32 c108, 0;
  mov.u32 c109, 0;
  mov.u32 c110, 0;
  mov.u32 c111, 0;
  mov.u32 c112, 0;
  mov.u32 c113, 0;
  mov.u32 c114, 0;
  mov.u32 c115, 0;
  mov.u32 c116, 0;
  mov.u32 c117, 0;
  mov.u32 c118, 0;
  mov.u32 c119, 0;
  mov.u32 c120, 0;
  mov.u32 c121, 0;
  mov.u32 c122, 0;
  mov.u32 c123, 0;
  mov.u32 c124, 0;
  mov.u32 c125, 0;
  mov.u32 c126, 0;
  mov.u32 c127, 0;

  st.shared.v4.b32 [storeAs], {A_fragment0, A_fragment1, A_fragment2, A_fragment3};
  st.shared.v4.b32 [storeAs+4608], {A_fragment4, A_fragment5, A_fragment6, A_fragment7};
  st.shared.v4.b32 [storeAs+9216], {A_fragment8, A_fragment9, A_fragment10, A_fragment11};
  st.shared.v4.b32 [storeAs+13824], {A_fragment12, A_fragment13, A_fragment14, A_fragment15};
  st.shared.v4.b32 [storeBs+0], {B_fragment0, B_fragment1, B_fragment2, B_fragment3};
  st.shared.v4.b32 [storeBs+4608], {B_fragment4, B_fragment5, B_fragment6, B_fragment7};
  st.shared.v4.b32 [storeBs+9216], {B_fragment8, B_fragment9, B_fragment10, B_fragment11};
  st.shared.v4.b32 [storeBs+13824], {B_fragment12, B_fragment13, B_fragment14, B_fragment15};
  bar.sync 0;
  ld.shared.b32 a0_0, [loadAs+0];
  ld.shared.b32 a0_1, [loadAs+576];
  ld.shared.b32 a0_2, [loadAs+1152];
  ld.shared.b32 a0_3, [loadAs+1728];
  ld.shared.b32 a0_4, [loadAs+2304];
  ld.shared.b32 a0_5, [loadAs+2880];
  ld.shared.b32 a0_6, [loadAs+3456];
  ld.shared.b32 a0_7, [loadAs+4032];
  ld.shared.b32 a0_8, [loadAs+4608];
  ld.shared.b32 a0_9, [loadAs+5184];
  ld.shared.b32 a0_10, [loadAs+5760];
  ld.shared.b32 a0_11, [loadAs+6336];
  ld.shared.b32 a0_12, [loadAs+6912];
  ld.shared.b32 a0_13, [loadAs+7488];
  ld.shared.b32 a0_14, [loadAs+8064];
  ld.shared.b32 a0_15, [loadAs+8640];
  ld.shared.b32 b0_0, [loadBs+0];
  ld.shared.b32 b0_1, [loadBs+576];
  ld.shared.b32 b0_2, [loadBs+1152];
  ld.shared.b32 b0_3, [loadBs+1728];
  ld.shared.b32 b0_4, [loadBs+2304];
  ld.shared.b32 b0_5, [loadBs+2880];
  ld.shared.b32 b0_6, [loadBs+3456];
  ld.shared.b32 b0_7, [loadBs+4032];

  .reg .pred p0;
  .reg .s32 k_iter;
  mov.u32 k_iter, 32;

main_loop:
  setp.lt.s32 p0, k_iter, k;
  add.s32 k_iter, k_iter, 32;
"""

insert = {}

insert = {}
for k in range(3):
  buf = (k+1)%2
  for row in range(16):    
    insert[k*64+row] = f'ld.shared.b32 a{buf}_{row}, [loadAs+{row*8*36*2 + (k+1)*8*2}];'
  for col in range(8):
    insert[k*64+16+col] = f'ld.shared.b32 b{buf}_{col}, [loadBs+{col*8*36*2 + (k+1)*8*2}];'
# last lds
for row in range(16):
  insert[3*64+20+row] = f'@p0 ld.shared.b32 a0_{row}, [loadAs+{row*8*36*2}];'
for col in range(8):
  insert[3*64+36+col] = f'@p0 ld.shared.b32 b0_{col}, [loadBs+{col*8*36*2}];'

# ldg and sts 
insert[24] = '@p0 add.s32 loadA, loadA, 32;'
insert[25] = '@p0 add.s32 loadB, loadB, 32;'
insert[26] = '@p0 mad.wide.s32 loadA_ptr, loadA, 2, A;'
insert[27] = '@p0 mad.wide.s32 loadB_ptr, loadB, 2, B;'
for a in range(4):
  insert[28+a*8] = f'@p0 ld.global.nc.v4.b32 {{A_fragment{a*4}, A_fragment{a*4+1}, A_fragment{a*4+2}, A_fragment{a*4+3}}}, [loadA_ptr];'
  if a != 3:
    insert[28+a*8+4] = '@p0 mad.wide.s32 loadA_ptr, loadA_stride, 1, loadA_ptr;'
for b in range(4):
  insert[64+28+b*8] = f'@p0 ld.global.nc.v4.b32 {{B_fragment{b*4}, B_fragment{b*4+1}, B_fragment{b*4+2}, B_fragment{b*4+3}}}, [loadB_ptr];'
  if b != 3:
    insert[64+28+b*8+4] = '@p0 mad.wide.s32 loadB_ptr, loadB_stride, 1, loadB_ptr;'
# sts
insert[64*2+40] = '@p0 bar.sync 0;'
for a in range(4):
  insert[64*2+42+a*5] = f'@p0 st.shared.v4.b32 [storeAs+{a*64*36*2}], {{A_fragment{a*4}, A_fragment{a*4+1}, A_fragment{a*4+2}, A_fragment{a*4+3}}};'
for b in range(4):
  insert[64*3+b*5] = f'@p0 st.shared.v4.b32 [storeBs+{b*64*36*2}], {{B_fragment{b*4}, B_fragment{b*4+1}, B_fragment{b*4+2}, B_fragment{b*4+3}}};'
insert[64*3+18] = '@p0 bar.sync 0;'

main_loop = []
for k in range(4):
  buf = k%2
  for col in range(8):
    for row in range(8):
      i = (col*8 + row)*2
      main_loop.append(f'mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {{c{i}, c{i+1}}}, {{a{buf}_{row*2}, a{buf}_{row*2+1}}}, {{b{buf}_{col}}}, {{c{i}, c{i+1}}};')
      if k*64+col*8+row in insert:
        main_loop.append(insert[k*64+col*8+row])

store_c = []
store_c.append("""
@p0 bra.uni main_loop;

.reg .s64 storeC_ptr;
.reg .s32 storeC_stride;

// storeC_stride = 2*n;
mul.lo.s32 storeC_stride, 2, n;

.reg .b32 C_fragment<4>;
""")
for rnd in range(4):
  store_c.append('bar.sync 0;')
  for col in range(8):
    for row in range(4):
      store_c.append(f'st.shared.b32 [storeCs+{(row*8*264+col*8)*2}], c{col*16+rnd*4+row};')
  store_c.append('bar.sync 0;')
  store_c.append('mad.wide.s32 storeC_ptr, storeC, 2, C;')
  for row in range(8):
    store_c.append(f'ld.shared.v4.b32 {{C_fragment0, C_fragment1, C_fragment2, C_fragment3}}, [loadCs+{row*264*2}];')
    store_c.append(f'st.global.v4.b32 [storeC_ptr], {{C_fragment0, C_fragment1, C_fragment2, C_fragment3}};')
    if row != 7:
      store_c.append('mad.wide.s32 storeC_ptr, storeC_stride, 1, storeC_ptr;')
  if rnd != 3:
    store_c.append('mad.lo.s32 storeC, 32, n, storeC;')


print(head, 
      '\n'.join(main_loop)+'\n', 
      '\n'.join(store_c)+'\n',
      'exit;\n}')