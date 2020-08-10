header = """

.version 6.4
.target sm_75
.address_size 64


.visible .entry sgemm_128x128_nt(
	.param .u64 param_A,
	.param .u64 param_B,
	.param .u64 param_C,
	.param .u32 param_m,
	.param .u32 param_n,
	.param .u32 param_k
){
	.reg .u64 A, B, C;
	.reg .u32 m, n, k;
	ld.param.u64 A, [param_A];
	ld.param.u64 B, [param_B];
	ld.param.u64 C, [param_C];
	ld.param.u32 m, [param_m];
	ld.param.u32 n, [param_n];
	ld.param.u32 k, [param_k];

	.reg .s32 lane, warp, bid_x, bid_y, bid_z;
	mov.s32 lane, %tid.x;
	mov.s32 warp, %tid.y;
	mov.s32 bid_x, %ctaid.x;
	mov.s32 bid_y, %ctaid.y;
	mov.s32 bid_z, %ctaid.z;

	.reg .s32 blk_x, blk_y;
	.reg .s32 gridDim_x;
	mov.u32 gridDim_x, %nctaid.x;
	mad.lo.s32 blk_x, bid_z, gridDim_x, bid_x;
	mov.s32 blk_y, bid_y;

	.reg .s32 loadA, loadB;
	.reg .s32 lane4;
	.reg .s64 loadA_ptr, loadB_ptr;
	mul.lo.s32 lane4, lane, 4;
	mad.lo.s32 loadA, blk_x, 128, lane4;
	mad.lo.s32 loadB, blk_y, 128, lane4;
	mad.lo.s32 loadA, warp, m, loadA;
	mad.lo.s32 loadB, warp, n, loadB;

	mad.wide.s32 loadA_ptr, loadA, 4, A;
	mad.wide.s32 loadB_ptr, loadB, 4, B;

	.reg .f32 A_frag<4>, B_frag<4>;
	ld.global.nc.v4.f32 {A_frag0, A_frag1, A_frag2, A_frag3}, [loadA_ptr];
	ld.global.nc.v4.f32 {B_frag0, B_frag1, B_frag2, B_frag3}, [loadB_ptr];

	// storeAs, storeBs
	.reg .s32 storeAs, storeBs;
	mad.lo.s32 storeAs, warp, 128, lane4;
	mul.lo.s32 storeAs, storeAs, 4; // sizeof float
	add.s32 storeBs, storeAs, 4096;

	// loadAs, loadBs
	// warp_x = warp % 2; warp_y = warp / 2;
	// lane_x = lane%16/2; lane_y = lane/16*2 + lane%16%2;
	.reg .s32 loadAs, loadBs;
	.reg .s32 warp_x, warp_y, lane_x, lane_y, lane_m16, lane_d16;
	and.b32 warp_x, warp, 1;
	shr.s32 warp_y, warp, 1;
	and.b32 lane_m16, lane, 15;
	shr.s32 lane_d16, lane, 4;
	shr.s32 lane_x, lane_m16, 1;
	and.b32 lane_y, lane_m16, 1;
	mad.lo.s32 lane_y, lane_d16, 2, lane_y;
	// loadAs = (warp_x*64 + lane_x*4)*sizeof(float);
	// loadBs = (warp_y*32 + lane_y*4 + 8*128)*sizeof(float);
	mul.lo.s32 loadAs, warp_x, 64;
	mad.lo.s32 loadAs, lane_x, 4, loadAs;
	mad.lo.s32 loadBs, warp_y, 32, 1024;
	mad.lo.s32 loadBs, lane_y, 4, loadBs;
	mul.lo.s32 loadAs, loadAs, 4;
	mul.lo.s32 loadBs, loadBs, 4;

	// storeC
	// int c_x = blockIdx_x*128 + warp_x*64 + lane_x*4; 
  // int c_y = blockIdx_y*128 + warp_y*32 + lane_y*4;
  // int storeC_offset = c_x + c_y*m;
	.reg .s32 storeC, c_x, c_y;
	mul.lo.s32 c_x, lane_x, 4;
	mad.lo.s32 c_x, warp_x, 64, c_x;
	mad.lo.s32 c_x, blk_x, 128, c_x;
	mul.lo.s32 c_y, lane_y, 4;
	mad.lo.s32 c_y, warp_y, 32, c_y;
	mad.lo.s32 c_y, blk_y, 128, c_y;
	mad.lo.s32 storeC, c_y, m, c_x;

	.reg .f32 c<64>;
	mov.f32 c0, 0f00000000;
	mov.f32 c1, 0f00000000;
	mov.f32 c2, 0f00000000;
	mov.f32 c3, 0f00000000;
	mov.f32 c4, 0f00000000;
	mov.f32 c5, 0f00000000;
	mov.f32 c6, 0f00000000;
	mov.f32 c7, 0f00000000;
	mov.f32 c8, 0f00000000;
	mov.f32 c9, 0f00000000;
	mov.f32 c10, 0f00000000;
	mov.f32 c11, 0f00000000;
	mov.f32 c12, 0f00000000;
	mov.f32 c13, 0f00000000;
	mov.f32 c14, 0f00000000;
	mov.f32 c15, 0f00000000;
	mov.f32 c16, 0f00000000;
	mov.f32 c17, 0f00000000;
	mov.f32 c18, 0f00000000;
	mov.f32 c19, 0f00000000;
	mov.f32 c20, 0f00000000;
	mov.f32 c21, 0f00000000;
	mov.f32 c22, 0f00000000;
	mov.f32 c23, 0f00000000;
	mov.f32 c24, 0f00000000;
	mov.f32 c25, 0f00000000;
	mov.f32 c26, 0f00000000;
	mov.f32 c27, 0f00000000;
	mov.f32 c28, 0f00000000;
	mov.f32 c29, 0f00000000;
	mov.f32 c30, 0f00000000;
	mov.f32 c31, 0f00000000;
	mov.f32 c32, 0f00000000;
	mov.f32 c33, 0f00000000;
	mov.f32 c34, 0f00000000;
	mov.f32 c35, 0f00000000;
	mov.f32 c36, 0f00000000;
	mov.f32 c37, 0f00000000;
	mov.f32 c38, 0f00000000;
	mov.f32 c39, 0f00000000;
	mov.f32 c40, 0f00000000;
	mov.f32 c41, 0f00000000;
	mov.f32 c42, 0f00000000;
	mov.f32 c43, 0f00000000;
	mov.f32 c44, 0f00000000;
	mov.f32 c45, 0f00000000;
	mov.f32 c46, 0f00000000;
	mov.f32 c47, 0f00000000;
	mov.f32 c48, 0f00000000;
	mov.f32 c49, 0f00000000;
	mov.f32 c50, 0f00000000;
	mov.f32 c51, 0f00000000;
	mov.f32 c52, 0f00000000;
	mov.f32 c53, 0f00000000;
	mov.f32 c54, 0f00000000;
	mov.f32 c55, 0f00000000;
	mov.f32 c56, 0f00000000;
	mov.f32 c57, 0f00000000;
	mov.f32 c58, 0f00000000;
	mov.f32 c59, 0f00000000;
	mov.f32 c60, 0f00000000;
	mov.f32 c61, 0f00000000;
	mov.f32 c62, 0f00000000;
	mov.f32 c63, 0f00000000;

	st.shared.v4.f32 [storeAs], {A_frag0, A_frag1, A_frag2, A_frag3};
	st.shared.v4.f32 [storeBs], {B_frag0, B_frag1, B_frag2, B_frag3};

	bar.sync 0;

	.reg .f32 a0_<8>, a1_<8>, b0_<8>, b1_<8>;
	ld.shared.v4.f32 {a0_0, a0_1, a0_2, a0_3}, [loadAs];
	ld.shared.v4.f32 {a0_4, a0_5, a0_6, a0_7}, [loadAs+128];
	ld.shared.v4.f32 {b0_0, b0_1, b0_2, b0_3}, [loadBs];
	ld.shared.v4.f32 {b0_4, b0_5, b0_6, b0_7}, [loadBs+64];

	xor.b32 storeAs, storeAs, 8192;
	xor.b32 storeBs, storeBs, 8192;

	.reg .pred p0;
	.reg .s32 k_local;
	mov.s32 k_local, 8;

main_loop:
  setp.lt.s32 p0, k_local, k;
	add.s32 k_local, k_local, 8;
"""

tail = """
  @p0 bra main_loop;

	.reg .s64 storeC_ptr;

	mad.wide.s32 storeC_ptr, storeC, 4, C;
	st.global.v4.f32 [storeC_ptr], {c0, c1, c2, c3};
	st.global.v4.f32 [storeC_ptr+128], {c4, c5, c6, c7};
	mad.wide.s32 storeC_ptr, 4, m, storeC_ptr;
	st.global.v4.f32 [storeC_ptr], {c8, c9, c10, c11};
	st.global.v4.f32 [storeC_ptr+128], {c12, c13, c14, c15};
	mad.wide.s32 storeC_ptr, 4, m, storeC_ptr;
	st.global.v4.f32 [storeC_ptr], {c16, c17, c18, c19};
	st.global.v4.f32 [storeC_ptr+128], {c20, c21, c22, c23};
	mad.wide.s32 storeC_ptr, 4, m, storeC_ptr;
	st.global.v4.f32 [storeC_ptr], {c24, c25, c26, c27};
	st.global.v4.f32 [storeC_ptr+128], {c28, c29, c30, c31};

	mad.wide.s32 storeC_ptr, 52, m, storeC_ptr; // 52 = 13*sizeof(float)


	st.global.v4.f32 [storeC_ptr], {c32, c33, c34, c35};
	st.global.v4.f32 [storeC_ptr+128], {c36, c37, c38, c39};
	mad.wide.s32 storeC_ptr, 4, m, storeC_ptr;
	st.global.v4.f32 [storeC_ptr], {c40, c41, c42, c43};
	st.global.v4.f32 [storeC_ptr+128], {c44, c45, c46, c47};
	mad.wide.s32 storeC_ptr, 4, m, storeC_ptr;
	st.global.v4.f32 [storeC_ptr], {c48, c49, c50, c51};
	st.global.v4.f32 [storeC_ptr+128], {c52, c53, c54, c55};
	mad.wide.s32 storeC_ptr, 4, m, storeC_ptr;
	st.global.v4.f32 [storeC_ptr], {c56, c57, c58, c59};
	st.global.v4.f32 [storeC_ptr+128], {c60, c61, c62, c63};
	ret;
}
"""

lds_dist = 2
lds_base = 1
insert = {}
for i in range(7):
  j = (i+1)%2
  insert[i*64+lds_base+lds_dist*0] = f'ld.shared.v4.f32 {{a{j}_0, a{j}_1, a{j}_2, a{j}_3}}, [loadAs+{(i+1)*512+0}];'
  insert[i*64+lds_base+lds_dist*2] = f'ld.shared.v4.f32 {{b{j}_0, b{j}_1, b{j}_2, b{j}_3}}, [loadBs+{(i+1)*512+0}];'
  insert[i*64+lds_base+lds_dist*1] = f'ld.shared.v4.f32 {{a{j}_4, a{j}_5, a{j}_6, a{j}_7}}, [loadAs+{(i+1)*512+128}];'
  insert[i*64+lds_base+lds_dist*3] = f'ld.shared.v4.f32 {{b{j}_4, b{j}_5, b{j}_6, b{j}_7}},  [loadBs+{(i+1)*512+64}];'

insert[16+lds_base+lds_dist*4] = 'mad.wide.s32 loadA_ptr, 32, m, loadA_ptr;'
insert[16+lds_base+lds_dist*4+20] = '@p0 ld.global.nc.v4.f32 {A_frag0, A_frag1, A_frag2, A_frag3}, [loadA_ptr];'
insert[16+64+lds_base+lds_dist*4] = 'mad.wide.s32 loadB_ptr, 32, n, loadB_ptr;'
insert[16+64+lds_base+lds_dist*4+20] = '@p0 ld.global.nc.v4.f32 {B_frag0, B_frag1, B_frag2, B_frag3}, [loadB_ptr];'

sts_dist = 5
sts_base = 2
insert[5*64+40] = '@p0 st.shared.v4.f32 [storeAs], {A_frag0, A_frag1, A_frag2, A_frag3};'
insert[6*64+40] = '@p0 st.shared.v4.f32 [storeBs], {B_frag0, B_frag1, B_frag2, B_frag3};'
insert[6*64+57] = '@p0 bar.sync 0;'
insert[7*64+8+1] = '@p0 xor.b32 loadAs, loadAs, 8192;'
insert[7*64+8+2] = '@p0 xor.b32 loadBs, loadBs, 8192;'
insert[7*64+8+3] = '@p0 xor.b32 storeAs, storeAs, 8192;'
insert[7*64+8+4] = '@p0 xor.b32 storeBs, storeBs, 8192;'
insert[7*64+18+lds_dist*0] = '@p0 ld.shared.v4.f32 {a0_0, a0_1, a0_2, a0_3}, [loadAs];'
insert[7*64+18+lds_dist*1] = '@p0 ld.shared.v4.f32 {b0_0, b0_1, b0_2, b0_3}, [loadBs];'
insert[7*64+18+lds_dist*2] = '@p0 ld.shared.v4.f32 {a0_4, a0_5, a0_6, a0_7}, [loadAs+128];'
insert[7*64+18+lds_dist*3] = '@p0 ld.shared.v4.f32 {b0_4, b0_5, b0_6, b0_7}, [loadBs+64];'



main_loop = []
for i in range(8):
  j = i%2
  main_loop.append(f'// k={i}')
  for b in range(8):
    for a in range(8):
      if i*64+b*8+a in insert:
        main_loop.append(insert[i*64+b*8+a])
      main_loop.append(f'fma.rn.f32 c{8*b+a}, a{j}_{a}, b{j}_{b}, c{8*b+a};')






print(header, '\n'.join(main_loop), tail)