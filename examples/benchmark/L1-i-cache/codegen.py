tail = """
  @p0 bra l1;
  cs2r.32 end, clock_lo;

  s32 time;
  iadd3 time, -start, end, rz;

  s64 ptr;
  s32 tid, imm4;
  s2r tid, threadIdx_x;
  mov imm4, 4;
  imad.wide ptr, tid, imm4, result;

  stg.32 [ptr], time;
  exit;
}
"""

for i in range(1024):
  header = f"""
  list{i}(s64 result){{
    s32 loop;
    mov loop, 1;
    s32 start, end;
    cs2r.32 start, clock_lo;

    pred p0;
    l1:
    isetp.lt.and p0, pt, loop, 512, pt;
    iadd3 loop, loop, 1, rz;
  """
  body = []
  for n in range(i):
    body.append("nop;")
  print(header, '\n'.join(body), tail)