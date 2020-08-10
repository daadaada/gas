grammar dada;

type_specifier
   : F16X2 | B16X2 | U32 | S32 | F32 | B32 | U64 | S64 | F64 | B64 | PRED; 

memory_operand
   : '[' ID ']'
   | '[' ID '+' NEG? CONSTANT ']'
   ;

operand
   : NEG? ID 
   | NEG? ID '[' CONSTANT ']' 
   | NEG? ID '[' CONSTANT ':' CONSTANT ']'
   | NEG? CONSTANT
   | NEG? FLOAT_CONSTANT
   | NOT ID
   | memory_operand
   ;

declare_operand_list
   : declare_operand 
   | declare_operand_list ',' declare_operand;

declare_operand
   : ID | ID '[' CONSTANT ']';

translation_unit
   : kernel_defination+;


compound_statement
    : '{' statement* '}'
    ;

label
   : ID ':';

statement
   : var_declaration | instruction | label;

var_declaration
   : type_specifier vector_length? declare_operand_list SEMI;

vector_length
   : V2 | V4;

predicate_mask
   : '@' NOT? ID;

instruction
   : predicate_mask? opcode flag* operand_list? SEMI
   ;

operand_list
   : operand
   | operand_list ',' operand
   ;

kernel_defination
    : ID '(' parameter_list ')' compound_statement; 

parameter_list
    : parameter
    | parameter_list ',' parameter
    ;

parameter
    : type_specifier ID;

opcode 
   : HADD2 | HFMA2 | HMUL2 
   | FADD  | FFMA  | FMUL | MUFU  
   | DADD  | DFMA  | DMUL
   | ISETP | PSETP |
   | IADD3 | IMAD  | LEA
   | HMMA  | IMMA  | BMMA 
   | SHF   | LOP3  | PLOP3 
   | IMNMX | FMNMX 
   | I2I   | I2F   | F2I  | F2F
   | LDG   | LDS   | LDSM | LDC | STG  | STS  | MOV  | SHFL
   | BRA   | EXIT | JMP 
   | P2R   | R2P
   | BAR   | CS2R  | NOP  | S2R
   ;

flag
   : RN | RZ | RM | RP 
   | TRUNC | FLOOR | CEIL
   | HMMA1688 | IMMA8816 | IMMA8832 | BMMA88128
   | WIDE
   | X
   | EQ | NE | LT | LE | GT | GE 
   | EQU | NEU | LEU | GTU | GEU | NAN | NUM
   | LDST16 | LDST32 | LDST64 | LDST128
   | COS | SIN | EX2 | LG2 | RCP | RSQ 
   | AND | XOR | OR 
   | CTA | GPU | SYS
   | MCONSTANT | MWEAK | MSTRONG
   | EF | EL | LU 
   | L | R 
   | LUT 
   | SYNC
   | F16T | F32T | S32T | U32T | U8T | S8T | U4T | S4T 
   ;

/* Lexer rules */
HADD2 : 'hadd2';
HFMA2 : 'hfma2';
HMUL2 : 'hmul2';
FADD  : 'fadd';
FFMA  : 'ffma';
FMUL  : 'fmul';
MUFU  : 'mufu';
DADD  : 'dadd';
DFMA  : 'dfma';
DMUL  : 'dmul';
IADD3 : 'iadd3';
IMAD  : 'imad';
ISETP : 'isetp';
PSETP : 'psetp';
LEA   : 'lea';
HMMA  : 'hmma';
IMMA  : 'imma';
BMMA  : 'bmma';
SHF   : 'shf';
LOP3  : 'lop3';
PLOP3 : 'plop3';
IMNMX : 'imnmx';
FMNMX : 'fmnmx';
I2I   : 'i2i';
I2F   : 'i2f';
F2I   : 'f2i';
F2F   : 'f2f';
LDG   : 'ldg';
LDS   : 'lds';
LDSM  : 'ldsm';
LDC   : 'ldc';
STG   : 'stg';
STS   : 'sts';
MOV   : 'mov';
SHFL  : 'shfl';
BRA   : 'bra';
EXIT  : 'exit';
JMP   : 'jmp';
P2R   : 'p2r';
R2P   : 'r2p';
BAR   : 'bar';
CS2R  : 'cs2r';
NOP   : 'nop';
S2R   : 's2r';

RN : '.rn';
RZ : '.rz';
RM : '.rm';
RP : '.rp';
TRUNC : '.trunc';
FLOOR : '.floor';
CEIL : '.ceil';
HMMA1688 : '.1688';
IMMA8816 : '.8816';
IMMA8832 : '.8832';
BMMA88128: '.88128';
WIDE : '.wide';
X : '.x';
EQ : '.eq';
NE : '.ne';
LT : '.lt';
LE : '.le';
GT : '.gt';
GE : '.ge';
EQU : '.equ';
NEU : '.neu';
LEU : '.leu';
GTU : '.gtu';
GEU : '.geu';
NAN : '.nan';
NUM : '.num';
LDST16 : '.16';
LDST32 : '.32';
LDST64 : '.64';
LDST128: '.128';
COS : '.cos';
SIN : '.sin';
EX2 : '.ex2';
LG2 : '.lg2';
RCP : '.rcp';
RSQ : '.rsq';
AND : '.and';
XOR : '.xor';
OR  : '.or';
CTA : '.cta';
GPU : '.gpu';
SYS : '.sys';
MCONSTANT : '.constant';
MWEAK     : '.weak';
MSTRONG   : '.strong';
EF : '.ef';
EL : '.el';
LU : '.lu';
L : '.l';
R : '.r';
LUT : '.lut';
SYNC : '.sync';
F16T : '.f16'; // F16T (fp16-type)
F32T : '.f32';
S32T : '.s32';
U32T : '.u32';
U8T  : '.u8';
S8T  : '.s8';
S4T  : '.s4';
U4T  : '.u4';

F16X2 : 'f16x2';
B16X2 : 'b16x2';
U32 : 'u32';
S32 : 's32';
F32 : 'f32';
B32 : 'b32';
U64 : 'u64';
S64 : 's64';
F64 : 'f64';
B64 : 'b64';
PRED: 'pred';

V2 : '.v2';
V4 : '.v4';

CONSTANT : [0-9]+;
FLOAT_CONSTANT : [0-9]+'.'[0-9]+;
ID: [a-zA-Z][a-zA-Z0-9_]*;
NOT: '!';
NEG: '-';
SEMI: ';';
WS : [ \t\n] -> skip ;
BlockComment : '/*' .*? '*/' -> skip;
LineComment  : '//' ~[\r\n]* -> skip;
