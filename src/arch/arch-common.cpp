#include "arch-common.h"

namespace dada {
  BaseArch* makeArch(int arch){
    BaseArch* return_ptr = nullptr;
    switch(arch){
      case 70:
        return_ptr = new SM70;
        break;
      case 75:
        return_ptr = new SM75;
        break;
    }       
    return return_ptr;
  }
}