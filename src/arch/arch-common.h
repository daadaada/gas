#pragma once

#include "base_arch.h"
#include "arch_sm7x.h"
#include "arch_sm70.h"
#include "arch_sm75.h"

namespace dada {
  BaseArch* makeArch(int arch);
}