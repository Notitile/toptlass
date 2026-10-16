#pragma once

#include "tops/tops_runtime.h"
#include <tops.h>


#define TOPS_CHECK(cmd)                                      \
{                                                            \
  topsError_t error  = cmd;                                  \
  if (error != topsSuccess) {                                \
    fprintf(stderr, "error: '%s'(%d) at %s:%d\n",            \
      topsGetErrorString(error), error, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
}