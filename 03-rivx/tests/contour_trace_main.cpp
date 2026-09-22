// Runs the self-test built into contour_trace.cpp (synthetic disk -> traced,
// simplified, resampled contour). Exit code 0 on pass.
#include "curve/contour_trace.hpp"
int main() { return rive_backend::contourTraceSelfTest(); }
