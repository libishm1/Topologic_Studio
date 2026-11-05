/**
 * The Flamescapes solver on the GPU through WebGPU: the paper's own
 * implementation runs WGSL compute kernels through WGPU. Until the kernels
 * land, createGpuRunner returns null and the worker runs the CPU solver.
 */
export async function createGpuRunner() {
  return null;
}
