#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;
typedef vector<vector<float>> matrix;

__global__ void matrix_init(float *m, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < nx && j < ny) {
    m[j * nx + i] = 0;
  }
}

__global__ void b_j_i(float *b, float *u, float *v, double dx, double dy, double dt, double rho, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
    b[j * nx + i] = rho * (1 / dt * 
                  ((u[j * nx + i + 1] - u[j * nx + i - 1]) / (2 * dx) + (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)) -
                  pow((u[j * nx + i + 1] - u[j * nx + i - 1]) / (2 * dx), 2) - 
                  2 * ((u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2 * dy) * (v[j * nx + i + 1] - v[j * nx + i - 1]) / (2 * dx)) - 
                  pow((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy), 2));
  }
}

__global__ void p_j_i(float *p, float *pn, float *b, double dx, double dy, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
    p[j * nx + i] = (dy * dy * (pn[j * nx + i + 1] + pn[j * nx + i - 1]) +
                                  dx * dx * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]) -
                                  b[j * nx + i] * dx * dx * dy * dy) /
                                  (2 * (dx * dx + dy * dy));
  }
}

__global__ void p_j(float *p, int nx, int ny) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j < ny) {
    p[j * nx] = p[j * nx + 1];
    p[j * nx + nx - 1] = p[j * nx + nx - 2];
  }
}

__global__ void p_i(float *p, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx) {
    p[i] = p[nx + i];
    p[(ny - 1) * nx + i] = 0;
  }
}

__global__ void u_and_v(float *u, float *v, float *un, float *vn, float *p, double dx, double dy, double dt, double rho, double nu, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
    u[j * nx + i] = un[j * nx + i] - un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + i - 1]) -
                    un[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]) -
                    dt / (2 * rho * dx) * (p[j * nx + i + 1] - p[j * nx + i - 1]) +
                    nu * dt / (dx * dx) * (un[j * nx + i + 1] - 2 * un[j * nx + i] + un[j * nx + i - 1]) +
                    nu * dt / (dy * dy) * (un[(j + 1) * nx + i] - 2 * un[j * nx + i] + un[(j - 1) * nx + i]);

    v[j * nx + i] = vn[j * nx + i] - vn[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + i - 1]) -
                    vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i]) -
                    dt / (2 * rho * dy) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) +
                    nu * dt / (dx * dx) * (vn[j * nx + i + 1] - 2 * vn[j * nx + i] + vn[j * nx + i - 1]) +
                    nu * dt / (dy * dy) * (vn[(j + 1) * nx + i] - 2 * vn[j * nx + i] + vn[(j - 1) * nx + i]);
  }
}

__global__ void u_j_and_v_j(float *u, float *v, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < nx && j < ny) {
    u[j * nx] = 0;
    u[j * nx + nx - 1] = 0;
    v[j * nx] = 0;
    v[j * nx + nx - 1] = 0;
  }
}

__global__ void u_i_and_v_i(float *u, float *v, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < nx && j < ny) {
    u[i] = 0;
    u[(ny - 1) * nx + i] = 1;
    v[i] = 0;
    v[(ny - 1) * nx + i] = 0;
  }
}

int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  // Number of threads in each thread block
  dim3 threadsPerBlock(16, 16);
  // Number of thread blocks in grid
  dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);


  // Initialize matrices
  float *u, *v, *p, *b, *un, *vn, *pn;

  cudaMallocManaged(&u, nx * ny * sizeof(float));
  matrix_init<<<numBlocks, threadsPerBlock>>>(u, nx, ny);
  cudaDeviceSynchronize(); // Blocks until the device has completed all preceding requested tasks.

  cudaMallocManaged(&v, nx * ny * sizeof(float));
  matrix_init<<<numBlocks, threadsPerBlock>>>(v, nx, ny);
  cudaDeviceSynchronize(); 

  cudaMallocManaged(&p, nx * ny * sizeof(float));
  matrix_init<<<numBlocks, threadsPerBlock>>>(p, nx, ny);
  cudaDeviceSynchronize();

  cudaMallocManaged(&b, nx * ny * sizeof(float));
  matrix_init<<<numBlocks, threadsPerBlock>>>(b, nx, ny);
  cudaDeviceSynchronize();

  cudaMallocManaged(&un, nx * ny * sizeof(float));
  matrix_init<<<numBlocks, threadsPerBlock>>>(un, nx, ny);
  cudaDeviceSynchronize();

  cudaMallocManaged(&vn, nx * ny * sizeof(float));
  matrix_init<<<numBlocks, threadsPerBlock>>>(vn, nx, ny);
  cudaDeviceSynchronize();

  cudaMallocManaged(&pn, nx * ny * sizeof(float));
  matrix_init<<<numBlocks, threadsPerBlock>>>(pn, nx, ny);
  cudaDeviceSynchronize();

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

  for (int n=0; n<nt; n++) {

    // Compute b[j][i]
    b_j_i<<<numBlocks, threadsPerBlock>>>(b, u, v, dx, dy, dt, rho, nx, ny);
    cudaDeviceSynchronize();

    for (int it=0; it<nit; it++) {
      // Copy p to pn
      cudaMemcpy(pn, p, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
      // Compute p[j][i]
      p_j_i<<<numBlocks, threadsPerBlock>>>(p, pn, b, dx, dy, nx, ny);
      cudaDeviceSynchronize();

      // Compute p[j][0] and p[j][nx-1]
      p_j<<<numBlocks, threadsPerBlock>>>(p, nx, ny);
      cudaDeviceSynchronize();


      // Compute p[0][i] and p[ny-1][i]
      p_i<<<numBlocks, threadsPerBlock>>>(p, nx, ny);
      cudaDeviceSynchronize();

    }

    // Copy u to un and v to vn
    cudaMemcpy(un, u, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vn, v, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);

    // Compute u[j][i] and v[j][i]
    u_and_v<<<numBlocks, threadsPerBlock>>>(u, v, un, vn, p, dx, dy, dt, rho, nu, nx, ny);
    cudaDeviceSynchronize();

    // Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
    u_j_and_v_j<<<numBlocks, threadsPerBlock>>>(u, v, nx, ny);
    cudaDeviceSynchronize();

    // Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
    u_i_and_v_i<<<numBlocks, threadsPerBlock>>>(u, v, nx, ny);
    cudaDeviceSynchronize();

    if (n % 10 == 0) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j * nx + i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j * nx + i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j * nx + i] << " ";
      pfile << "\n";
    }
  }

  // Free memory
  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(b);
  cudaFree(un);
  cudaFree(vn);
  cudaFree(pn);

  ufile.close();
  vfile.close();
  pfile.close();
}
