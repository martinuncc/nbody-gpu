#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <cuda_runtime.h>

static constexpr double G         = 6.674e-11;
static constexpr double SOFTENING = 0.1;

struct Simulation {
    size_t n;
    std::vector<double> mass;
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> fx, fy, fz;

    explicit Simulation(size_t nb)
        : n(nb), mass(nb),
          x(nb), y(nb), z(nb),
          vx(nb), vy(nb), vz(nb),
          fx(nb, 0.0), fy(nb, 0.0), fz(nb, 0.0) {}
};

struct DevBuffers {
    double *mass = nullptr;
    double *x = nullptr, *y = nullptr, *z = nullptr;
    double *vx = nullptr, *vy = nullptr, *vz = nullptr;
    double *fx = nullptr, *fy = nullptr, *fz = nullptr;
    size_t n = 0;

    void allocate(size_t nb) {
        n = nb;
        size_t bytes = nb * sizeof(double);
        cudaMalloc(&mass, bytes);
        cudaMalloc(&x,    bytes); cudaMalloc(&y,    bytes); cudaMalloc(&z,    bytes);
        cudaMalloc(&vx,   bytes); cudaMalloc(&vy,   bytes); cudaMalloc(&vz,   bytes);
        cudaMalloc(&fx,   bytes); cudaMalloc(&fy,   bytes); cudaMalloc(&fz,   bytes);
    }

    void free_all() {
        cudaFree(mass);
        cudaFree(x);  cudaFree(y);  cudaFree(z);
        cudaFree(vx); cudaFree(vy); cudaFree(vz);
        cudaFree(fx); cudaFree(fy); cudaFree(fz);
    }

    void upload(const Simulation &s) {
        size_t bytes = n * sizeof(double);
        cudaMemcpy(mass, s.mass.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(x,  s.x.data(),  bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(y,  s.y.data(),  bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(z,  s.z.data(),  bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(vx, s.vx.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(vy, s.vy.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(vz, s.vz.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(fx, s.fx.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(fy, s.fy.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(fz, s.fz.data(), bytes, cudaMemcpyHostToDevice);
    }

    void download(Simulation &s) const {
        size_t bytes = n * sizeof(double);
        cudaMemcpy(s.x.data(),  x,  bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(s.y.data(),  y,  bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(s.z.data(),  z,  bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(s.vx.data(), vx, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(s.vy.data(), vy, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(s.vz.data(), vz, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(s.fx.data(), fx, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(s.fy.data(), fy, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(s.fz.data(), fz, bytes, cudaMemcpyDeviceToHost);
    }
};

__global__ void kernel_reset_force(double *fx, double *fy, double *fz, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fx[i] = 0.0;
    fy[i] = 0.0;
    fz[i] = 0.0;
}

__global__ void kernel_compute_forces(
    const double * __restrict__ mass,
    const double * __restrict__ px, const double * __restrict__ py, const double * __restrict__ pz,
          double * __restrict__ fx,       double * __restrict__ fy,       double * __restrict__ fz,
    size_t n)
{
    extern __shared__ double smem[];
    double *sm_mass = smem;
    double *sm_x    = smem +     blockDim.x;
    double *sm_y    = smem + 2 * blockDim.x;
    double *sm_z    = smem + 3 * blockDim.x;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    double xi = (i < n) ? px[i] : 0.0;
    double yi = (i < n) ? py[i] : 0.0;
    double zi = (i < n) ? pz[i] : 0.0;
    double mi = (i < n) ? mass[i] : 0.0;

    double accx = 0.0, accy = 0.0, accz = 0.0;

    for (size_t tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; ++tile) {
        size_t j = tile * blockDim.x + threadIdx.x;

        sm_mass[threadIdx.x] = (j < n) ? mass[j] : 0.0;
        sm_x[threadIdx.x]    = (j < n) ? px[j]   : 0.0;
        sm_y[threadIdx.x]    = (j < n) ? py[j]   : 0.0;
        sm_z[threadIdx.x]    = (j < n) ? pz[j]   : 0.0;
        __syncthreads();

        if (i < n) {
            size_t tile_end = min((size_t)blockDim.x, n - tile * blockDim.x);
            for (size_t k = 0; k < tile_end; ++k) {
                size_t src = tile * blockDim.x + k;
                if (src == i) continue;

                double dx = sm_x[k] - xi;
                double dy = sm_y[k] - yi;
                double dz = sm_z[k] - zi;

                double dist_sq = dx*dx + dy*dy + dz*dz;
                double norm    = sqrt(dist_sq);
                double F_mag   = G * mi * sm_mass[k] / (dist_sq + SOFTENING);
                double inv_norm = (norm > 0.0) ? (1.0 / norm) : 0.0;

                accx += F_mag * dx * inv_norm;
                accy += F_mag * dy * inv_norm;
                accz += F_mag * dz * inv_norm;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        fx[i] = accx;
        fy[i] = accy;
        fz[i] = accz;
    }
}

__global__ void kernel_integrate(
    const double * __restrict__ mass,
    const double * __restrict__ fx, const double * __restrict__ fy, const double * __restrict__ fz,
          double * __restrict__ vx,       double * __restrict__ vy,       double * __restrict__ vz,
          double * __restrict__ x,        double * __restrict__ y,        double * __restrict__ z,
    double dt, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double inv_m = 1.0 / mass[i];
    vx[i] += fx[i] * inv_m * dt;
    vy[i] += fy[i] * inv_m * dt;
    vz[i] += fz[i] * inv_m * dt;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

void random_init(Simulation &s)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dismass(0.9, 1.0);
    std::normal_distribution<double>       dispos(0.0, 1.0);

    for (size_t i = 0; i < s.n; ++i) {
        s.mass[i] = dismass(gen);
        s.x[i]    = dispos(gen);
        s.y[i]    = dispos(gen);
        s.z[i]    = 0.0;
        s.vx[i]   =  s.y[i] * 1.5;
        s.vy[i]   = -s.x[i] * 1.5;
        s.vz[i]   = 0.0;
    }
}

void init_solar(Simulation &s)
{
    enum Planets { SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE,MOON };
    s = Simulation(10);

    s.mass[SUN]     = 1.9891e30;
    s.mass[MERCURY] = 3.285e23;
    s.mass[VENUS]   = 4.867e24;
    s.mass[EARTH]   = 5.972e24;
    s.mass[MARS]    = 6.39e23;
    s.mass[JUPITER] = 1.898e27;
    s.mass[SATURN]  = 5.683e26;
    s.mass[URANUS]  = 8.681e25;
    s.mass[NEPTUNE] = 1.024e26;
    s.mass[MOON]    = 7.342e22;

    const double AU = 1.496e11;
    s.x  = {0, 0.39*AU, 0.72*AU, 1.0*AU, 1.52*AU, 5.20*AU, 9.58*AU, 19.22*AU, 30.05*AU, 1.0*AU + 3.844e8};
    s.y  = {0,0,0,0,0,0,0,0,0,0};
    s.z  = {0,0,0,0,0,0,0,0,0,0};
    s.vx = {0,0,0,0,0,0,0,0,0,0};
    s.vy = {0,47870,35020,29780,24130,13070,9680,6800,5430,29780+1022};
    s.vz = {0,0,0,0,0,0,0,0,0,0};
}

void load_from_file(Simulation &s, const std::string &filename)
{
    std::ifstream in(filename);
    size_t nbpart; in >> nbpart;
    s = Simulation(nbpart);
    for (size_t i = 0; i < s.n; ++i) {
        in >> s.mass[i];
        in >> s.x[i]  >> s.y[i]  >> s.z[i];
        in >> s.vx[i] >> s.vy[i] >> s.vz[i];
        in >> s.fx[i] >> s.fy[i] >> s.fz[i];
    }
    if (!in.good()) throw std::runtime_error("Failed to read: " + filename);
}

void dump_state(const Simulation &s, std::ofstream &logFile)
{
    std::cout << s.n << '\t';
    for (size_t i = 0; i < s.n; ++i) {
        std::cout << s.mass[i] << '\t'
                  << s.x[i]  << '\t' << s.y[i]  << '\t' << s.z[i]  << '\t'
                  << s.vx[i] << '\t' << s.vy[i] << '\t' << s.vz[i] << '\t'
                  << s.fx[i] << '\t' << s.fy[i] << '\t' << s.fz[i] << '\t';
        logFile << s.mass[i] << '\t'
                << s.x[i]  << '\t' << s.y[i]  << '\t' << s.z[i]  << '\t'
                << s.vx[i] << '\t' << s.vy[i] << '\t' << s.vz[i] << '\t'
                << s.fx[i] << '\t' << s.fy[i] << '\t' << s.fz[i] << '\n';
    }
    std::cout << '\n';
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << "  " << cudaGetErrorString(err) << '\n';              \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

int main(int argc, char *argv[])
{
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " <input> <dt> <nbstep> <printevery>\n";
        return EXIT_FAILURE;
    }

    std::ofstream logFile("log.tsv");
    if (!logFile.is_open()) { std::cerr << "Failed to open log.tsv\n"; return EXIT_FAILURE; }

    const double dt         = std::atof(argv[2]);
    const size_t nbstep     = std::atol(argv[3]);
    const size_t printevery = std::atol(argv[4]);

    Simulation s(1);
    {
        size_t nbpart = std::atol(argv[1]);
        if (nbpart > 0) { s = Simulation(nbpart); random_init(s); }
        else {
            std::string input = argv[1];
            if (input == "planet") init_solar(s);
            else                   load_from_file(s, input);
        }
    }

    constexpr int BLOCK = 256;
    int    grid       = static_cast<int>((s.n + BLOCK - 1) / BLOCK);
    size_t smem_bytes = 4 * BLOCK * sizeof(double);

    DevBuffers dev;
    dev.allocate(s.n);
    dev.upload(s);

    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << "  SMs=" << prop.multiProcessorCount << '\n';
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < nbstep; ++step) {
        if (step % printevery == 0) {
            dev.download(s);
            dump_state(s, logFile);
        }

        kernel_reset_force<<<grid, BLOCK>>>(dev.fx, dev.fy, dev.fz, s.n);

        kernel_compute_forces<<<grid, BLOCK, smem_bytes>>>(
            dev.mass,
            dev.x,  dev.y,  dev.z,
            dev.fx, dev.fy, dev.fz,
            s.n);

        kernel_integrate<<<grid, BLOCK>>>(
            dev.mass,
            dev.fx, dev.fy, dev.fz,
            dev.vx, dev.vy, dev.vz,
            dev.x,  dev.y,  dev.z,
            dt, s.n);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - t_start);

    std::cout << "Simulation took " << duration.count() << " microseconds.\n";
    logFile   << "Simulation took " << duration.count() << " microseconds.\n";

    dev.free_all();
    logFile.close();
    return EXIT_SUCCESS;
}