# CUDA 编程入门（6）：程序计时与性能指标

既然我们专注于高性能计算，那么程序的运行时间就是一个非常值得关注的指标，在 CUDA 中，提供了很有用的 API 来帮助我们对程序在 Device 端的运行时间进行统计。下面的代码展示了如何来使用这些工具：

```cpp
void vectorAdd(const float *a,
                const float *b,
                const int n,
                float* c) {
    
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;

    // 声明开始和结束事件
    cudaEvent_t start, stop;
    // 记录开始事件
    cudaEventCreate(&start);
    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, n, d_c);
    // 等待kernel执行完成
    cudaDeviceSynchronize();
    // 记录结束事件
    cudaEventCreate(&stop);
    // 计算事件时间差
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("time: %f ms\n", time);
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}
```

CUDA API 记录时间的逻辑很简单，首先是声明了两个事件分别代表开始和结束，然后在 kernel 执行前后分别记录这两个事件，最后通过 `cudaEventElapsedTime` 计算时间差。值得注意的是 `cudaDeviceSynchronize()` 方法，它的作用是阻塞 Host 线程，直到 Device 端的 kernel 执行完成，因为 kernel 函数执行相对于 Host 线程来说是异步的。

获得程序运行时间之后，我们就可以测量吞吐量来评估程序的性能，然后通过比较硬件的理论性能来评估程序的加速效率。吞吐量的计算公式为

$$
  T = \frac{N}{t}
  $$

其中 $N$ 表示程序运行过程中从内存中读写的数据量，通常以字节为单位，$t$ 表示程序运行时间，单位为秒。所以吞吐量的单位就是 `byte/s`，或者更常用的 `GBps`。比如上面的程序中，设 `n = 1<<25`，每个浮点数占 4 个字节，每次加法运算涉及 3 次内存操作（2次读取+1次写入），于是 `N = n x 4 x 3 = 0.375GB`，假如 kernel 的执行时间为 `2ms`，那么程序的吞吐量就是 `0.375GB / 2ms * 1000ms/s = 187.5GB/s`。