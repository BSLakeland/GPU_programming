import numpy as np
import pyopencl as cl

print("Hello GPU!")


# Define the data
a_np = np.random.rand(50000).astype(np.float32)
b_np = np.random.rand(50000).astype(np.float32)


# This makes the CPU talk to the GPU
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


m_f = cl.mem_flags # Passing the varuabkes to the GPU
a_g = cl.Buffer(ctx, m_f.READ_ONLY | m_f.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, m_f.READ_ONLY | m_f.COPY_HOST_PTR, hostbuf=b_np)


# Define kernel to do the sum, can be defined in another file and read in as string
prg = cl.Program(
    ctx,
    """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g, __global float *res_g
    ){
        int gid = get_global_id(0);
        res_g[gid] = a_g[gid] + b_g[gid];
    }
    """,
).build()

# Define empty array to store the result
res_g = cl.Buffer(ctx, m_f.WRITE_ONLY, b_np.nbytes)
knl = prg.sum
# Run the kernel
knl(queue, a_np.shape, None, a_g, b_g, res_g)  # Not sure why I pass None here

# Copy result back to cpu
res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

# Check the result
print(res_np - (a_np + b_np))
print(np.linalg.norm(res_np - (a_np + b_np)))
assert np.allclose(res_np, a_np + b_np)
