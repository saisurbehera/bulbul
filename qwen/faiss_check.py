import faiss
import time 
import numpy as np

# First, check if GPU is available
gpu_available = faiss.get_num_gpus()
print(f"Number of GPUs available: {gpu_available}")

# Create some sample data
dimension = 64  # vector dimension
nb = 100    # database size
nq = 10     # number of queries
np.random.seed(1234)
xb = np.random.random((nb, dimension)).astype('float32')
xq = np.random.random((nq, dimension)).astype('float32')

# Create an index
index_cpu = faiss.IndexFlatL2(dimension)  # First create CPU index
print("created index on CPU") 
# Move index to GPU
# If you have multiple GPUs, you can specify which one to use: gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
res = faiss.StandardGpuResources()  
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
print("moved index to GPU")
# Add vectors to the index
gpu_index.add(xb)

# Search
k = 5  # we want to see 5 nearest neighbors
start_time = time.time()
D, I = gpu_index.search(xq, k)
end_time = time.time()

print(f"Search time: {end_time - start_time:.2f} seconds")
print("\nFirst 5 distances:", D[0])
print("First 5 indices:", I[0])