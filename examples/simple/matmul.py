import torch

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')  # GPU
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')   # CPU
        print("CUDA is not available. Using CPU.")

    # Define matrix dimensions
    rows_a, cols_a = 3, 2
    rows_b, cols_b = 2, 4

    # Create random matrices A and B
    A_cpu = torch.rand(rows_a, cols_a)
    B_cpu = torch.rand(rows_b, cols_b)

    # Move matrices to GPU
    A_gpu = A_cpu.to(device)
    B_gpu = B_cpu.to(device)

    # Perform matrix multiplication on GPU
    C_gpu = torch.matmul(A_gpu, B_gpu)

    # Move the result back to CPU
    C_cpu = C_gpu.to('cpu')

    # Print results
    print("Matrix A (CPU):")
    print(A_cpu)

    print("\nMatrix B (CPU):")
    print(B_cpu)

    print("\nMatrix C = A * B (Resultant Matrix on CPU):")
    print(C_cpu)

if __name__ == "__main__":
    main()
