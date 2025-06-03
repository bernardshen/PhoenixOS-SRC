# Define the compiler and flags
NVCC = nvcc
CFLAGS = -std=c++11

# Define the executable name and source files
TARGET = matmul
SRCS = matmul.cu

# Define the rule to build the executable
all: $(TARGET)

# Compile the CUDA source files into an executable
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $@ $^ -g

# Clean up generated files
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
