NVCC        := nvcc
NVCCFLAGS   := -O3 -arch=sm_61 --use_fast_math -std=c++17
TARGET      := nbody-gpu

.PHONY: all clean

all: $(TARGET)

$(TARGET): nbody-gpu.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $

test: $(TARGET)
	./$(TARGET) 1024 0.01 500 100

solar: $(TARGET)
	./$(TARGET) planet 3600 8760 876

clean:
	rm -f $(TARGET) log.tsv
EOF