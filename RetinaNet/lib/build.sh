CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61"


# Build NMS
cd nms/src/cuda
echo "Compiling nms kernels by nvcc..."
/usr/local/cuda-9.0/bin/nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export CPATH=/usr/local/cuda-9.0/include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
cd ../../
python build.py
cd ../
