TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda-9.1/bin/nvcc -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I /home/lab-tang.jiapeng/anaconda3/envs/py27/lib/python2.7/site-packages/tensorflow/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 && g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /home/lab-tang.jiapeng/anaconda3/envs/py27/lib/python2.7/site-packages/tensorflow/include -I /home/lab-tang.jiapeng/anaconda3/envs/py27/lib/python2.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.1/lib64/ -L $TF_LIB -ltensorflow_framework -O2
