# 2048 dqn native

current performance is worse than the python one.

i'm still playing with those hyperparameters.

## Compile

install tensorflow c++ libraries and headers first.

and

```sh
gcc 2048.c -c -o 2048.o
g++ dqnagent.cpp 2048.o -ltensorflow_framework -ltensorflow_cc -lpthread
```
