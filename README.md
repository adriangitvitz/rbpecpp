### Create Python bindings

```sh
cd pybind11_extension/
```

```sh
 c++ -O2 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) rbpe_tokenizer.cpp -o rbpe_tokenizer$(python3-config --extension-suffix) $(python3-config --embed --ldflags)
```
