## RBPECPP

### Create Python bindings

```sh
cd pybind11_extension/
```

```sh
 c++ -O2 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) rbpe_tokenizer.cpp -o rbpe_tokenizer$(python3-config --extension-suffix) $(python3-config --embed --ldflags)
```

![Screenshot 2025-05-03 at 4 05 13 a m](https://github.com/user-attachments/assets/f0e9594d-0c6b-4e95-b3cd-b7b13cf92d82)
