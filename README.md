## RBPECPP

### Create Python bindings

```sh
cd pybind11_extension/
```

```sh
 c++ -O2 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) rbpe_tokenizer.cpp -o rbpe_tokenizer$(python3-config --extension-suffix) $(python3-config --embed --ldflags)
```
![Screenshot 2025-04-28 at 12 18 29 a m](https://github.com/user-attachments/assets/52525da9-3de1-483b-b0e9-a58c5f19113a)
