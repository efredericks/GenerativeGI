import random
import os, tempfile
from typing import *
from typeguard import check_type
import inspect

import techniques 
from PIL import Image, ImageDraw


Outcome = str
class Runner:
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"

    def __init__(self) -> None:
        pass

    def run(self, inp: str) -> Any:
        return (inp, Runner.UNRESOLVED)

# Runner program to pull what functions/parameters are available in the techniques module 
# The fuzzer will be responsible for generating the list of calls, their instantiation, and any additional parameters
# we don't want to hard-code
class TechniqueRunner(Runner):
    def __init__(self, module, dim, seed) -> None:
        self.module = module
        self.module_name = module.__name__
        self.dim = dim
        self.background = (0,0,0)
        self.seed = seed
        self.rng = random.Random(seed)

        self.functions = {}

        local_functions = [(name, func) for name, func in inspect.getmembers(self.module, inspect.isfunction) if func.__module__ == self.module_name]
        for _name, _func in local_functions:
            sig = inspect.signature(_func)
            self.functions[_name] = {
                'function': _func, 
                'signature': sig
            }

        self.getParameterList()


    def getParameterList(self) -> None:
        for f in self.functions:
            print(f"Function {f}")
            for pn, p in self.functions[f]['signature'].parameters.items():
                if p.rannotation != inspect._empty:
                     print(f"Param {pn}, Type {p.annotation}")
            print('---')

    def run(self, inp: str) -> Any:
        image = Image.new("RGBA", (self.dim[0], self.dim[1]), self.background)
        print(inp)
        return (image, Runner.UNRESOLVED)
    
## --

class Fuzzer:
    def __init__(self) -> None:
        pass

    def fuzz(self) -> str:
        return ""

    def run(self, runner: Runner = Runner()) -> Outcome:
        return runner.run(self.fuzz())

    def runs(self, runner: Runner = Runner(), trials: int = 10) -> List[Outcome]:
        return [self.run(runner) for i in range(trials)]

class TechniqueFuzzer:
    def __init__(self, min_calls: int = 1, max_calls: int=100, functions: List=[Any], rng: random.Random=random.Random()) -> None:
        self.min_calls = min_calls
        self.max_calls = max_calls
        self.functions = functions
        self.rng = rng

    def fuzz_int(self, min=-500, max=500, signed=False)-> int:
        if signed:
            max += min
            min = 0
        return random.randint(min, max)

    def fuzz_float(self, min=-20.0, max=20.0) -> float:
        return random.randrange(min, max)

    def fuzz_color(self) -> Tuple[int, int, int]:
        return (self.fuzz_int(0, 255), self.fuzz_int(0, 255), self.fuzz_int(0, 255))

    def fuzz_palette(self, num_colors=5) -> List[Tuple[int, int, int]]:
        palette = [self.fuzz_color() for i in range(num_colors)]
        return palette

    # Grab a random value from a pre-determined set of choices
    def fuzz_list(self, val: List[Any]) -> Any:
        return random.choice(val)

    def fuzz_parameters(self, signature: Dict) -> List[Any]:
        # for parameter_name, parameter in signature.parameters.items():
        #     if parameter.annotation != inspect._empty:
        #         print(f"Param {parameter_name}, Type {parameter.annotation}")
        return []

    # Construct an ordered list of functions to call and fuzzed parameters based on their signature
    def fuzz(self) -> List[Any]:
        fn_list = []
        listed_fns = list(self.functions.keys())
        for i in range(self.min_calls, self.max_calls+1):
            fn = random.choice(listed_fns)
            signature = self.functions[fn]['signature']
            params = self.fuzz_parameters(signature)
            fn_list.append(lambda: fn(*params))

        return fn_list

## --

if __name__ == "__main__":
    DIM = (1000,1000)
    seed = 1

    tr = TechniqueRunner(techniques, DIM, seed)
    img = tr.run("")

    technique_fuzzer = TechniqueFuzzer(min_calls=1, max_calls=100, functions=tr.functions, rng=tr.rng)
    # for i in range(50):
    #     print(technique_fuzzer.fuzz())