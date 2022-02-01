import importlib.util
import sys
from types import ModuleType


def lazy_import(name):
    spec = importlib.util.find_spec(name)
    if spec is not None:
        spec_loader = spec.loader
        assert spec_loader is not None
        loader = importlib.util.LazyLoader(spec_loader)
        spec.loader = loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        return module
    return ModuleType("not found", None)
