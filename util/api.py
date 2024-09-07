from inspect import currentframe, getframeinfo

class ReturnObject:
    def __init__(self, ret: dict) -> None:
        self.dict = ret
    def __getattr__(self, __name: str):
        ret = self.dict.get(__name)
        if __name not in self.dict:
            print(f"{__name} not found in return object, returning None!")
        return ret
    def __contains__(self, key):
        return key in self.dict


def return_as_obj(func):
    "Make things cleaner by turning ret['key'] into ret.key"
    def inner(*args, **kwargs):
        ret = func(*args, **kwargs)
        return ReturnObject(ret)
    return inner


def yield_as_obj(iter_func):
    def inner(*args, **kwargs):
        for ret in iter_func(*args, **kwargs):
            yield ReturnObject(ret)
    return inner



