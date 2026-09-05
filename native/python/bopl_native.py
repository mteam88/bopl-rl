"""Dependency-free Python access to the versioned Bopl C ABI."""
import ctypes as C
import json
from pathlib import Path
import sys

class Action(C.Structure):
    _fields_=[('angle',C.c_uint8),('buttons',C.c_uint8)]
class Observation(C.Structure):
    _fields_=[(name,C.c_int64) for name in ('x','y','vx','vy','rotation','local','grounded_speed','cooldown')]+[
        ('id',C.c_uint32),('ground_id',C.c_uint32)]+[(name,C.c_uint8) for name in ('alive','grounded','dashing','reserved')]

class World:
    """Owns one isolated world. Close it explicitly or use a with statement."""
    def __init__(self,library=None):
        suffix='.dylib' if sys.platform=='darwin' else '.so'
        library=library or Path(__file__).resolve().parents[1]/('target/release/libbopl_sim'+suffix)
        self._lib=C.CDLL(str(library));self._handle=None
        lib=self._lib
        signatures={
            'bopl_abi_version':([],C.c_uint32),
            'bopl_create':([],C.c_void_p),
            'bopl_destroy':([C.c_void_p],None),
            'bopl_reset':([C.c_void_p],C.c_int32),
            'bopl_step':([C.c_void_p,C.POINTER(Action),C.c_size_t],C.c_int32),
            'bopl_observe':([C.c_void_p,C.POINTER(Observation),C.c_size_t],C.c_int32),
            'bopl_terminal':([C.c_void_p],C.c_int32),
            'bopl_state_json':([C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)],C.c_int32),
        }
        for name,(args,result) in signatures.items():
            fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
        if lib.bopl_abi_version()!=1:raise RuntimeError('incompatible Bopl ABI')
        self._handle=lib.bopl_create()
        if not self._handle:raise RuntimeError('could not create Bopl world')
    def _live(self):
        if not self._handle:raise RuntimeError('world is closed')
        return self._handle
    @staticmethod
    def _check(status):
        if status:raise RuntimeError({1:'invalid argument',2:'unsupported action',3:'episode ended; reset required',4:'buffer too small',5:'simulation error'}[status])
    def reset(self):self._check(self._lib.bopl_reset(self._live()))
    def step(self,actions):
        if len(actions)!=2:raise ValueError('exactly two actions are required')
        for angle,buttons in actions:
            if not isinstance(angle,int) or not 0<=angle<=255 or not isinstance(buttons,int) or not 0<=buttons<=3:
                raise ValueError('angle must be 0..255 and buttons 0..3')
        inputs=(Action*2)(*(Action(*a) for a in actions))
        self._check(self._lib.bopl_step(self._live(),inputs,2))
    def observe(self):
        values=(Observation*2)();self._check(self._lib.bopl_observe(self._live(),values,2))
        return [{name:getattr(value,name) for name,_ in Observation._fields_ if name!='reserved'} for value in values]
    @property
    def terminal(self):return bool(self._lib.bopl_terminal(self._live()))
    def state(self):
        length=C.c_size_t()
        status=self._lib.bopl_state_json(self._live(),None,0,C.byref(length))
        if status!=4:self._check(status)
        output=C.create_string_buffer(length.value)
        self._check(self._lib.bopl_state_json(self._live(),output,len(output),C.byref(length)))
        return json.loads(output.value)
    def close(self):
        if getattr(self,'_handle',None):self._lib.bopl_destroy(self._handle);self._handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):self.close()
