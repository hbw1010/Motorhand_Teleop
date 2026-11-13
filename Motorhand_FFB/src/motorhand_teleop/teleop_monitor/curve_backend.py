"""曲线路径后端：Qt版本匹配才启用C++，否则保留全点Python后备。"""
import ctypes
import math
from pathlib import Path
import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import QT_VERSION_STR, qVersion
from PyQt5.QtGui import QPainterPath

LIBRARY=Path(__file__).resolve().parents[3]/'.cache/teleop-monitor/build/libmonitor_curve.so'


class CurveBackend:
    def __init__(self,mode='auto',library=LIBRARY):
        self.name='Python';self.reason='';self.lib=None
        if mode not in ('auto','python','native'):raise ValueError('曲线后端必须为auto/python/native')
        if mode=='python':return
        try:
            lib=ctypes.CDLL(str(library))
            for name,expected in (('curve_qt_build_version',QT_VERSION_STR),('curve_qt_runtime_version',qVersion())):
                fn=getattr(lib,name);fn.argtypes=[];fn.restype=ctypes.c_char_p
                if fn().decode()!=expected:raise RuntimeError('C++与PyQt的Qt版本不一致')
            lib.build_curve.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]+[ctypes.c_double]*9
            lib.build_curve.restype=ctypes.c_int
            self.lib=lib;self.name='C++'
        except (OSError,AttributeError,RuntimeError) as exc:
            self.reason=str(exc)
            if mode=='native':raise RuntimeError('C++曲线后端不可用，请运行 pixi run build-monitor') from exc

    def path(self,times,values,breaks,start,end,area,low,high,scale):
        times=np.ascontiguousarray(times,dtype=np.uint64)
        values=np.ascontiguousarray(values,dtype=np.float64)
        breaks=np.ascontiguousarray(breaks,dtype=np.uint8)
        if times.ndim!=1 or values.shape!=times.shape or breaks.shape!=times.shape:
            raise ValueError('曲线时间、数值与断点维数必须一致')
        params=(float(start),float(end),area.left(),area.bottom(),area.width(),area.height(),low,high,scale)
        if not all(math.isfinite(v) for v in params) or end<=start or high<=low or scale<=0:
            raise ValueError('曲线绘制范围无效')
        result=QPainterPath()
        if not len(times):return result
        if self.lib is not None:
            status=self.lib.build_curve(int(sip.unwrapinstance(result)),times.ctypes.data,values.ctypes.data,
                                        breaks.ctypes.data,len(times),*params)
            if status:raise RuntimeError(f'C++曲线路径生成失败：{status}')
            return result
        # 与C++相同的完整采样点与断点规则，不做显示抽稀。
        valid=np.isfinite(values)&(times>=start)&(times<=end)
        moves=np.array(breaks,dtype=bool)
        moves[0]=True
        if len(times)>1:
            moves[1:] |= (~valid[:-1]) | (times[1:]<=times[:-1]) | (
                (times[1:]-times[:-1])>250_000_000)
        x=area.left()+(times[valid].astype(float)-start)/(end-start)*area.width()
        y=area.bottom()-(values[valid]/scale-low)/(high-low)*area.height()
        previous=False
        for xx,yy,move in zip(x,y,moves[valid]):
            if not math.isfinite(xx) or not math.isfinite(yy):previous=False;continue
            if move or not previous:result.moveTo(float(xx),float(yy))
            else:result.lineTo(float(xx),float(yy))
            previous=True
        return result
