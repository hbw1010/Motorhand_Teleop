import numpy as np
import pytest
from PyQt5.QtCore import QRectF
from motorhand_teleop.teleop_monitor.curve_backend import CurveBackend,LIBRARY


@pytest.mark.parametrize('count',[0,1,250,3000])
def test_native_path_keeps_all_points_and_matches_python(count):
    if not LIBRARY.exists():pytest.skip('先运行build-monitor')
    native=CurveBackend('native');python=CurveBackend('python')
    t=np.arange(count,dtype=np.uint64)*20_000_000+1_000_000_000
    v=np.sin(np.arange(count)*.37);breaks=np.zeros(count,np.uint8)
    if count>10:
        v[3]=np.nan;t[5]=t[4];t[7]=t[6]-1;breaks[9]=1
    args=(t,v,breaks,900_000_000,70_000_000_000,QRectF(0,0,700,300),-1.2,1.2,1.)
    a,b=native.path(*args),python.path(*args)
    assert a.elementCount()==b.elementCount()
    for i in range(a.elementCount()):
        x,y=a.elementAt(i),b.elementAt(i)
        assert x.type==y.type
        assert x.x==pytest.approx(y.x,abs=1e-8)
        assert x.y==pytest.approx(y.y,abs=1e-8)


def test_full_sample_count_is_not_decimated():
    if not LIBRARY.exists():pytest.skip('先运行build-monitor')
    t=np.arange(3000,dtype=np.uint64)*20_000_000+1_000_000_000
    v=np.arange(3000)%2
    path=CurveBackend('native').path(t,v,np.zeros(3000,np.uint8),0,70_000_000_000,QRectF(0,0,50,200),-1,2,1)
    assert path.elementCount()==3000


def test_unavailable_native_library_falls_back_without_dropping_data(tmp_path):
    backend=CurveBackend(library=tmp_path/'missing.so')
    assert backend.name=='Python' and backend.reason
