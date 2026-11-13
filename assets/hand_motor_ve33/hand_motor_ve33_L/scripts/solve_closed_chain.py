#!/usr/bin/env python3
# FK adapted from jsk-ros-pkg/sw2robot, Apache-2.0; see THIRD_PARTY_LICENSE.
import numpy as np
import yaml
import xml.etree.ElementTree as ET
from scipy.optimize import least_squares

def _rpy_matrix(rpy):
    rx, ry, rz = rpy
    cx, cy, cz = (np.cos(rx), np.cos(ry), np.cos(rz))
    sx, sy, sz = (np.sin(rx), np.sin(ry), np.sin(rz))
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rzm @ rym @ rxm
    return m

class _FK:
    """Minimal URDF forward kinematics (revolute/continuous/prismatic/fixed)."""

    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.j = {}
        for j in root.findall('joint'):
            o = j.find('origin')
            xyz = [float(x) for x in o.get('xyz', '0 0 0').split()] if o is not None else [0, 0, 0]
            rpy = [float(x) for x in o.get('rpy', '0 0 0').split()] if o is not None else [0, 0, 0]
            ax = j.find('axis')
            axis = np.array([float(x) for x in ax.get('xyz').split()]) if ax is not None else np.array([0.0, 0.0, 1.0])
            t = _rpy_matrix(rpy)
            t[:3, 3] = xyz
            self.j[j.get('name')] = (j.get('type'), j.find('child').get('link'), j.find('parent').get('link'), t, axis)
        self.cj = {v[1]: n for n, v in self.j.items()}

    @staticmethod
    def _rot(a, q):
        a = a / (np.linalg.norm(a) or 1.0)
        x, y, z = a
        c, s = (np.cos(q), np.sin(q))
        cc = 1.0 - c
        return np.array([[c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s, 0], [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s, 0], [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc, 0], [0, 0, 0, 1.0]])

    def _tj(self, n, q):
        typ, _c, _p, t, axis = self.j[n]
        if typ in ('revolute', 'continuous'):
            return t @ self._rot(axis, q)
        if typ == 'prismatic':
            m = np.eye(4)
            m[:3, 3] = axis / (np.linalg.norm(axis) or 1.0) * q
            return t @ m
        return t

    def world(self, link, q):
        chain, ln = ([], link)
        while ln in self.cj:
            n = self.cj[ln]
            chain.append(n)
            ln = self.j[n][2]
        m = np.eye(4)
        for n in reversed(chain):
            m = m @ self._tj(n, q.get(n, 0.0))
        return m
class ClosureSolver:
    def __init__(self, urdf_file, config_file):
        self.fk=_FK(urdf_file)
        with open(config_file,encoding='utf8') as f: self.config=yaml.safe_load(f)
        self.independent=self.config['independent'];self.dependent=self.config['dependent']
        self.tolerance=self.config['closure_tolerance_m'];self.groups=self.config['groups']
        self.bounds={}
        for joint in ET.parse(urdf_file).getroot().findall('joint'):
            limit=joint.find('limit')
            if joint.get('type')!='continuous' and limit is not None:
                self.bounds[joint.get('name')]=(float(limit.get('lower','-inf')),float(limit.get('upper','inf')))
        self.witnesses=[]
        for c in self.config['closures']:
            a,b=c['link_a'],c['link_b'];p=np.array(c['point']);axis=np.array(c['axis']);axis/=np.linalg.norm(axis)
            for pos in (p,p+.03*axis):self.witnesses.append((a,np.linalg.inv(self.fk.world(a,{}))@np.r_[pos,1],b,np.linalg.inv(self.fk.world(b,{}))@np.r_[pos,1]))
        # Each finger is an independent constraint block. Precompute which
        # joint ancestors affect each closure endpoint, once per model.
        def ancestors(link):
            result=set()
            while link in self.fk.cj:
                n=self.fk.cj[link];result.add(n);link=self.fk.j[n][2]
            return result
        self._blocks={}
        for group,deps in self.groups.items():
            witnesses=[w for w in self.witnesses if set(deps)&(ancestors(w[0])|ancestors(w[2]))]
            masks=[np.array([[n in ancestors(w[side]) for n in deps] for w in witnesses]) for side in (0,2)]
            self._blocks[group]=(witnesses,*masks)
        self.reset()
    def reset(self):self.state={n:0.0 for n in self.independent+self.dependent}
    def residual(self,q):
        cache={}
        def pose(link):
            if link not in cache:
                name=self.fk.cj.get(link)
                cache[link]=np.eye(4) if name is None else pose(self.fk.j[name][2])@self.fk._tj(name,q.get(name,0.0))
            return cache[link]
        return np.concatenate([(pose(a)@pa-pose(b)@pb)[:3] for a,pa,b,pb in self.witnesses])
    def max_gap(self,q):return float(np.linalg.norm(self.residual(q).reshape(-1,3),axis=1).max())
    def _group_residual_jac(self,q,group):
        """Exact point-constraint Jacobian; only evaluate this finger's FK."""
        cache={}
        def pose(link):
            if link not in cache:
                name=self.fk.cj.get(link)
                cache[link]=np.eye(4) if name is None else pose(self.fk.j[name][2])@self.fk._tj(name,q.get(name,0.0))
            return cache[link]
        witnesses,mask_a,mask_b=self._blocks[group];deps=self.groups[group]
        a=np.array([(pose(la)@pa)[:3] for la,pa,lb,pb in witnesses])
        b=np.array([(pose(lb)@pb)[:3] for la,pa,lb,pb in witnesses])
        worlds=[pose(self.fk.j[n][1]) for n in deps]
        origins=np.array([w[:3,3] for w in worlds])
        axes=np.array([w[:3,:3]@(self.fk.j[n][4]/np.linalg.norm(self.fk.j[n][4])) for n,w in zip(deps,worlds)])
        da=np.cross(axes[None,:,:],a[:,None,:]-origins[None,:,:])
        db=np.cross(axes[None,:,:],b[:,None,:]-origins[None,:,:])
        for k,n in enumerate(deps):
            if self.fk.j[n][0]=='prismatic':da[:,k]=axes[k];db[:,k]=axes[k]
        jac=(da*mask_a[:,:,None]-db*mask_b[:,:,None]).transpose(0,2,1).reshape(-1,len(deps))
        return (a-b).ravel(),jac
    def solve(self,driven):
        if set(driven)-set(self.independent):raise ValueError('Only the 12 motor input joints may be controlled')
        if any(not np.isfinite(float(v)) for v in driven.values()):raise ValueError('Non-finite angle')
        state=dict(self.state)
        changed={n:float(v) for n,v in driven.items() if abs(float(v)-state[n])>1e-12}
        count=max(1,int(np.ceil(max((abs(v-state[n])/.02 for n,v in changed.items()),default=0))))
        if count>240:raise ValueError('Move in smaller increments')
        def advance(previous,targets,depth=0):
            candidate={**previous,**targets};jump=0.0
            for group in {n.split('_')[0] for n in changed}:
                deps=self.groups[group];initial=np.array([previous[n] for n in deps])
                lower=np.array([self.bounds.get(n,(-np.inf,np.inf))[0] for n in deps])
                upper=np.array([self.bounds.get(n,(-np.inf,np.inf))[1] for n in deps])
                last_x=None;last_result=None
                def evaluate(x):
                    nonlocal last_x,last_result
                    if last_x is None or not np.array_equal(x,last_x):
                        last_x=x.copy();last_result=self._group_residual_jac({**candidate,**dict(zip(deps,x))},group)
                    return last_result
                fit=least_squares(lambda x:evaluate(x)[0],np.clip(initial,lower,upper),jac=lambda x:evaluate(x)[1],bounds=(lower,upper),max_nfev=120,xtol=1e-12,ftol=1e-12,gtol=1e-12)
                jump=max(jump,float(np.max(np.abs(fit.x-initial),initial=0)))
                candidate.update(zip(deps,map(float,fit.x)))
            if self.max_gap(candidate)<=self.tolerance and jump<=.2:return candidate
            if depth>=8 or max(abs(v-previous[n]) for n,v in targets.items())<1e-5:
                raise ValueError('Target exceeds this closed-chain assembly branch; last valid pose retained')
            # A near-toggle Newton step may converge toward the other assembly
            # root. Retry from the same valid state with a shorter continuation
            # step; never accept a constraint violation or change branch.
            midpoint={n:(previous[n]+v)/2 for n,v in targets.items()}
            middle=advance(previous,midpoint,depth+1)
            return advance(middle,targets,depth+1)
        for step in range(1,count+1):
            targets={n:self.state[n]+(v-self.state[n])*step/count for n,v in changed.items()}
            state=advance(state,targets)
        self.state=state
        return dict(state)
