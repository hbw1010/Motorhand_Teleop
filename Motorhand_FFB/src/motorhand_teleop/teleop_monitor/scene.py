"""只渲染控制端的姿态；实时显示不调用闭链求解器或任何硬件API。"""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import mujoco
from .state import finite_number

ROOT = Path(__file__).resolve().parents[3]
GLOVE = ROOT.parent / 'assets/hand_motor_ve33/hand_motor_ve33_R'
WUJI = ROOT / 'vendor/wuji-retargeting/wuji_retargeting/wuji-description/hand2/hand2_beta1/body/mjcf/right.xml'
WUJI_JOINTS = ('r_thumb_cmc_flex','r_thumb_cmc_abd','r_thumb_mcp','r_thumb_ip',
    *(f'r_{finger}_{joint}' for finger in ('index_finger','middle_finger','ring_finger','pinky')
      for joint in ('mcp_flex','mcp_abd','pip','dip')))


def asset_spec(path):
    root = ET.parse(path).getroot()
    # 这里只做FK显示；不继承两模型各自的仿真步进设置。
    option=root.find('option')
    if option is not None:root.remove(option)
    compiler = root.find('compiler')
    meshdir = path.parent / compiler.attrib.pop('meshdir', '')
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', str((meshdir / mesh.get('file')).resolve()))
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))


def complete(values, count):
    return values is not None and len(values) == count and all(
        finite_number(x) for x in values)


class MonitorScene:
    def __init__(self):
        spec = mujoco.MjSpec.from_string('''<mujoco><visual><global offwidth="1280" offheight="720"/>
          <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/></visual>
          <worldbody><light pos="0 -1 2"/><geom type="plane" size="1 1 .01" rgba=".08 .11 .16 1" pos="0 0 -.015"/></worldbody></mujoco>''')
        for path, prefix, pos, quat in (
            (GLOVE/'hand_closed_chain.xml','glove_',[-.15,0,0],[.5,-.5,-.5,-.5]),
            (WUJI,'wuji_',[.15,0,0],[0,1,0,0])):
            child = asset_spec(path)
            for geom in child.geoms:
                geom.contype = geom.conaffinity = 0
                if prefix=='wuji_' and geom.group==2:
                    geom.rgba=[0,0,0,0]
            frame=spec.worldbody.add_frame(pos=pos,quat=quat)
            spec.attach(child,prefix=prefix,frame=frame)
        self.model=spec.compile()
        self.xml=spec.to_xml()
        closure=yaml.safe_load((GLOVE/'config/loop_closures.yaml').read_text())
        self.tree_names=closure['independent']+closure['dependent']
        self.glove_indices=np.array([int(self.model.joint('glove_'+n).qposadr[0]) for n in self.tree_names])
        self.wuji_indices=np.array([int(self.model.joint('wuji_'+n).qposadr[0]) for n in WUJI_JOINTS])
        self.last_q=self.model.qpos0.copy()
        self.camera=SimpleNamespace(lookat=np.array([0.,0.,.11]),azimuth=90.,elevation=-12.,distance=.43)

    def availability(self, sample):
        missing=[]
        if not complete(sample.get('glove_tree_rad'),68):missing.append('手套姿态缺失')
        if not complete(sample.get('wuji_position_rad'),20):missing.append('Wuji姿态缺失')
        return ' · '.join(missing)

    def pose(self, sample):
        q=self.last_q.copy()
        if complete(sample.get('glove_tree_rad'),68):q[self.glove_indices]=sample['glove_tree_rad']
        if complete(sample.get('wuji_position_rad'),20):q[self.wuji_indices]=sample['wuji_position_rad']
        self.last_q=q
        ghost=q.copy()
        if complete(sample.get('wuji_target_rad'),20):ghost[self.wuji_indices]=sample['wuji_target_rad']
        return q,ghost
