"""显示用力矩换算与关节名称；不会修改硬件控制参数。"""
from dataclasses import dataclass
from pathlib import Path
import json
import math

FINGERS=('拇指','食指','中指','无名指','小指')
THUMB=('掌根屈伸','掌根外展','指根屈伸','指尖屈伸')
FINGER=('指根屈伸','指根侧摆','中节屈伸','末节屈伸')
DEFAULT_CONFIG=Path(__file__).resolve().parents[4]/'config/feedback/wuji-monitor-torque.json'


def finite(x):
    try:return type(x) in (int,float) and math.isfinite(x)
    except OverflowError:return False


def joint_label(hand,index):
    if hand=='wuji':return FINGERS[index//4]+'·'+(THUMB if index<4 else FINGER)[index%4]
    if hand=='glove_joint':
        return '拇指·'+THUMB[index] if index<4 else FINGERS[1+(index-4)//2]+'·'+('指根屈伸' if index%2==0 else '中节屈伸')
    if index<4:label='拇指·'+THUMB[index]
    else:label=FINGERS[1+(index-4)//2]+'·'+('指根屈伸' if index%2==0 else '中末节驱动')
    return f'J{index+1} {label}'


@dataclass
class TorqueConversion:
    glove: list
    wuji: list
    glove_source: str
    wuji_source: str
    glove_calibrated: bool = False
    glove_calibration_id: str = ''
    glove_bias_a: list | None = None

    def __post_init__(self):
        for values,count in ((self.glove,12),(self.wuji,20)):
            if len(values)!=count or any(x is not None and (not finite(x) or x==0) for x in values):
                raise ValueError('Kt必须为每轴非零有限Nm/A或null（未标定）')
        for source in (self.glove_source,self.wuji_source):
            if not isinstance(source,str) or not source.strip() or len(source)>512:raise ValueError('换算必须说明系数来源')
        if type(self.glove_calibrated) is not bool:
            raise ValueError('独立标定声明必须为布尔值')
        if not isinstance(self.glove_calibration_id,str) or len(self.glove_calibration_id)>512:
            raise ValueError('独立标定来源ID无效')
        if self.glove_calibrated and (not self.glove_calibration_id.strip() or all(k is None for k in self.glove)):
            raise ValueError('独立标定必须提供来源ID和至少一轴有效Kt')
        if self.glove_bias_a is not None and (not isinstance(self.glove_bias_a,list)
                or len(self.glove_bias_a)!=12 or any(not finite(v) for v in self.glove_bias_a)):
            raise ValueError('手套电流偏置必须为12轴有限A值')

    @classmethod
    def load(cls,path=DEFAULT_CONFIG):
        d=json.loads(Path(path).read_text())
        if d.get('schema_version')!=1:raise ValueError('未知力矩换算配置版本')
        return cls(d['glove_nm_per_a'],d['wuji_nm_per_a'],d['glove_source'],d['wuji_source'],
                   d.get('glove_calibrated',False),d.get('glove_calibration_id',''),d.get('glove_bias_a'))

    @classmethod
    def unavailable(cls):return cls([None]*12,[None]*20,'未提供','未提供')


def enrich(packet,config):
    p=dict(packet)
    for prefix,constants in (('glove',config.glove),('wuji',config.wuji)):
        currents=p.get(prefix+'_current_a') or []
        values=[]
        for i,k in enumerate(constants):
            v=currents[i] if i<len(currents) else None
            value=v*k if finite(v) and k is not None else None
            values.append(value if finite(value) else None)
        p[prefix+'_estimated_torque_nm']=values
    p['torque_conversion']={'glove_nm_per_a':list(config.glove),'wuji_nm_per_a':list(config.wuji),
                           'glove_source':config.glove_source,'wuji_source':config.wuji_source,
                           'glove_calibrated':config.glove_calibrated,
                           'glove_calibration_id':config.glove_calibration_id,
                           'glove_bias_a':list(config.glove_bias_a) if config.glove_bias_a is not None else None}
    # 空载参考仍保留在诊断字段，不能进入验证力矩或刚度分析。
    p['glove_validated_torque_nm']=[None]*12
    if config.glove_calibrated:
        currents=p.get('glove_current_a') or []
        for i,k in enumerate(config.glove):
            current=currents[i] if i<len(currents) else None
            bias=config.glove_bias_a[i] if config.glove_bias_a is not None else 0.
            value=(current-bias)*k if finite(current) and k is not None else None
            p['glove_validated_torque_nm'][i]=value if finite(value) else None
    matrix=p.get('wuji_to_glove_torque_matrix')
    p['wuji_reflected_torque_nm']=[None]*12
    paired=p.get('wuji_projection_current_a')
    if (matrix is not None and len(matrix)==240 and all(finite(x) for x in matrix)
            and paired is not None and len(paired)==20 and p.get('wuji_projection_received_ns')):
        # 负号表示向主手反射从手执行器负载，不代表测得净接触力。
        for row in range(12):
            used=[j for j in range(20) if matrix[row*20+j]!=0]
            if any(not finite(paired[j]) or config.wuji[j] is None for j in used):continue
            value=-sum(matrix[row*20+j]*paired[j]*config.wuji[j] for j in used)
            p['wuji_reflected_torque_nm'][row]=value if finite(value) else None
    def age(a,b):
        x,y=p.get(a),p.get(b)
        return (x-y)/1e6 if type(x) is int and type(y) is int and x>=y else None
    p['timing_ms']=[age('published_ns','wuji_received_ns'),age('published_ns','glove_received_ns'),
                    age('cycle_completed_ns','cycle_started_ns'),age('published_ns','glove_output_source_ns')]
    p['feedback_chain_delay_ms']=[age('glove_submit_ns','wuji_feedback_received_ns')
                                  if p.get('feedback_request_id') and p.get('glove_submit_ns')
                                  and p.get('wuji_feedback_received_ns') else None]
    return p
