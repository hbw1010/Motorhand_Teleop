"""选区分析工作线程与全采样散点视图；统计不回写控制。"""
from concurrent.futures import ThreadPoolExecutor
import time
import math
import numpy as np
from PyQt5.QtCore import Qt,QPointF,QRectF,QRect
from PyQt5.QtGui import QPainter,QColor,QPen,QFont,QPolygonF
from PyQt5.QtWidgets import QWidget,QRubberBand
from .validation_analysis import analyze_axis
from .quantities import finite
from .plots import axis_range

PHASES={'press':('按入','#54d6cf'),'release':('回开','#f4b45a'),
        'hold':('低速保持','#aab8ca'),'unknown':('方向未知','#b19be9')}


class AnalysisBroker:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='monitor-analysis')
        self.future=None;self.pending_key=None;self.result=None;self.result_key=None;self.last_started=0.
        self.error=''
    def get(self,frames,index,bounds,base,revision,force=False):
        if self.future is not None and self.future.done():
            try:self.result=self.future.result();self.result_key=self.pending_key;self.error=''
            except Exception as exc:self.result=None;self.result_key=None;self.error=str(exc)
            self.future=None
        key=(base,revision)
        now=time.monotonic()
        needs=key!=self.result_key
        if self.future is None and needs and (force or now-self.last_started>=.2):
            self.pending_key=key;self.last_started=now
            self.future=self.pool.submit(analyze_axis,list(frames),index,bounds)
        if self.result_key and self.result_key[0]==base:
            if not force or self.result_key[1]==revision:return self.result
        return None
    def close(self):self.pool.shutdown(wait=False,cancel_futures=True)


def number(value,unit=''):
    return f'{value:.4g}{unit}' if finite(value) else '不可用'


def analysis_summary(result,axis_label):
    if result is None:return axis_label+'：分析准备中；缺少新版遥测时不会补造结果。'
    fit=result['fit'];stats=result['stability'];error=result['torque_error']
    text=f"{axis_label} | 有效配对 {len(result['points'])} 点 | 响应单位 {result['response_unit']}"
    text+=' | '+(f"斜率 {fit['slope']:.4g} {result['slope_unit']}，R²={fit['r_squared']:.3g}，n={fit['count']}" if fit else '拟合不可用（查看筛选条件/样本跨度）')
    text+=f"\n保持：角漂移 {number(stats.get('angle_drift_rad_s'),' rad/s')}；电流峰峰值 {number(stats.get('current_peak_to_peak_a'),' A')}；电流RMS {number(stats.get('current_rms_a'),' A')}"
    text+='\n力矩误差：'+(f"偏差 {error['mean_nm']:.4g} Nm；RMSE {error['rmse_nm']:.4g} Nm；峰值 {error['peak_abs_nm']:.4g} Nm" if error else '无完整独立标定与有效匹配，不评价Nm准确性')
    counts=result['counts']
    text+=f"\n筛选：低速≤0.05 rad/s、Kp>0、无释放/限幅；有效角跨度 {result['angle_span_rad']:.4g} rad；限幅 {counts.get('limited',0)} 点。悬停查看完整规则。"
    return text


class ScatterCurve(QWidget):
    def __init__(self,owner,spec):
        super().__init__();self.owner=owner;self.spec=spec;self.setMinimumHeight(370)
        self.x_view=self.y_view=None;self.area=None;self.bounds=None;self.drag=None;self.rubber=QRubberBand(QRubberBand.Rectangle,self)
    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton and self.area and self.bounds and self.area.contains(e.pos()):
            self.owner.pause.setChecked(True);self.drag=e.pos();self.old_bounds=self.bounds
    def mouseMoveEvent(self,e):
        if self.drag is not None:
            self.rubber.setGeometry(QRect(self.drag,e.pos()).normalized().intersected(self.area.toRect()));self.rubber.show()
    def mouseReleaseEvent(self,e):
        if self.drag is None:return
        box=QRect(self.drag,e.pos()).normalized().intersected(self.area.toRect());self.drag=None;self.rubber.hide()
        xl,xh,xs,yl,yh,ys=self.old_bounds
        if box.width()>8 and box.height()>8:
            a=(xl+(box.left()-self.area.left())/self.area.width()*(xh-xl))*xs
            b=(xl+(box.right()-self.area.left())/self.area.width()*(xh-xl))*xs
            c=(yl+(self.area.bottom()-box.bottom())/self.area.height()*(yh-yl))*ys
            d=(yl+(self.area.bottom()-box.top())/self.area.height()*(yh-yl))*ys
            if all(finite(x) for x in (a,b,c,d)) and a<b and c<d:self.x_view=(a,b);self.y_view=(c,d)
        self.update()
    def mouseDoubleClickEvent(self,e):self.x_view=self.y_view=None;self.update()
    def wheelEvent(self,e):
        if not self.area or not self.bounds or not self.area.contains(e.pos()):return
        self.owner.pause.setChecked(True);xl,xh,xs,yl,yh,ys=self.bounds
        factor=math.exp(max(-2.,min(2.,-e.angleDelta().y()/600.)))
        x=xl+(e.pos().x()-self.area.left())/self.area.width()*(xh-xl)
        y=yl+(self.area.bottom()-e.pos().y())/self.area.height()*(yh-yl)
        values=((x+(xl-x)*factor)*xs,(x+(xh-x)*factor)*xs,(y+(yl-y)*factor)*ys,(y+(yh-y)*factor)*ys)
        if all(finite(v) for v in values) and values[0]<values[1] and values[2]<values[3]:self.x_view=values[:2];self.y_view=values[2:]
        self.update()
    def paintEvent(self,event):
        self.owner.curve_rate.mark(key=self.owner.view_generation)
        p=QPainter(self);p.fillRect(self.rect(),QColor('#141e2c'));font=QFont('Noto Sans CJK SC');font.setPixelSize(19);p.setFont(font)
        p.setPen(QColor('#dce7f6'));p.drawText(16,28,self.spec.title)
        font.setPixelSize(15);p.setFont(font);p.setPen(QColor('#adbed2'))
        note=p.boundingRect(QRectF(16,40,self.width()-32,90),Qt.TextWordWrap,self.spec.description)
        p.drawText(note,Qt.TextWordWrap,self.spec.description)
        result=self.owner.current_analysis();self.bounds=None;self.area=QRectF(80,note.bottom()+55,max(60,self.width()-100),max(100,self.height()-note.bottom()-105))
        if result is None:
            p.drawText(self.area,Qt.AlignCenter,'正在配对分析 / 等待新版遥测');return
        nominal=self.spec.variant=='nominal';unit='Nm（名义）' if nominal else result['response_unit']
        points=[]
        for point in result['points']:
            y=(-point['blocking_direction']*point['nominal_nm']) if nominal and finite(point.get('nominal_nm')) else None if nominal else point['response']
            if finite(point['angle_excess_rad']) and finite(y):points.append((point['angle_excess_rad'],y,point))
        for j,(phase,(label,color)) in enumerate(PHASES.items()):
            p.setPen(QColor(color));p.drawText(16+j*max(100,int((self.width()-32)/4)),int(note.bottom()+27),label)
        if not points:p.setPen(QColor('#aab8ca'));p.drawText(self.area,Qt.AlignCenter,'无有效因果配对点；不补值、不猜测刚度');return
        x=np.array([v[0] for v in points]);y=np.array([v[1] for v in points])
        xl,xh,xs=axis_range(x);yl,yh,ys=axis_range(y)
        if self.x_view:xs=max(abs(v) for v in self.x_view) or 1.;xl,xh=(v/xs for v in self.x_view)
        if self.y_view:ys=max(abs(v) for v in self.y_view) or 1.;yl,yh=(v/ys for v in self.y_view)
        if xl>=xh:self.x_view=None;xl,xh,xs=axis_range(x)
        if yl>=yh:self.y_view=None;yl,yh,ys=axis_range(y)
        self.bounds=(xl,xh,xs,yl,yh,ys)
        for j in range(5):
            xx=self.area.left()+j/4*self.area.width();yy=self.area.bottom()-j/4*self.area.height()
            p.setPen(QPen(QColor('#2b3c51'),1));p.drawLine(QPointF(xx,self.area.top()),QPointF(xx,self.area.bottom()));p.drawLine(QPointF(self.area.left(),yy),QPointF(self.area.right(),yy))
            p.setPen(QColor('#aab8ca'));p.drawText(QRectF(xx-45,self.area.bottom()+3,90,20),Qt.AlignCenter,f'{(xl+(xh-xl)*j/4)*xs:.3g}')
            p.drawText(QRectF(0,yy-10,72,22),Qt.AlignRight,f'{(yl+(yh-yl)*j/4)*ys:.3g}')
        p.drawText(QRectF(self.area.left(),self.height()-26,self.area.width(),22),Qt.AlignCenter,f'按入方向超过阻挡参考角 · rad   /   抵抗响应 · {unit}')
        p.save();p.setClipRect(self.area)
        for phase,(_,color) in PHASES.items():
            pts=[]
            for xx,yy,point in points:
                if point['phase']!=phase:continue
                xn=xx/xs;yn=yy/ys
                if not xl<=xn<=xh or not yl<=yn<=yh:continue
                pts.append(QPointF(self.area.left()+(xn-xl)/(xh-xl)*self.area.width(),self.area.bottom()-(yn-yl)/(yh-yl)*self.area.height()))
            p.setPen(QPen(QColor(color),4));p.drawPoints(QPolygonF(pts))
        # 限幅点保留在图中，橙圈提示不会进入刚度拟合。
        p.setPen(QPen(QColor('#ff7766'),1.2));p.setBrush(Qt.NoBrush)
        for xx,yy,point in points:
            xn=xx/xs;yn=yy/ys
            if point['limited'] and xl<=xn<=xh and yl<=yn<=yh:
                p.drawEllipse(QPointF(self.area.left()+(xn-xl)/(xh-xl)*self.area.width(),self.area.bottom()-(yn-yl)/(yh-yl)*self.area.height()),4,4)
        if result['fit'] and not nominal:
            fit=result['fit'];a=xl*xs;b=xh*xs
            ya=(fit['slope']*a+fit['intercept'])/ys;yb=(fit['slope']*b+fit['intercept'])/ys
            if finite(ya) and finite(yb):
                p.setPen(QPen(QColor('#f1e7b0'),1.5,Qt.DashLine));p.drawLine(QPointF(self.area.left(),self.area.bottom()-(ya-yl)/(yh-yl)*self.area.height()),QPointF(self.area.right(),self.area.bottom()-(yb-yl)/(yh-yl)*self.area.height()))
        p.restore()
