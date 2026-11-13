"""全点同图对比：数值缓存、原始采样点游标与框选缩放。"""
import math
import time
import numpy as np
from PyQt5.QtCore import Qt,QPointF,QRectF,QRect
from PyQt5.QtGui import QPainter,QColor,QPen,QFont,QPolygonF
from PyQt5.QtWidgets import QWidget,QRubberBand
from .quantities import finite
from .plots import axis_range, STATE_FIELDS


def font(pixels):
    result=QFont('Noto Sans CJK SC');result.setPixelSize(pixels);return result


class Curve(QWidget):
    def __init__(self,owner,spec):
        super().__init__();self.owner=owner;self.spec=spec;self.setMinimumHeight(330)
        self.hidden=set();self.legend_boxes=[];self.plot_area=None;self.y_view=None
        self.drag_start=None;self.drag_bounds=None;self.draw_series=[]
        self.rubber=QRubberBand(QRubberBand.Rectangle,self)

    def mousePressEvent(self,event):
        self.drag_start=None
        if event.button()!=Qt.LeftButton:return
        for i,box in enumerate(self.legend_boxes):
            if box.contains(event.pos()):
                if i in self.hidden:self.hidden.remove(i)
                else:self.hidden.add(i)
                self.owner.invalidate_view();return
        if self.plot_area is not None and self.plot_area.contains(event.pos()):
            self.owner.pause.setChecked(True)
            self.drag_start=event.pos()
            self.drag_bounds=(self.start_ns,self.end_ns,QRectF(self.plot_area),self.low,self.high,self.scale)

    def mouseMoveEvent(self,event):
        if self.drag_start is not None and event.buttons() & Qt.LeftButton:
            box=QRect(self.drag_start,event.pos()).normalized().intersected(self.drag_bounds[2].toRect())
            if (event.pos()-self.drag_start).manhattanLength()>5:
                self.rubber.setGeometry(box);self.rubber.show()

    def _sample_at(self,point,bounds):
        start,end,area,low,high,scale=bounds
        best=None
        for index,(trace,_) in enumerate(self.draw_series):
            if index in self.hidden:continue
            series=self.owner.state.series_for(trace)
            t,v=series.arrays()
            with np.errstate(over='ignore',invalid='ignore'):normalized=v/scale
            valid=np.isfinite(v)&(t>=start)&(t<=end)&(normalized>=low)&(normalized<=high)
            if not valid.any():continue
            ts=t[valid];vs=v[valid]
            x=area.left()+(ts.astype(float)-start)/(end-start)*area.width()
            y=area.bottom()-(vs/scale-low)/(high-low)*area.height()
            distance=(x-point.x())**2+(y-point.y())**2
            selected=int(np.argmin(distance));candidate=(float(distance[selected]),int(ts[selected]))
            if best is None or candidate[0]<best[0]:best=candidate
        return None if best is None else best[1]

    def mouseReleaseEvent(self,event):
        if self.drag_start is None:return
        start,end,area,low,high,scale=self.drag_bounds
        box=QRect(self.drag_start,event.pos()).normalized().intersected(area.toRect())
        dragged=(event.pos()-self.drag_start).manhattanLength()>5
        self.drag_start=None;self.rubber.hide()
        if dragged:
            if box.width()>=8:
                a=int(start+(box.left()-area.left())/area.width()*(end-start))
                b=int(start+(box.right()-area.left())/area.width()*(end-start))
                self.owner.set_time_range(a,b)
            if box.height()>=8:
                a=(low+(area.bottom()-box.bottom())/area.height()*(high-low))*scale
                b=(low+(area.bottom()-box.top())/area.height()*(high-low))*scale
                if finite(a) and finite(b) and a<b:self.y_view=(a,b)
        else:
            stamp=self._sample_at(event.pos(),self.drag_bounds)
            if stamp is not None:
                if event.modifiers() & Qt.ShiftModifier:self.owner.cursor_b_ns=stamp
                else:self.owner.cursor_ns=stamp
        self.owner.invalidate_view()

    def mouseDoubleClickEvent(self,event):
        if self.plot_area and self.plot_area.contains(event.pos()):self.owner.reset_zoom()
        else:self.owner.show_metric_info(self.spec.traces[0].field)

    def wheelEvent(self,event):
        if not self.plot_area or not self.plot_area.contains(event.pos()):return
        self.owner.pause.setChecked(True)
        factor=math.exp(max(-2.,min(2.,-event.angleDelta().y()/600.)))
        anchor=self.start_ns+(event.pos().x()-self.plot_area.left())/self.plot_area.width()*(self.end_ns-self.start_ns)
        self.owner.set_time_range(int(anchor+(self.start_ns-anchor)*factor),int(anchor+(self.end_ns-anchor)*factor))
        center=self.low+(self.plot_area.bottom()-event.pos().y())/self.plot_area.height()*(self.high-self.low)
        a=(center+(self.low-center)*factor)*self.scale;b=(center+(self.high-center)*factor)*self.scale
        if finite(a) and finite(b) and a<b:self.y_view=(a,b)
        self.owner.invalidate_view();event.accept()

    def paintEvent(self,event):
        self.owner.curve_rate.mark(key=self.owner.view_generation)
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing);p.fillRect(self.rect(),QColor('#141e2c'))
        p.setFont(font(20));p.setPen(QColor('#dce7f6'));p.drawText(16,29,self.spec.title)
        p.setFont(font(16));p.setPen(QColor('#adbed2'))
        note=p.boundingRect(QRectF(16,41,self.width()-32,110),Qt.TextWordWrap,self.spec.description)
        p.drawText(note,Qt.TextWordWrap,self.spec.description)
        traces=self.spec.traces;legend_top=note.bottom()+10
        columns=max(1,int((self.width()-32)/290));rows=math.ceil(len(traces)/columns)
        plot_top=legend_top+rows*28+14+len(self.spec.states)*22
        self.setMinimumHeight(max(330,int(plot_top+155)))
        area=QRectF(88,plot_top,max(50,self.width()-108),max(40,self.height()-plot_top-35))
        start,end=self.owner.time_range()
        self.draw_series=[(trace,self.owner.state.series_for(trace)) for trace in traces]
        chunks=[]
        for i,(_,series) in enumerate(self.draw_series):
            if i in self.hidden:continue
            t,v=series.arrays();chunks.append(v[(t>=start)&(t<=end)&np.isfinite(v)])
        values=np.concatenate(chunks) if chunks else np.empty(0)
        if self.y_view:
            scale=max(abs(self.y_view[0]),abs(self.y_view[1])) or 1.
            low,high=self.y_view[0]/scale,self.y_view[1]/scale
            if low>=high:self.y_view=None;low,high,scale=axis_range(values)
        else:low,high,scale=axis_range(values)
        self.plot_area=area;self.start_ns=start;self.end_ns=end;self.legend_boxes=[]
        self.low,self.high,self.scale=low,high,scale
        # 软件状态条按真实状态更新时间持有，丢帧/未知值处断开，不假定物理接触。
        for row,trace in enumerate(self.spec.states):
            series=self.owner.state.series_for(trace);ts,vs=series.arrays();gaps=series.breaks().astype(bool)
            y=legend_top+rows*28+row*22+3
            p.setFont(font(14));p.setPen(QColor('#adbed2'));p.drawText(QRectF(12,y,74,18),Qt.AlignVCenter,trace.label)
            p.fillRect(QRectF(area.left(),y,area.width(),16),QColor('#223043'))
            if len(ts):
                split=np.ones(len(ts),bool)
                if len(ts)>1:
                    split[1:]=gaps[1:]|(~np.isfinite(vs[1:]))|(~np.isfinite(vs[:-1]))|(vs[1:]!=vs[:-1])|(ts[1:]<=ts[:-1])|((ts[1:]-ts[:-1])>100_000_000)
                edges=np.r_[np.flatnonzero(split),len(ts)]
                for first,last in zip(edges[:-1],edges[1:]):
                    if not finite(float(vs[first])):continue
                    right=int(ts[last-1])
                    if last<len(ts) and not gaps[last] and np.isfinite(vs[last]) and 0<int(ts[last])-right<=100_000_000:right=int(ts[last])
                    left=int(ts[first])
                    if right<start or left>end:continue
                    x0=area.left()+(max(start,left)-start)/(end-start)*area.width()
                    x1=area.left()+(min(end,right)-start)/(end-start)*area.width()
                    color=QColor(trace.color if vs[first] else '#3b526b');color.setAlpha(170 if vs[first] else 110)
                    p.fillRect(QRectF(x0,y,max(2,x1-x0),16),color)
            p.setFont(font(16))
        for j in range(5):
            y=area.bottom()-j*area.height()/4;p.setPen(QPen(QColor('#263448'),1))
            p.drawLine(QPointF(area.left(),y),QPointF(area.right(),y));p.setPen(QColor('#91a5bd'))
            value=low+(high-low)*j/4;value=0 if abs(value)<(high-low)*1e-5 else value
            text=f'{value*scale:.3g}' if finite(value*scale) else f'{value:.2g}×{scale:.1g}'
            p.drawText(QRectF(0,y-12,80,24),Qt.AlignRight|Qt.AlignVCenter,text)
        for j in range(3):
            x=area.left()+j*area.width()/2
            p.drawText(QRectF(min(self.width()-74,max(0,x-30)),area.bottom()+4,72,25),Qt.AlignCenter,f'{-(1-j/2)*(end-start)/1e9:g}s')
        cursor=self.owner.cursor_b_ns if self.owner.cursor_b_ns is not None else self.owner.cursor_ns
        for index,(trace,series) in enumerate(self.draw_series):
            color=QColor(trace.color);p.setPen(color)
            t,v=series.arrays();value=None
            if len(t):
                selected=int(np.argmin(np.abs(t.astype(float)-cursor))) if cursor is not None else len(t)-1
                if cursor is None or abs(int(t[selected])-cursor)<=250_000_000:value=float(v[selected])
            missing='未标定/无数据' if trace.field.endswith(('estimated_torque_nm','reflected_torque_nm','validated_torque_nm')) else '未提供'
            label=trace.display_label(self.owner.state.display_frame)+' '+(f'{value:.3g}' if finite(value) else missing)
            box=QRectF(16+(index%columns)*(self.width()-32)/columns,legend_top+(index//columns)*28,(self.width()-32)/columns,26)
            self.legend_boxes.append(box)
            if index in self.hidden:p.setPen(QColor('#58687c'))
            p.drawText(box,Qt.AlignLeft|Qt.AlignVCenter,label)
            if index in self.hidden:continue
            # 布尔事件在实际更新时间阶跃，绝不用斜线暗示提前改变状态。
            draw_t,draw_v,draw_breaks=t,v,series.breaks()
            if trace.field in STATE_FIELDS and len(t):
                draw_t=np.repeat(t,2)[1:];draw_v=np.repeat(v,2)[:-1];draw_breaks=np.repeat(draw_breaks,2)[1:]
            try:path=self.owner.curve_backend.path(draw_t,draw_v,draw_breaks,start,end,area,low,high,scale)
            except (ValueError,RuntimeError) as exc:
                self.owner.notice.setText(f'曲线绘制错误：{exc}');continue
            p.save();p.setClipRect(area);p.setRenderHint(QPainter.Antialiasing,self.owner.smooth.isChecked())
            p.setPen(QPen(color,1.8,Qt.PenStyle(trace.style)));p.drawPath(path)
            with np.errstate(over='ignore',invalid='ignore'):normalized=v/scale
            valid=np.isfinite(v)&(t>=start)&(t<=end)&(normalized>=low)&(normalized<=high)
            if 0<np.count_nonzero(valid)<=area.width()/8:
                x=area.left()+(t[valid].astype(float)-start)/(end-start)*area.width()
                y=area.bottom()-(v[valid]/scale-low)/(high-low)*area.height()
                p.setPen(QPen(color,4));p.drawPoints(QPolygonF([QPointF(float(a),float(b)) for a,b in zip(x,y)]))
            p.restore()
        if not len(values):
            p.setPen(QColor('#91a5bd'));p.drawText(area,Qt.AlignCenter,'当前范围无有效采样 / 系数未提供')
        for name,stamp,color in (('A',self.owner.cursor_ns,'#54d6cf'),('B',self.owner.cursor_b_ns,'#f4b45a')):
            if stamp is not None and start<=stamp<=end:
                x=area.left()+(stamp-start)/(end-start)*area.width();p.setPen(QPen(QColor(color),1,Qt.DashLine))
                p.drawLine(QPointF(x,area.top()),QPointF(x,area.bottom()))
                p.drawText(QRectF(min(x+4,area.right()-160),area.top(),160,24),Qt.AlignLeft,f'{name} {(stamp-end)/1e6:.1f} ms')
