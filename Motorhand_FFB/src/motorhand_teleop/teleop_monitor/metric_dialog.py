"""面向操作者的指标释义，公式和时间来源来自逐项代码核对。"""
from html import escape
from PyQt5.QtWidgets import QDialog,QHBoxLayout,QListWidget,QTextBrowser,QSplitter
from PyQt5.QtCore import Qt
from .metric_info import METRICS, metric_details


class MetricDialog(QDialog):
    def __init__(self,parent=None,field=None,frame=None):
        super().__init__(parent);self.frame=frame;self.setWindowTitle('指标说明 · 数值含义与时间来源');self.resize(1150,800)
        self.setStyleSheet('QWidget{background:#111c2a;color:#dce7f6;font-size:18px;} QListWidget{border:1px solid #334b66;}')
        layout=QHBoxLayout(self);split=QSplitter(Qt.Horizontal);layout.addWidget(split)
        self.names=QListWidget();self.names.setWordWrap(True);self.text=QTextBrowser()
        split.addWidget(self.names);split.addWidget(self.text);split.setSizes([340,800])
        from .plots import build_pages
        preferred=[]
        for specs in build_pages(1).values():
            for spec in specs:
                for trace in spec.traces+spec.states:
                    if trace.field in METRICS and trace.field not in preferred:preferred.append(trace.field)
        self.keys=preferred+[key for key in METRICS if key not in preferred]
        for key in self.keys:self.names.addItem(metric_details(key,frame)['name'])
        self.names.currentRowChanged.connect(self.show_metric)
        self.names.setCurrentRow(self.keys.index(field) if field in self.keys else 0)

    def show_metric(self,row):
        if not 0<=row<len(self.keys):return
        info=metric_details(self.keys[row],self.frame)
        labels=(('meaning','是什么'),('formula','如何得到'),('coordinates','坐标与方向'),
                ('time_source','时间戳是什么'),('limitations','不能据此推断什么'))
        html=f"<h2>{escape(info['name'])}</h2><p>单位：{escape(info['unit'])}　类型：{escape(info['kind'])}</p>"
        for key,title in labels:html+=f'<h3>{title}</h3><p>{escape(info[key])}</p>'
        html+='<p>曲线连线用于显示离散采样点，不生成新的实测数据。A/B游标吸附原始采样点；时间差仍受采样率和主机时间戳含义限制。</p>'
        self.text.setHtml(html)
