"""python -m motorhand_teleop.teleop_monitor [--demo]。只读，不启动硬件。"""
import argparse
import json
from pathlib import Path
import socket
import sys
import time


def main(argv=None):
    parser=argparse.ArgumentParser(description='独立双手3D与力反馈监视器；不连接或控制硬件')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--torque-config',type=Path,help='仅显示用的两侧电流-力矩参考系数JSON')
    parser.add_argument('--demo',action='store_true',help='离线合成演示，不接设备')
    parser.add_argument('--fps',type=int,default=60,help='界面目标刷新频率，10至120，实际FPS在窗口显示')
    parser.add_argument('--curve-backend',choices=('auto','native','python'),default='auto')
    parser.add_argument('--smooth-curves',action='store_true',help='开启曲线抗锯齿，数据仍保持全部原始点')
    parser.add_argument('--duration',type=float,default=0,help='界面自动退出秒数，0持续运行')
    parser.add_argument('--screenshot',type=Path,help='自动退出前保存窗口截图（需duration>0）')
    args=parser.parse_args(argv)
    if not 1<=args.port<=65535 or not 0<=args.duration<86400:parser.error('端口或时长无效')
    if not 10<=args.fps<=120:parser.error('fps必须为10至120')
    if args.screenshot and args.duration<=0:parser.error('截图需指定 --duration')
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    from .state import decode_packet
    from .window import MonitorWindow
    app=QApplication.instance() or QApplication(sys.argv[:1])
    receiver=None
    if not args.demo:
        try:
            receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);receiver.setblocking(False)
            receiver.bind(('127.0.0.1',args.port))
        except OSError as e:
            if receiver:receiver.close()
            print(f'遥测端口不可用：{e}。请关闭其他监听同端口的窗口或指定 --port。',file=sys.stderr);return 2
    try:
        from .quantities import TorqueConversion
        config=TorqueConversion.load(args.torque_config) if args.torque_config else TorqueConversion.load()
        window=MonitorWindow(args.port,args.demo,config,fps=args.fps,backend=args.curve_backend,smooth=args.smooth_curves)
        timer=QTimer(window);invalid=0;started=time.monotonic()
        if args.demo:
            from .demo import DemoSource
            demo=DemoSource()
            def receive():
                now=time.monotonic_ns()
                window.state.ingest(decode_packet(json.dumps(demo.sample(time.monotonic()-started,now)).encode()),now)
        else:
            def receive():
                nonlocal invalid
                for _ in range(128):
                    try:data,_=receiver.recvfrom(65535)
                    except BlockingIOError:break
                    except OSError as e:
                        timer.stop();window.notice.setText(f'遥测接收停止：{e}');break
                    try:window.state.ingest(decode_packet(data),time.monotonic_ns())
                    except (ValueError,RecursionError) as e:
                        invalid+=1;window.notice.setText(f'已忽略 {invalid} 个无效包：{e}')
        timer.timeout.connect(receive);timer.start(20)
        window.show()
        result={'screenshot_ok':True}
        if args.screenshot:
            def capture():
                args.screenshot.parent.mkdir(parents=True,exist_ok=True)
                result['screenshot_ok']=window.rendered and window.grab().save(str(args.screenshot))
                print(f'截图：{args.screenshot}；三维已渲染={window.rendered}；渲染错误={window.render_error}',flush=True)
            QTimer.singleShot(max(100,int(args.duration*1000)-400),capture)
        if args.duration:QTimer.singleShot(int(args.duration*1000),window.close)
        app.aboutToQuit.connect(timer.stop)
        status=app.exec_();window.close()
        return status if result['screenshot_ok'] else 3
    finally:
        if receiver:receiver.close()


if __name__=='__main__':
    raise SystemExit(main())
