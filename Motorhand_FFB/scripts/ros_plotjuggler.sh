#!/usr/bin/env bash
# ROS Humble 使用系统 Python 3.10；不要把 Pixi Python 3.12 或 Qt 库混入进程。
set -e
root="$(dirname "$(dirname "$(readlink -f "$0")")")"
ros=/opt/ros/humble
local_root="${PLOTJUGGLER_HUMBLE_ROOT:-$HOME/.local/opt/plotjuggler-humble/root}"
local_prefix="$local_root/opt/ros/humble"
mode="${1:-}"
if [ "$#" -gt 0 ]; then shift; fi
if [ ! -r "$ros/setup.bash" ]; then
    echo "找不到 ROS Humble: $ros/setup.bash" >&2
    exit 1
fi
# 只影响子进程；保留ROS_DOMAIN_ID、RMW_IMPLEMENTATION与桌面环境。
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# ROS官方安装的环境脚本可能访问未设置的变量，因此不启用set -u。
source "$ros/setup.bash"
if [ -x "$local_prefix/lib/plotjuggler/plotjuggler" ]; then
    export AMENT_PREFIX_PATH="$local_prefix:${AMENT_PREFIX_PATH:-}"
    export LD_LIBRARY_PATH="$local_prefix/lib:$local_root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$local_prefix/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
    plotter="$local_prefix/lib/plotjuggler/plotjuggler"
    # ROS插件由AMENT索引自动发现；只手动添加主程序插件，避免重复加载。
    plugins="$local_prefix/lib/plotjuggler"
else
    plotter="$ros/lib/plotjuggler/plotjuggler"
    plugins="$ros/lib/plotjuggler"
fi
case "$mode" in
    plot)
        if [ ! -x "$plotter" ]; then
            echo "找不到PlotJuggler；需要ros-humble-plotjuggler和ros-humble-plotjuggler-ros。" >&2
            exit 1
        fi
        exec "$plotter" --plugin_folders "$plugins" "$@"
        ;;
    bridge)
        exec /usr/bin/python3 "$root/scripts/plotjuggler_bridge.py" "$@"
        ;;
    demo)
        exec /usr/bin/python3 "$root/scripts/plotjuggler_bridge.py" --demo "$@"
        ;;
    *)
        echo "用法: bash scripts/ros_plotjuggler.sh {plot|bridge|demo} [参数...]" >&2
        exit 2
        ;;
esac
