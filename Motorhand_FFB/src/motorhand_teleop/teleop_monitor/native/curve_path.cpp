// 只构造显示路径：逐点保留，不抽稀、不平滑，不参与设备控制。
#include <QPainterPath>
#include <QtGlobal>
#include <cstdint>
#include <cmath>

extern "C" const char* curve_qt_build_version() noexcept { return QT_VERSION_STR; }
extern "C" const char* curve_qt_runtime_version() noexcept { return qVersion(); }
extern "C" int build_curve(void* handle, const std::uint64_t* times, const double* values,
                           const unsigned char* breaks, int count, double start, double end,
                           double left, double bottom, double width, double height,
                           double low, double high, double scale) noexcept {
    if (!handle || !times || !values || !breaks || count < 0) return -1;
    for (double v : {start,end,left,bottom,width,height,low,high,scale})
        if (!std::isfinite(v)) return -1;
    if (end <= start || high <= low || width <= 0 || height <= 0 || scale <= 0) return -1;
    try {
        auto& path = *static_cast<QPainterPath*>(handle);
        bool previous = false;
        std::uint64_t last = 0;
        for (int i = 0; i < count; ++i) {
            const double value = values[i];
            const auto time = times[i];
            if (!std::isfinite(value) || time < start || time > end) { previous = false; continue; }
            const double x = left + (double(time) - start) / (end - start) * width;
            const double y = bottom - (value / scale - low) / (high - low) * height;
            if (!std::isfinite(x) || !std::isfinite(y)) { previous = false; continue; }
            if (!previous || breaks[i] || time <= last || time - last > 250'000'000)
                path.moveTo(x,y);
            else path.lineTo(x,y);
            previous = true; last = time;
        }
        return 0;
    } catch (...) { return -2; }
}
