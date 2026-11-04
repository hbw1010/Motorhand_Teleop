#include "playback_control.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace motorhand::native {
namespace {
std::runtime_error system_error(const char* operation, int error) {
    return std::runtime_error(std::string(operation) + ": " + std::strerror(error));
}
}

PlaybackClock::PlaybackClock(std::uint64_t duration_ns, std::uint64_t start_ns,
                             std::uint64_t transition_ns)
    : duration_ns_(duration_ns), transition_ns_(transition_ns), last_ns_(start_ns),
      transition_remaining_ns_(transition_ns) {
    if (!duration_ns) throw std::invalid_argument("playback duration must be positive");
}

std::uint64_t PlaybackClock::position(std::uint64_t now_ns, bool ready) {
    if (now_ns < last_ns_) throw std::invalid_argument("playback clock went backwards");
    std::uint64_t elapsed = now_ns - last_ns_;
    last_ns_ = now_ns;
    if (!playing_ || !ready) return position_ns_;
    const auto transition = std::min(elapsed, transition_remaining_ns_);
    transition_remaining_ns_ -= transition;
    elapsed -= transition;
    position_ns_ += std::min(elapsed, duration_ns_ - position_ns_);
    return position_ns_;
}

void PlaybackClock::pause(std::uint64_t now_ns, bool ready) {
    position(now_ns, ready);
    playing_ = false;
}

void PlaybackClock::resume(std::uint64_t now_ns) {
    if (playing_) throw std::logic_error("playback is already playing");
    position(now_ns, false);
    playing_ = true;
    transition_remaining_ns_ = transition_ns_;
}

TerminalKeys::TerminalKeys(int fd) : fd_(fd) {
    if (!::isatty(fd_)) return;
    if (::tcgetattr(fd_, &saved_termios_) != 0) throw system_error("tcgetattr", errno);
    saved_flags_ = ::fcntl(fd_, F_GETFL);
    if (saved_flags_ < 0) throw system_error("fcntl F_GETFL", errno);
    termios raw = saved_termios_;
    ::cfmakeraw(&raw);
    // Ctrl+C must remain a signal handled by the owning process.
    raw.c_lflag |= ISIG;
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (::tcsetattr(fd_, TCSANOW, &raw) != 0) throw system_error("tcsetattr", errno);
    if (::fcntl(fd_, F_SETFL, saved_flags_ | O_NONBLOCK) != 0) {
        const int error = errno;
        ::tcsetattr(fd_, TCSANOW, &saved_termios_);
        throw system_error("fcntl F_SETFL", error);
    }
    active_ = true;
}

TerminalKeys::~TerminalKeys() {
    if (active_) {
        ::fcntl(fd_, F_SETFL, saved_flags_);
        ::tcsetattr(fd_, TCSANOW, &saved_termios_);
    }
}

KeyEvent TerminalKeys::read() {
    if (!active_) return KeyEvent::none;
    for (;;) {
        unsigned char key;
        const auto n = ::read(fd_, &key, 1);
        if (n == 1) {
            if (key == 't' || key == 'T' || key == ' ') return KeyEvent::toggle;
            if (key == 'q' || key == 'Q') return KeyEvent::quit;
            continue;
        }
        if (n == 0) throw std::runtime_error("terminal input closed");
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return KeyEvent::none;
        throw system_error("terminal read", errno);
    }
}
} // namespace motorhand::native
