#pragma once

#include <cstdint>
#include <termios.h>

namespace motorhand::native {

// All timestamps are steady-clock nanoseconds in the same clock domain.
// Call position each cycle with the latest readiness, including while paused.
class PlaybackClock {
public:
    static constexpr std::uint64_t default_transition_ns = 500'000'000;
    PlaybackClock(std::uint64_t duration_ns, std::uint64_t start_ns,
                  std::uint64_t transition_ns = default_transition_ns);
    std::uint64_t position(std::uint64_t now_ns, bool ready = true);
    void pause(std::uint64_t now_ns, bool ready = true);
    void resume(std::uint64_t now_ns);
    bool playing() const noexcept { return playing_; }
    std::uint64_t duration_ns() const noexcept { return duration_ns_; }
private:
    std::uint64_t duration_ns_, transition_ns_, last_ns_, position_ns_ = 0;
    std::uint64_t transition_remaining_ns_;
    bool playing_ = true;
};

enum class KeyEvent { none, toggle, quit };

// Owns terminal settings and file status flags, but not the descriptor.
// Non-TTY descriptors remain untouched and always return KeyEvent::none.
class TerminalKeys {
public:
    explicit TerminalKeys(int fd = 0);
    ~TerminalKeys();
    TerminalKeys(const TerminalKeys&) = delete;
    TerminalKeys& operator=(const TerminalKeys&) = delete;
    bool active() const noexcept { return active_; }
    // Read one recognized event, skipping unrelated input. Call until none.
    // EOF or a read error is reported by exception so playback can stop safely.
    KeyEvent read();
private:
    int fd_;
    int saved_flags_ = 0;
    termios saved_termios_{};
    bool active_ = false;
};

} // namespace motorhand::native
