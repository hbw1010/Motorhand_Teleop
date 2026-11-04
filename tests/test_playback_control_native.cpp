#include "../src/motorhand_teleop/real_hand/native_control/playback_control.hpp"

#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <termios.h>
#include <string>
#include <unistd.h>

using motorhand::native::KeyEvent;
using motorhand::native::PlaybackClock;
using motorhand::native::TerminalKeys;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void same_terminal(const termios& a, const termios& b) {
    require(a.c_iflag == b.c_iflag && a.c_oflag == b.c_oflag &&
            a.c_cflag == b.c_cflag && a.c_lflag == b.c_lflag,
            "terminal flags not restored");
    for (unsigned i = 0; i < NCCS; ++i) require(a.c_cc[i] == b.c_cc[i], "terminal control characters not restored");
    require(cfgetispeed(&a) == cfgetispeed(&b) && cfgetospeed(&a) == cfgetospeed(&b), "terminal speeds not restored");
}
void clock_test() {
    constexpr std::uint64_t s = 1'000'000'000;
    PlaybackClock c(2 * s, 10 * s);
    require(c.position(10 * s + 200'000'000) == 0, "startup transition advances trajectory");
    require(c.position(11 * s + 200'000'000, false) == 0, "not-ready advances trajectory");
    require(c.position(11 * s + 500'000'000) == 0, "startup transition did not freeze");
    require(c.position(11 * s + 750'000'000) == 250'000'000, "clock is not 1x");
    c.pause(12 * s, true);
    require(c.position(20 * s) == 500'000'000, "takeover did not freeze position");
    c.resume(20 * s);
    require(c.position(20 * s + 200'000'000) == 500'000'000, "resume transition advances trajectory");
    require(c.position(21 * s + 200'000'000, false) == 500'000'000, "resume not-ready advanced trajectory");
    require(c.position(21 * s + 500'000'000) == 500'000'000, "resume transition did not freeze");
    require(c.position(22 * s) == s, "resume was discontinuous");
    c.pause(22 * s + 100'000'000, false);
    require(c.position(23 * s) == s, "not-ready pause advanced trajectory");
    c.resume(23 * s);
    require(c.position(30 * s) == 2 * s && c.position(31 * s) == 2 * s, "duration not clamped");
    bool threw = false;
    try { c.position(30 * s); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "backwards clock accepted");
    threw = false;
    try { PlaybackClock invalid(0, 0); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "zero duration accepted");
}
void terminal_test() {
    int master = posix_openpt(O_RDWR | O_NOCTTY);
    require(master >= 0, "posix_openpt failed");
    require(grantpt(master) == 0 && unlockpt(master) == 0, "PTY setup failed");
    char* name = ptsname(master);
    require(name != nullptr, "ptsname failed");
    int slave = open(name, O_RDWR | O_NOCTTY);
    require(slave >= 0, "PTY slave open failed");
    termios before{}, during{}, after{};
    require(tcgetattr(slave, &before) == 0, "PTY tcgetattr failed");
    const int flags = fcntl(slave, F_GETFL);
    require(flags >= 0, "PTY F_GETFL failed");
    try {
        TerminalKeys keys(slave);
        require(keys.active(), "TTY not active");
        require(tcgetattr(slave, &during) == 0, "PTY tcgetattr during failed");
        require(!(during.c_lflag & ICANON) && !(during.c_lflag & ECHO) && (during.c_lflag & ISIG), "incorrect terminal mode");
        require((fcntl(slave, F_GETFL) & O_NONBLOCK) != 0, "TTY is blocking");
        require(keys.read() == KeyEvent::none, "unexpected key");
        const char data[] = "xT q";
        require(write(master, data, sizeof(data) - 1) == sizeof(data) - 1, "PTY write failed");
        pollfd readable{slave, POLLIN, 0};
        require(poll(&readable, 1, 1000) == 1 && (readable.revents & POLLIN), "PTY input not ready");
        require(keys.read() == KeyEvent::toggle, "T not recognized");
        require(keys.read() == KeyEvent::toggle, "space not recognized");
        require(keys.read() == KeyEvent::quit, "q not recognized");
        require(keys.read() == KeyEvent::none, "unexpected trailing key");
        throw std::runtime_error("unwind");
    } catch (const std::runtime_error& e) {
        require(std::string(e.what()) == "unwind", e.what());
    }
    require(tcgetattr(slave, &after) == 0, "PTY tcgetattr after failed");
    same_terminal(before, after);
    require(fcntl(slave, F_GETFL) == flags, "descriptor flags not restored");
    close(slave);
    close(master);
    int fds[2];
    require(pipe(fds) == 0, "pipe failed");
    {
        TerminalKeys keys(fds[0]);
        require(!keys.active(), "non-TTY marked active");
        require(write(fds[1], "q", 1) == 1, "pipe write failed");
        require(keys.read() == KeyEvent::none, "non-TTY produced key event");
    }
    close(fds[0]); close(fds[1]);
}
}
int main() {
    try { clock_test(); terminal_test(); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}
