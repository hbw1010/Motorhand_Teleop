#pragma once
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <string>
#include <cstring>

namespace lqs {
class PosixSerial {
public:
    PosixSerial() = default;
    ~PosixSerial() { Close(); }
    bool Open(const std::string& port, int baudrate = 921600) {
        Close();
        fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) return false;
        struct termios tty{};
        if (tcgetattr(fd_, &tty) != 0) { Close(); return false; }
        speed_t speed = (baudrate == 2000000) ? B2000000 : B921600;
        cfsetospeed(&tty, speed);
        cfsetispeed(&tty, speed);
        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8 | CLOCAL | CREAD;
        tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        tty.c_iflag &= ~(IXON | IXOFF | IXANY | IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
        tty.c_oflag &= ~OPOST;
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 0;
        if (tcsetattr(fd_, TCSANOW, &tty) != 0) { Close(); return false; }
        tcflush(fd_, TCIOFLUSH);
        return true;
    }
    void Close() { if (fd_ >= 0) { close(fd_); fd_ = -1; } }
    bool IsOpen() const { return fd_ >= 0; }
    int Write(const void* data, size_t size) { return fd_ < 0 ? -1 : write(fd_, data, size); }
    int Read(void* buffer, size_t max_size) { return fd_ < 0 ? -1 : read(fd_, buffer, max_size); }
private:
    int fd_ = -1;
};
}
