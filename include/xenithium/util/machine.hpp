// Xenithium · the machine a process runs on, as a benchmark records it: the OS, the kernel, the CPU, its threads and the
// host memory. The server reports its own in GET /v1/models; xenithium-bench records the machine it ran on. Read from
// /etc/os-release, uname(2), /proc/cpuinfo and /proc/meminfo: no root, nothing that names a user or a path.
#pragma once

#include <sys/utsname.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <string>

namespace xenithium::util {

struct Machine {
  std::string os;  // os-release's PRETTY_NAME, "Ubuntu 26.04.1 LTS"; empty = not known
  std::string kernel;  // uname's release, "7.0.0-43-generic"
  std::string cpu;  // the first "model name" of /proc/cpuinfo, "AMD EPYC 7282 16-Core Processor"
  int cpu_threads = 0;  // online logical CPUs
  int memory_gib = 0;  // MemTotal, rounded to whole GiB
};

inline Machine this_machine() {
  Machine m;
  std::string line;
  if (std::ifstream os("/etc/os-release"); os)
    while (std::getline(os, line))
      if (line.rfind("PRETTY_NAME=", 0) == 0) {
        m.os = line.substr(12);
        if (m.os.size() >= 2 && m.os.front() == '"' && m.os.back() == '"') m.os = m.os.substr(1, m.os.size() - 2);
      }
  if (utsname u{}; ::uname(&u) == 0) m.kernel = u.release;
  if (std::ifstream cpu("/proc/cpuinfo"); cpu)
    while (std::getline(cpu, line))
      if (line.rfind("model name", 0) == 0) {
        const size_t colon = line.find(':');
        const size_t at = colon == std::string::npos ? colon : line.find_first_not_of(" \t", colon + 1);
        if (at != std::string::npos) m.cpu = line.substr(at);  // nothing after the colon: not known
        break;
      }
  if (const long n = ::sysconf(_SC_NPROCESSORS_ONLN); n > 0) m.cpu_threads = static_cast<int>(n);
  if (std::ifstream mem("/proc/meminfo"); mem)
    while (std::getline(mem, line))
      if (line.rfind("MemTotal:", 0) == 0) {
        m.memory_gib = static_cast<int>(std::lround(std::stod(line.substr(9)) / (1024.0 * 1024.0)));  // kB
        break;
      }
  return m;
}

}  // namespace xenithium::util
