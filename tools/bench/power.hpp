// xenithium-bench: what the GPUs draw while the benchmark runs, from the kernel's hwmon files.
// A card's board energy counter (energy1_input, microjoules, cumulative) gives exact energy between two readings; the
// actual GPU clock (act_freq, MHz) is a momentary value, so a thread samples both a few times per second. Tokens per
// joule (= tok/s per watt) is then the efficiency of a scenario. The counters cover the whole card: everything else
// running on it is included, and the host's CPU is not.
// Only meaningful when the endpoint runs on this machine's cards (`--power auto` turns it on for a loopback URL).
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "types.hpp"

namespace xenithium::bench {

// One reading of every card.
struct PowerSample {
  double at_s = 0;  // since the meter started
  std::vector<uint64_t> energy_uj;  // cumulative, per card
  std::vector<double> mhz;  // actual clock per card (0 = not readable)
  std::vector<double> temp_c;  // GPU package temperature per card (0 = not readable)
  std::vector<std::string> throttle;  // throttle reasons per card, space-separated ("" = none or not readable)
};

class PowerMeter {
 public:
  // The cards under `drm_root` that have an energy counter; nullptr when there is none (another GPU vendor, a
  // container without sysfs, another machine).
  static std::unique_ptr<PowerMeter> open(const std::string &drm_root = "/sys/class/drm");
  ~PowerMeter();
  PowerMeter(const PowerMeter &) = delete;
  PowerMeter &operator=(const PowerMeter &) = delete;

  const std::vector<PowerCard> &cards() const { return cards_; }
  // Seconds since the meter started, after taking a reading now: the start or end of a span.
  double mark();
  // What the cards drew between two marks.
  PowerUse use(double from_s, double to_s) const;
  // The readings from `from_s` on as at most `max_points` points of mean power, their times counted from `from_s.
  std::vector<PowerPoint> trace(double from_s, size_t max_points) const;

 private:
  struct CardFiles {
    std::string energy_path, clock_path, temp_path, throttle_path;
  };
  PowerMeter(std::vector<PowerCard> cards, std::vector<CardFiles> files);
  PowerSample read_now() const;
  void sample_until_stopped(std::stop_token stop);

  std::vector<PowerCard> cards_;
  std::vector<CardFiles> files_;
  std::chrono::steady_clock::time_point started_;
  mutable std::mutex mutex_;
  std::vector<PowerSample> samples_;  // ascending at_s
  std::jthread sampler_;  // last member: it stops and joins before the samples go
};

}  // namespace xenithium:bench
