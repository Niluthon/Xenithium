// Xenithium · what must not leave the machine when a process says how it was started: a value whose name is a key,
// token, password or the like, a value that looks like a credential whatever its name, and the password of a URL.
// xenithium-server hides these in GET /v1/models' launch report (--report-launch); xenithium-bench in a run's
// server_cmd note on its page. Header-only: the bench links no server code.
#pragma once

#include <cctype>
#include <string>
#include <string_view>

namespace xenithium::util {

// A name whose value must not leave the machine: one of its parts (split at '_' and '-') is KEY, TOKEN, PASS or the
// like, or it contains AUTH, BEARER, JWT, SECRET, PASSW, CREDENTIAL or COOKIE anywhere (XEN_AUTHORIZATION). KEY and
// TOKEN as whole parts only: XEN_KV_POOL_TOKENS is a count.
inline bool secret_name(std::string_view name) {
  std::string part;
  for (size_t at = 0; at <= name.size(); ++at) {
    if (at < name.size() && name[at] != '_' && name[at] != '-') {
      part += static_cast<char>(std::toupper(static_cast<unsigned char>(name[at])));
      continue;
    }
    for (const char* word : {"KEY", "APIKEY", "TOKEN", "PASS", "PWD", "SIG", "SIGNATURE"})
      if (part == word) return true;
    for (const char* inside : {"AUTH", "BEARER", "JWT", "SECRET", "PASSW", "CREDENTIAL", "COOKIE"})
      if (part.find(inside) != std::string::npos) return true;
    part.clear();
  }
  return false;
}

// A value that looks like a credential whatever its name: a known token prefix (hf_, sk-, ghp_, github_pat_, xox?-,
// glpat-, AKIA, a JWT's eyJ...) or "Bearer ...". For a secret passed under an ordinary flag.
inline bool secret_value(std::string_view value) {
  if (value.rfind("Bearer ", 0) == 0 || value.rfind("bearer ", 0) == 0) return true;
  for (const char *prefix : {"hf_", "sk-", "ghp_", "gho_", "ghs_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "AKIA", "eyJ"})
    if (value.rfind(prefix, 0) == 0 && value.size() >= std::string_view(prefix).size() + 16) return true;
  return false;
}

// `value` with the password of every URL in it hidden: "http://user:pass@proxy:3128" -> "http://user:(hidden)@proxy:3128"
// (a proxy setting carries credentials whatever its name).
inline std::string hide_url_credentials(std::string value) {
  for (size_t scheme = value.find("://"); scheme != std::string::npos; scheme = value.find("://", scheme + 3)) {
    const size_t start = scheme + 3;
    const size_t end = value.find_first_of("/?# ", start);
    const size_t at = value.rfind('@', end == std::string::npos ? std::string::npos : end);
    if (at == std::string::npos || at < start) continue;
    const size_t colon = value.find(':', start);
    if (colon == std::string::npos || colon > at) continue;  // a user name alone: no password to hide
    value.replace(colon + 1, at - colon - 1, "(hidden)");
  }
  return value;
}

}  // namespace xenithium::util
