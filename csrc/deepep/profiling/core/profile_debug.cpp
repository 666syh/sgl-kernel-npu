/*
 * SPDX-License-Identifier: MIT
 */

#include "profiling/core/profile_debug.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace deep_ep::profiling::debug {

bool IsEnabled()
{
    const char *env = std::getenv("DEEPEP_PROFILE_DEBUG");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0) {
        env = std::getenv("DEEPEP_FUSED_PROFILE_DEBUG");
    }
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

void Print(const std::string &msg)
{
    if (!IsEnabled()) {
        return;
    }
    std::fprintf(stderr, "[DeepEPProfileDebug] %s\n", msg.c_str());
}

}  // namespace deep_ep::profiling::debug
