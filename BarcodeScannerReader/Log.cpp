#include "stdafx.h"
#include "Log.h"


std::shared_ptr<spdlog::logger> Log::logger_;

void Log::init(std::string loggerName, std::string pattern) {
	spdlog::set_pattern(pattern);
	logger_ = spdlog::stdout_color_mt(loggerName);
	logger_->set_level(spdlog::level::trace);
}

