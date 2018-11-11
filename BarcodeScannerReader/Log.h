#pragma once
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
class Log
{
public:
	static void init(std::string loggerName, std::string pattern = "%^[%T][%l] %n: %v%$");
	static inline std::shared_ptr<spdlog::logger>& getLogger(){return logger_;}

private:
	static std::shared_ptr<spdlog::logger> logger_;
};

//Logging Macros:

#define LOG_TRACE(...)  ::Log::getLogger()->trace(__VA_ARGS__) 
#define LOG_INFO(...)   ::Log::getLogger()->info(__VA_ARGS__)
#define LOG_WARN(...)   ::Log::getLogger()->warn(__VA_ARGS__)
#define LOG_ERROR(...)  ::Log::getLogger()->error(__VA_ARGS__)
#define LOG_FATAL(...)  ::Log::getLogger()->fatal(__VA_ARGS__)