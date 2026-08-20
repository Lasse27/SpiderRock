#pragma once
#include <Arduino.h>

namespace RFC5424
{

#define WIFI_CONNECTION_MISSING "The device is not connected to a WIFI."
#define WIFI_BEGIN_FAILED "WiFiUDP::begin failed. Socket initialization failed."
#define STARTUP_MESSAGE_FAILED "Error while sending startup message."
#define NTP_TIMEOUT_FAIL "Time synchonization with NTP Server failed."
#define SYSLOG_DEFAULT_PORT 514
#define SYSLOG_MESSAGE_LENGTH_LIMIT 2048
#define EMPTY_STRING ""
#define NIL_VALUE "-"

    /// @brief Configuration (struct) for the syslog client.
    typedef struct
    {
        /// @brief The IP-address of the syslog server the client is sending messages to.
        char serverIp[255] = EMPTY_STRING;

        /// @brief The hostname, IPv4-Address or the IPv6-address of the client (defaults to "-").
        char hostname[255] = NIL_VALUE;

        /// @brief The name of the app that is using the syslog client. (defaults to "-").
        char appname[48] = NIL_VALUE;

        /// @brief The port of of the syslog server the client is sending messages to (defaults to 514).
        uint16_t serverPort = SYSLOG_DEFAULT_PORT;
    } client_conf_t;

    // RFC 5424 Facility Codes
    enum Facility : unsigned char
    {
        KERNEL_MESSAGES = 0,
        USER_LEVEL_MESSAGES = 1,
        MAIL_SYSTEM = 2,
        SYSTEM_DAEMONS = 3,
        SECURITY_AUTHORIZATION_MESSAGES = 4,
        SYSLOGD_INTERNAL = 5,
        LINE_PRINTER_SUBSYSTEM = 6,
        NETWORK_NEWS_SUBSYSTEM = 7,
        UUCP_SUBSYSTEM = 8,
        CLOCK_DAEMON = 9,
        SECURITY_AUTHORIZATION_MESSAGES2 = 10,
        FTP_DAEMON = 11,
        NTP_SUBSYSTEM = 12,
        LOG_AUDIT = 13,
        LOG_ALERT = 14,
        CLOCK_DAEMON2 = 15,
        LOCAL_USE_0 = 16,
        LOCAL_USE_1 = 17,
        LOCAL_USE_2 = 18,
        LOCAL_USE_3 = 19,
        LOCAL_USE_4 = 20,
        LOCAL_USE_5 = 21,
        LOCAL_USE_6 = 22,
        LOCAL_USE_7 = 23
    };

    // RFC 5424 Severity Level
    enum Severity : unsigned char
    {
        EMERGENCY = 0,
        ALERT = 1,
        CRITICAL = 2,
        ERROR = 3,
        WARNING = 4,
        NOTICE = 5,
        INFO = 6,
        DEBUG = 7
    };
}