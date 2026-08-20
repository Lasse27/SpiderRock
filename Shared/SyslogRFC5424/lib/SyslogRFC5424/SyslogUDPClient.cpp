#include "RFC5424Helpers.h"
#include "SyslogUDPClient.h"

namespace RFC5424
{
    bool waitForTimeSync(uint32_t timeoutMs = 10000)
    {
        struct tm timeinfo;
        uint32_t start = millis();

        while (!getLocalTime(&timeinfo, 1000))
        { // Arduino-ESP32-Helper, wartet bis zu 1s pro Versuch
            if (millis() - start > timeoutMs)
            {
                return false; // Timeout, keine Sync erreicht
            }
        }
        return true;
    }

    // ================================================================
    bool UDPClient::initialize(char *out_error, uint8_t out_error_size)
    {
        bool isErrorWriting = out_error != nullptr && out_error_size > 0;
        if (!WiFi.isConnected())
        {
            if (isErrorWriting)
            {
                snprintf(out_error, out_error_size, WIFI_CONNECTION_MISSING);
                log_e(WIFI_CONNECTION_MISSING);
            }
            return false;
        }
        if (!_wifiUDP.begin(SYSLOG_DEFAULT_PORT))
        {
            if (isErrorWriting)
            {
                snprintf(out_error, out_error_size, WIFI_BEGIN_FAILED);
                log_e(WIFI_BEGIN_FAILED);
            }
            return false;
        }

        // Get time from ntp
        configTime(0, 0, "pool.ntp.org");
        if (!waitForTimeSync())
        {
            if (isErrorWriting)
            {
                snprintf(out_error, out_error_size, NTP_TIMEOUT_FAIL);
                log_e(NTP_TIMEOUT_FAIL);
            }
            return false;
        }

        // Write startup message.
        if (!_sendStartupMessage())
        {
            if (isErrorWriting)
            {
                snprintf(out_error, out_error_size, STARTUP_MESSAGE_FAILED);
                log_e(STARTUP_MESSAGE_FAILED);
            }
            return false;
        }

        return true;
    }

    // ================================================================
    bool UDPClient::send(Facility facility, Severity severity, char *msgId, char *message)
    {
        return false;
    }

    // ================================================================
    bool UDPClient::_sendStartupMessage()
    {
        return _send("Connected");
    }

    // ================================================================
    bool UDPClient::_send(const char *messsage)
    {
        char *fmtMessage = _formatMessage(Facility::KERNEL_MESSAGES, Severity::ALERT, "-", messsage);
        if (_wifiUDP.beginPacket(_configuration.serverIp, _configuration.serverPort))
        {
            _wifiUDP.print(messsage);
            if (_wifiUDP.endPacket())
            {
                return true;
            }
        }
        return false;
    }

#define MOVE_CURSOR(lenVar, cursorVar, remainingVar, bufferVar)        \
    do                                                                 \
    {                                                                  \
        if (lenVar < 0 || static_cast<size_t>(lenVar) >= remainingVar) \
        {                                                              \
            delete[] bufferVar;                                        \
            return nullptr;                                            \
        }                                                              \
        cursorVar += lenVar;                                           \
        remainingVar -= lenVar;                                        \
    } while (0)

    // ================================================================
    char *UDPClient::_formatMessage(Facility facility, Severity severity, const char *msgId, const char *message)
    {
        char *buffer = new char[SYSLOG_MESSAGE_LENGTH_LIMIT];
        char *cursor = buffer;
        size_t remaining = SYSLOG_MESSAGE_LENGTH_LIMIT;

        // Priority
        unsigned short priority = calculatePriority(facility, severity);
        int len = snprintf(cursor, remaining, "<%u> ", priority);
        MOVE_CURSOR(len, cursor, remaining, buffer);
        Serial.println(buffer);

        // Version
        len = snprintf(cursor, remaining, "1 ");
        MOVE_CURSOR(len, cursor, remaining, buffer);
        Serial.println(buffer);

        // Timestamp
        struct timeval tv;
        gettimeofday(&tv, nullptr);

        struct tm timeinfo;
        gmtime_r(&tv.tv_sec, &timeinfo); // UTC! Für lokale Zeit: localtime_r

        char dateTimePart[20]; // "YYYY-MM-DDTHH:MM:SS" = 19 Zeichen + '\0'
        strftime(dateTimePart, sizeof(dateTimePart), "%Y-%m-%dT%H:%M:%S", &timeinfo);

        // Millisekunden anhängen + 'Z' für UTC
        len = snprintf(cursor, remaining, "%s.%03ldZ", dateTimePart, tv.tv_usec / 1000);
        MOVE_CURSOR(len, cursor, remaining, buffer);
        Serial.println(buffer);

        return buffer;
    }

} // namespace RFC5424
