#pragma once
#include <esp_random.h>
#include "RFC5424Defs.h"

namespace RFC5424
{
    /// @brief Generates a random boot-specific id.
    inline unsigned int generateBootId()
    {
#if defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_ESP8266)
        return esp_random(); // echter Hardware-Zufallsgenerator, plattformübergreifend auf ESP-Cores verfügbar
#else
        randomSeed(analogRead(0) ^ micros()); // AVR: kein HW-RNG, Seed aus "Rauschen"
        return random(1, 0x7FFFFFFF);
#endif
    }

    /// @brief Calculates the severity for a syslog message.
    /// @param facility The given facility.
    /// @param severity The given severity.
    /// @return The facility multiplied by 8 and added to the severity.
    inline unsigned short calculatePriority(Facility facility, Severity severity)
    {
        return facility * 8 + severity;
    }

    bool append(char *source, const unsigned int sourceSize, const char *appendix, const unsigned int appendixSize, unsigned int *index)
    {
        if (*index >= sourceSize || (*index) + appendixSize > sourceSize)
        {
            return false;
        }
        for (unsigned int i = 0; i < appendixSize; i++)
        {
            source[*index] = appendix[i];
            (*index)++;
        }
        return true;
    }

    bool append(char *source, unsigned int sourceSize, char appendix, unsigned int *index)
    {
        if (*index >= sourceSize)
        {
            return false;
        }
        source[*index] = appendix;
        (*index)++;
        return true;
    }

} // namespace RFC5424