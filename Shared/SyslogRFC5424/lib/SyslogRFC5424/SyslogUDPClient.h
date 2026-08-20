#pragma once

#include "WiFi.h"
#include "RFC5424Defs.h"

namespace RFC5424
{
    /// @brief Syslog client that transmits log messages via UDP (RFC 5424 format).
    /// @details Not copyable, since two instances must not share the same underlying
    ///          WiFiUDP socket state. Movable, since ownership of the socket can be
    ///          safely transferred to another instance.
    class UDPClient
    {
    public:
        /// @brief Creates a new instance of UDPClient. Explicit keyword to prohibit implicit castings.
        /// @param config The client configuration for the UDPClient.
        explicit UDPClient(client_conf_t config) : _configuration(config) {}

        /// @brief Destructs an instance of UDPClient and stops any active UDP socket.
        ~UDPClient()
        {
            _wifiUDP.stop();
        }

        /// @brief Copy constructor. Deleted because there can't be two clients using the same socket reference.
        UDPClient(const UDPClient &) = delete;

        /// @brief Copy assignment operator. Deleted because there can't be two clients using the same socket reference.
        UDPClient &operator=(const UDPClient &) = delete;

        /// @brief Move constructor. Transfers socket ownership from another instance.
        UDPClient(UDPClient &&) noexcept = default;

        /// @brief Move assignment operator. Transfers socket ownership from another instance.
        UDPClient &operator=(UDPClient &&) noexcept = default;

        /// @brief Initializes the UDPClient. Returns possible errors as outbound char[].
        /// @param out_error Buffer that receives a human-readable error message on failure. Buffer must be at least 64 bytes.
        /// @param out_error_size The size of the outbound buffer.
        /// @return true if the initialization succeeded, otherwise false.
        bool initialize(char *out_error = nullptr, uint8_t out_error_size = 0);

        /// @brief
        /// @param facility
        /// @param severity
        /// @param msgId
        /// @param message
        /// @return
        bool send(Facility facility, Severity severity, char *msgId, char *message);

    private:
        bool _sendStartupMessage();
        bool _send(const char *messsage);
        char *_formatMessage(Facility facility, Severity severity, const char *msgId, const char *message);

        /// @brief Configuration used to establish and label the UDP connection (server, port, hostname, app name).
        client_conf_t _configuration;

        /// @brief Underlying UDP socket used for sending syslog messages. Owned by this instance (RAII member).
        WiFiUDP _wifiUDP;
    };

} // namespace RFC5424