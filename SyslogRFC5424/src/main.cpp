#include <Arduino.h>
#include <WiFi.h>
#include "SyslogUDPClient.h"
#include "esp_sntp.h"

int status = WL_IDLE_STATUS;
char ssid[] = WIFI_SSID;
char pass[] = WIFI_PASS;

RFC5424::client_conf_t createConfig()
{
  RFC5424::client_conf_t config{};
  strcpy(config.serverIp, "DESKTOP-C3EE3NS");
  strcpy(config.hostname, "esp32");
  strcpy(config.appname, "DoorNode");
  config.serverPort = 514;
  return config;
}

RFC5424::client_conf_t config = createConfig();
RFC5424::UDPClient client(createConfig());

void printWifiStatus()
{
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your WiFi shield's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}

void setupWiFi()
{
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, pass);
  Serial.print("Connecting to WiFi ..");
  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.print('.');
    delay(1000);
  }
  printWifiStatus();
}

void setup()
{

  // Initialize serial and wait for port to open:
  Serial.begin(115200);
  while (!Serial)
  {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  setupWiFi();
}

void loop()
{
  char error_str[64];
  char error_str_len = 64;
  Serial.println(error_str);
  client.initialize(error_str, error_str_len);
  delay(5000);
}
