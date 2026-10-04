# Wakeup-Word-Module

Dieses Modul befasst sich mit der Umsetzung der Wakeup-Word Mechanik. 

Rocky soll kontinuierlich die Umgebung mit externen Mikrofonen belauschen und auf ein Signalwort hin das Audiosignal der Nutzeranfrage aufnehmen.
Vergleichen kann man das mit Amazon Alexa oder beispielsweise Siri auf dem Iphone. 

Die Wakeup-Word-Mechanik soll ausschließlich der Erkennung dienen und nach der Erkennung den Signalstrang nach dem Wakeup-Word aufzeichnen und 
zur Interpretation an das Speech-Unterstanding-Modul schicken. 

Zur Umsetzung davon kommt ein kleines CNN zum Einsatz, welches im Ordner `Model` beschrieben ist. Die dafür notwendigen Datensätze wurden mit dem
Open-Source TTS Modell QwentTTS3 generiert und anschließend durch eigene Erweiterungen im Rahmen der Audioverarbeitung vorbereitet.

Da das Wakeup-Word-Modul keine krassen Hardwareanforderungen hat, wird es auf einem ESP32 umgesetzt. Dabei kommt das Framework ESP-DL zur Inference des
Modells auf dem Mikrocontroller zum Einsatz. Das ist alles in `Deployment` beschrieben.