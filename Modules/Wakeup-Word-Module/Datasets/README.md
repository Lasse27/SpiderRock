# Datasets

Hier landen die erstellten .wav Dateien und die vorverarbeiteten Datensets.

# Trainings-Datensatz

- Zweck: Für das Training des Modells mit PyTorch
- Anteil: 75%

---

## Inhalte

### Verschiedene Sätze:

- **Positive Sätze**: Stellen Sätze dar, die als Wakeword erkannt werden sollen
- **Negative Similars**: Stellen Sätze dar, die zwar ähnlich klingen, jedoch nicht erkannt werden sollen
- **Negative Sätze**: Stellen Sätze dar, die nicht ähnlich klingen und nicht erkannt werden sollen.
- **Noise**: Whitenoise oder Störgeräusche, die nicht erkannt werden sollen

### Verschiedene Sprecher:

- **kokoro**: 28 Sprecher (20 - Amerikaner, 8 - Briten)

### Verschiedene Lautstärken:

- 3 Level (leise, normal, laut)

### Verschiedene Positionen in der Audio:

- vorne, mittig, am Ende

### Verschiedene Hintergrundgeräusche:

- Die selben wie unter **Noise** bei den verschiedenen Sätzen.

---

## Generierung

Für jeden Satz gilt:

1. Für jeden Satz einer Rubrik wird ein Voice-Sample für jeden Sprecher generiert.
2. Für jedes Voice-Sample werden drei verschiedenen Lautstärken generiert.
3. Jede Lautstärke wird mit 10 Audios von Hintergrundgeräuschen versehen (3 Sekunden lang).

> Verschiedene Positionen des Wakewords in der Audio ergeben sich durch die unterschiedlichen Sätze in der Wakeword-Datei

Das ergibt pro Sample:

$$
    \begin{split}
        count &= samples \cdot speakers \cdot volumes \cdot noises \\
              &= samples \cdot 28 \cdot 3 \cdot 10 \\
              &= samples \cdot 840
     \end{split}
$$

Allgemein gilt:

- Angestrebt ist eine pessimistisches Modell, das bedeutet das Modell soll eher kein Wakeword erkennen, als ein Wakeword zu erahnen und falsch zu erkennen.
- Daher wird der Traininsdatensatz überwiegend negative Samples enthalten. Angezielt ist hier eine Verteilung von 80% negativen zu 20% positiven Samples.
- Die Anzahl an negativen Samples bestimmt also die Anzahl an positiven Samples die benötigt werden und umgekehrt. Verhältnis 1 zu 4.
- Die Hintergrundgeräusche werden hierbei außer Acht gelassen, sie kommen am Ende dazu, sodass die Verteilung eher in Richtung 90% zu 10% liegt.

Das bedeutet für die Anzahl der Wörter in den Files: pro Wort in der positiven Wakeword-Datei müssen:

- 2 Wörter in der negativen Similar-Wakeword-Datei stehen
- 2 Wörter in der negativen Wakeword-Datei stehen

Bei genauen Zahlen (50 positive Samples):

$$
    \begin{split}
        count_{positiv} &= samples_{positiv} \cdot speakers \cdot volumes \cdot noises \\
                        &= samples_{positiv} \cdot 840 \\
                        &= 50 \cdot 840 \\
                        &= 42.000
     \end{split}
$$

$$
    \begin{split}
        count_{negativ} &= samples_{negativ} + samples_{similar}\\
                        &= 84.000 + 84.000 \\
                        &= 168.000
     \end{split}
$$

Bei 100kBit pro .wav Datei ergibt das eine maximale Zwischengröße von 2.100.000kbit = 2050mBit

