import matplotlib.pyplot as plt
import pandas as pd

RESULTS_0 = pd.read_csv("results_0.csv")
RESULTS_1 = pd.read_csv("results_1.csv")

plt.figure()
plt.plot(RESULTS_0["precision"], label="Precision")
plt.plot(RESULTS_0["recall"], label="recall")
plt.plot(RESULTS_0["f1-score"], label="f1-score")
plt.title("Klasse 0")
plt.grid(which="both")
plt.legend()
plt.show()

plt.figure()
plt.plot(RESULTS_1["precision"], label="Precision")
plt.plot(RESULTS_1["recall"], label="recall")
plt.plot(RESULTS_1["f1-score"], label="f1-score")
plt.title("Klasse 1")
plt.grid(which="both")
plt.legend()
plt.show()
