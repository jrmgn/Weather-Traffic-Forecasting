# a_visualize.py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("driver_delay_weather.csv", parse_dates=["date"], index_col="date")
df = df.asfreq("D").ffill()

train = df[:-30]
test = df[-30:]
exog_train = train[["temp","precip","traffic_index"]]
exog_test = test[["temp","precip","traffic_index"]]

model = SARIMAX(train["delays"], order=(1,1,1), exog=exog_train, enforce_stationarity=False)
res = model.fit(disp=False)
forecast = res.get_forecast(steps=30, exog=exog_test)
pred_mean = forecast.predicted_mean

plt.figure(figsize=(12,5))
plt.plot(df.index, df["delays"], label="Delays (actual)")
plt.plot(pred_mean.index, pred_mean, label="Delays (forecast)", linestyle="--")
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(df.index, df["precip"], alpha=0.3, label="Precip (mm)", color="green")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.title("Delays: Actual vs Forecast (with Precipitation Trend)")
plt.savefig("delays_vs_weather.png", dpi=150)
print("Saved delays_vs_weather.png")