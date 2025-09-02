import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta


def get_data():
    aapl = yf.download("AAPL", start="2010-01-01")
    dates = np.arange(len(aapl.index))      # numeric days for SVR
    prices = aapl["Open"].to_numpy()        # Open prices
    return dates, prices, aapl.index        # real dates for plotting


def predict_prices(dates, prices, real_dates, future_days):
    dates_reshaped = dates.reshape(-1, 1)
    prices = prices.ravel()

 
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)


    svr_lin.fit(dates_reshaped, prices)
    svr_poly.fit(dates_reshaped, prices)
    svr_rbf.fit(dates_reshaped, prices)


    plt.figure(figsize=(12, 6))
    plt.scatter(real_dates, prices, color='black', s=10, label='Actual Data')
    plt.plot(real_dates, svr_lin.predict(dates_reshaped), color='red', label='Linear')
    plt.plot(real_dates, svr_poly.predict(dates_reshaped), color='green', label='Polynomial')
    plt.plot(real_dates, svr_rbf.predict(dates_reshaped), color='blue', label='RBF')


    future_dates = [real_dates[-1] + timedelta(days=int(d)) for d in future_days]
    future_pred_lin = svr_lin.predict(np.array(future_days).reshape(-1, 1))
    future_pred_poly = svr_poly.predict(np.array(future_days).reshape(-1, 1))
    future_pred_rbf = svr_rbf.predict(np.array(future_days).reshape(-1, 1))

    plt.scatter(future_dates, future_pred_lin, color='red', marker='x', s=100, label='Future Linear')
    plt.scatter(future_dates, future_pred_poly, color='green', marker='x', s=100, label='Future Poly')
    plt.scatter(future_dates, future_pred_rbf, color='blue', marker='x', s=100, label='Future RBF')

  
    plt.xticks(real_dates[::500], [real_dates[i].year for i in range(0, len(real_dates), 500)], rotation=45)

    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Apple Stock Price Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return future_pred_lin, future_pred_poly, future_pred_rbf


dates, prices, real_dates = get_data()


future_days = [len(dates) + 100]
predictions = predict_prices(dates, prices, real_dates, future_days)

print("Predictions (Linear, Poly, RBF):", predictions)





