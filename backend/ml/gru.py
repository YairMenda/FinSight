import sys
import json
from datetime import datetime, timedelta
from model_utils import GRU_model


def main():
    symbol = sys.argv[1]
    from_date = sys.argv[2]
    to_date = sys.argv[3]
    days_to_predict = int(sys.argv[4])

    from_dt = datetime.strptime(from_date, "%Y-%m-%d")
    start_date = (from_dt - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    forecast_df = GRU_model([symbol], start_date, to_date, symbol)
    forecast_df = forecast_df[forecast_df['Date'] >= from_date].head(days_to_predict)

    result = [
        {"day": row["Date"].strftime("%Y-%m-%d"), "price": round(row["Forecast"], 2)}
        for _, row in forecast_df.iterrows()
    ]
    print(json.dumps(result))


if __name__ == '__main__':
    main()
