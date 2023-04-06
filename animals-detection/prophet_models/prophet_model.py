from prophet import Prophet
import numpy as np
import pandas as pd
from prophet.serialize import model_to_json, model_from_json
from config import *
from datetime import datetime, timedelta


def prepare_model_and_data(scores_df, freq="15T"):
    prophet_df = scores_df.copy()
    prophet_df.resample(freq, on="time").mean()
    prophet_df.rename(columns={"time": "ds", "score": "y"}, inplace=True)
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    m = Prophet().fit(prophet_df)
    return prophet_df, m


def predict_period(
    prophet_model,
    period=None,
    start=None,
    freq="15T",
    normalize_score=True,
    filter_timetable=(8, 23),
):
    start = (
        pd.to_datetime("today").round(freq)
        if not start
        else datetime.strptime(start, "%Y-%m-%d")
    )
    period = (
        pd.Timedelta(start + pd.offsets.Week(weekday=0) - start)
        - pd.Timedelta(hours=start.hour)
        if not period
        else period
    )
    pred_dates = pd.date_range(
        start=start,
        end=start + pd.Timedelta(period),
        freq=freq,
    )
    pred_dates = pred_dates.to_frame(index=False, name="ds")
    forecast = prophet_model.predict(pred_dates)
    if normalize_score:
        forecast["yhat"] = forecast["yhat"].clip(0, 1)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(0, 1)
        forecast["yhat_upper"] = forecast["yhat_upper"].clip(0, 1)
    if filter_timetable:
        hour_start, hour_end = filter_timetable
        forecast = forecast[
            (forecast["ds"].dt.hour >= hour_start) & (forecast["ds"].dt.hour < hour_end)
        ]
    return forecast


if __name__ == "__main__":
    dates = pd.date_range(start="2023-02-10", end="2023-03-10", freq="5T")
    synth_df = dates.to_frame(index=False, name="time")
    synth_df["score"] = np.cos(2 * np.pi * synth_df["time"].dt.hour / 16)
    synth_df["score"] = (synth_df["score"] - synth_df["score"].min()) / (
        synth_df["score"].max() - synth_df["score"].min()
    )
    synth_df["score"] = synth_df["score"] > synth_df["score"].mean()

    prophet_df, model = prepare_model_and_data(synth_df)

    # Save model
    with open("prophet_models/prophet_model.json", "w") as fout:
        fout.write(model_to_json(model))

    # Load model
    with open("prophet_models/prophet_model.json", "r") as fin:
        model = model_from_json(fin.read())

    pred = predict_period(model)[["ds", "yhat"]]
    pd.set_option("display.max_rows", len(pred))
    print(f"Predicted scores:{pred}")

    for cnl in cnls_cnfg:
        cnl_df = synth_df.copy()
        cnl_df["score"] = cnl_df["score"] + np.random.randint(0, 2, len(synth_df))
        cnl_df["score"] = cnl_df["score"] > cnl_df["score"].mean()
        prophet_df, cnl_model = prepare_model_and_data(cnl_df)

        # Save model
        with open(f"prophet_models/prophet_model_{cnl}.json", "w") as fout:
            fout.write(model_to_json(cnl_model))
