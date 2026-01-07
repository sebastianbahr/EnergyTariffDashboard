import streamlit as st
import requests
import time
import datetime
from datetime import date
from zoneinfo import ZoneInfo
from urllib import parse
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Energy Tariff Dashboard", layout="wide")

st.title("Energy Tariff Dashboard — Agile / Tracker vs Fixed")
st.caption("Shows today's and tomorrow's cheapest & highest periods, weighted agile prices and a future price chart.")

# ---------------------------
# Helpers & data fetching
# ---------------------------

now_utc = pd.Timestamp.now(tz='UTC').floor('30min')
now_local = now_utc.tz_convert("Europe/London")

# Dishwasher Consumption Profile (3:30 hours / 7 slots)
DISHWASHER_CONSUMPTION = [0.06, 0.33, 0.1, 0.28, 0.01, 0.02, 0.04] # kWh

@st.cache_data(ttl=300)
def fetch_tariffs(baseurl: str, period_from: str, period_to: str):
    url = f"{baseurl}?period_from={period_from}Z&period_to={period_to}Z"
    tariffs = []
    while url:
        try:
            r = requests.get(url)
            r.raise_for_status()
            price_data = r.json()
            tariffs.extend(price_data.get("results", []))
            url = price_data.get("next")
            time.sleep(0.1) # slight delay to be polite to API
        except requests.RequestException:
            break
            
    if not tariffs:
        return pd.DataFrame()
    df = pd.json_normalize(tariffs)
    df['valid_from'] = pd.to_datetime(df.valid_from)
    df['valid_to'] = pd.to_datetime(df.valid_to)
    df['date'] = df.valid_from.dt.date
    df['time'] = df.valid_from.dt.time
    return df


def add_daytime_shading(fig, now_local, end_dt):
    """
    Adds light grey background shading from 07:00–22:00 (local time),
    starting no earlier than now_local.
    """
    current_day = now_local.normalize()

    while current_day <= end_dt:
        day_start = current_day + pd.Timedelta(hours=7)
        day_end = current_day + pd.Timedelta(hours=22)

        # Clip start so we never shade before "now"
        rect_start = max(day_start, now_local)

        # Only add rectangle if there's something to show
        if rect_start < day_end:
            fig.add_vrect(
                x0=rect_start,
                x1=day_end,
                fillcolor="lightgrey",
                opacity=0.25,
                layer="below",
                line_width=0,
            )

        current_day += pd.Timedelta(days=1)

# ---------------------------
# Dishwasher Optimizer Function
# ---------------------------
def optimal_dishwasher_start_time(full_tariffs_df: pd.DataFrame, consumption: list, today_date_str, tomorrow_date_str, finish_by_time):
    run_length = len(consumption)
    
    today = pd.to_datetime(today_date_str).date()
    tomorrow = pd.to_datetime(tomorrow_date_str).date()
    
    # Filter tariffs for the required time window (from 11pm today until user's finish time tomorrow)
    run_window_df = full_tariffs_df[
        ((full_tariffs_df['date'] == today) & (full_tariffs_df['time'] >= datetime.time(23, 0))) |
        ((full_tariffs_df['date'] == tomorrow) & (full_tariffs_df['time'] < finish_by_time))
    ].sort_values(by=['date', 'time']).reset_index(drop=True)

    if run_window_df.empty or len(run_window_df) < run_length:
        return None, None, "Not enough tariff data available."
    
    # The last index we can start at is (Total slots in window - number of slots for run)
    max_start_index = len(run_window_df) - run_length

    results = []
    for start_index in range(max_start_index + 1):
        prices_for_run = run_window_df['agile_price'].iloc[start_index : start_index + run_length].tolist()
        total_cost_p = sum(c * p for c, p in zip(consumption, prices_for_run))
        
        actual_start_time = run_window_df.iloc[start_index]['valid_from']
        last_slot_start = run_window_df.iloc[start_index + run_length - 1]['valid_from']
        actual_end_time = last_slot_start + datetime.timedelta(minutes=30)
        
        results.append({
            'Start_Time': actual_start_time,
            'End_Time': actual_end_time,
            'Total_Cost_p': total_cost_p
        })
        
    if not results:
        return None, None, "No valid window found."
        
    results_df = pd.DataFrame(results)
    best_run = results_df.loc[results_df['Total_Cost_p'].idxmin()]
    
    return best_run['Start_Time'].to_pydatetime(), best_run['End_Time'].to_pydatetime(), best_run['Total_Cost_p']

# ---------------------------
# Configuration & Setup
# ---------------------------
DEFAULT_BASEURL = 'https://api.octopus.energy/v1/products/AGILE-24-10-01/electricity-tariffs/E-1R-AGILE-24-10-01-C/standard-unit-rates/'
DEFAULT_GAS_URL = 'https://api.octopus.energy/v1/products/SILVER-25-09-02/gas-tariffs/G-1R-SILVER-25-09-02-N/standard-unit-rates/'

date_today = date.today().strftime('%Y-%m-%d')
date_tomorrow_obj = (date.today() + datetime.timedelta(days=1))
date_tomorrow = (date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
date_yesterday = (date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

# Dishwasher Time Logic
st.sidebar.header("Dishwasher Settings")

# Determine default based on weekday/weekend (Tomorrow's day)
# Weekday is 0-4, Weekend is 5-6
if date_tomorrow_obj.weekday() < 5:
    default_time = datetime.time(7, 0)
else:
    default_time = datetime.time(8, 0)

# Create steps: 06:00 to 12:00 in 30 min increments
time_options = []
for hour in range(6, 13):
    time_options.append(datetime.time(hour, 0))
    if hour < 12: # Don't add 12:30
        time_options.append(datetime.time(hour, 30))

selected_finish_time = st.sidebar.select_slider(
    "Finish dishwasher by:",
    options=time_options,
    value=default_time,
    format_func=lambda x: x.strftime("%H:%M")
)

# Electricity Period
elec_period_from = f'{date_yesterday}T00:00'
elec_period_to = f'{date_tomorrow}T23:00'

st.sidebar.header("Electricity Configuration")
fixed_price = st.sidebar.number_input("Fixed tariff (pence/kWh)", min_value=0.0, value=25.24, step=0.01, format="%.2f")
standing_charge_diff = st.sidebar.number_input("Difference in SC (pence)", min_value=0.0, value=6.5, step=0.01, format="%.2f")
regions_multiplier = st.sidebar.number_input("Region multiplier", value=2.1, step=0.01)
regions_peak_adder = st.sidebar.number_input("Peak adder (pence)", value=13.0, step=0.01)
high_threshold = st.sidebar.number_input("High price threshold (pence)", value=30.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Wind Forecast Configuration")

wind_days_ahead = st.sidebar.slider("Days ahead for wind forecast",min_value=1,max_value=7,value=3,step=1)

st.sidebar.markdown("---")
st.sidebar.header("Gas Configuration")
gas_fixed_price = st.sidebar.number_input("Fixed gas price (pence/kWh)", min_value=0.0, value=5.60, step=0.01, format="%.2f")

# ---------------------------
# Electricity Data Fetching
# ---------------------------
with st.spinner("Fetching electricity tariffs..."):
    try:
        raw = fetch_tariffs(DEFAULT_BASEURL.strip(), elec_period_from, elec_period_to)
    except Exception as e:
        st.error(f"Failed to fetch electricity tariffs: {e}")
        st.stop()

if raw.empty:
    st.warning("No electricity tariff data returned.")
    st.stop()

# Calculate Agile Price
standing_charge_diff_std = standing_charge_diff / 3.67
raw['agile_price'] = np.where(
    (raw.valid_from.dt.hour >= 16) & (raw.valid_from.dt.hour < 19),
    (((raw.value_inc_vat - 12) / 2) * regions_multiplier) + regions_peak_adder,
    (raw.value_inc_vat / 2 * regions_multiplier)
)

tariffs_visuals = raw[['agile_price','valid_from','date','time']].copy()
tariffs = tariffs_visuals[~tariffs_visuals.time.isin([datetime.time(23, 0), datetime.time(23, 30)])].reset_index(drop=True)
tariffs['period'] = np.where(
    (tariffs['time'] >= datetime.time(0, 0)) & (tariffs['time'] < datetime.time(6, 30)),
    'night',
    'day')

# gives night a weight of 4 as I can only use 4 hours during the night for the dishwasher
weights = {'night': 4, 'day': 33}
avg_prices = tariffs.groupby(['date', 'period'])['agile_price'].mean().reset_index()
avg_prices['weight'] = avg_prices['period'].map(weights)
weighted_avg = (
    avg_prices.groupby('date')
    .apply(lambda g: (g['agile_price'] * g['weight']).sum() / g['weight'].sum())
    .reset_index(name='weighted_agile_price')
)
weighted_avg = weighted_avg.sort_values('date').reset_index(drop=True)
weighted_avg['relative_diff_to_yda'] = round(weighted_avg.weighted_agile_price.pct_change() * 100, 1)
weighted_avg['relative_diff_to_fixed'] = round(100 * (weighted_avg.weighted_agile_price - (fixed_price - standing_charge_diff_std)) / (fixed_price - standing_charge_diff_std), 1)

day_window = tariffs[
    (tariffs['time'] >= datetime.time(9, 0)) &
    (tariffs['time'] < datetime.time(16, 0))
]
cheapest_per_date = day_window.loc[
    day_window.groupby('date')['agile_price'].idxmin()
].reset_index(drop=True).rename(columns={'agile_price': 'cheapest_price', 'time': 'cheapest_time'})

highest_per_date = tariffs.loc[
    tariffs.groupby('date')['agile_price'].idxmax()
].reset_index(drop=True).rename(columns={'agile_price': 'highest_price', 'time': 'highest_time'})

daily_aggregate = cheapest_per_date[['date','cheapest_time','cheapest_price']].merge(
    highest_per_date[['date','highest_time','highest_price']], how='left', on='date')
daily_aggregate = daily_aggregate.merge(weighted_avg, how='left', on='date')


# ---------------------------
# Wind Generation Data Fetching
# ---------------------------
@st.cache_data(ttl=900)
def fetch_wind_forecast(now_utc: pd.Timestamp, days_ahead: int):
    end_time = now_utc + pd.Timedelta(days=days_ahead)

    sql_query = f"""SELECT * FROM "93c3048e-1dab-4057-a2a9-417540583929" WHERE "Date" >= '{now_utc.isoformat()}' AND "Date" <= '{end_time.isoformat()}' ORDER BY "_id" ASC"""

    params = {"sql": sql_query}

    try:
        response = requests.get(
            "https://api.neso.energy/api/3/action/datastore_search_sql",
            params=parse.urlencode(params)
        )
        response.raise_for_status()
        data = response.json()["result"]["records"]
        df = pd.DataFrame(data)

        if df.empty:
            return df

        df["forecast_datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df["forecast_datetime_local"] = df["forecast_datetime"].dt.tz_convert("Europe/London")

        # Ensure numeric
        df["Wind_Forecast"] = pd.to_numeric(
            df["Wind_Forecast"], errors="coerce"
        )

        return df.dropna(subset=["Wind_Forecast"])

    except Exception as e:
        st.warning(f"Failed to fetch wind forecast data: {e}")
        return pd.DataFrame()


# ---------------------------
# KPI Display
# ---------------------------
try:
    row_today = daily_aggregate[daily_aggregate['date'] == pd.to_datetime(date_today).date()].iloc[0]
except Exception:
    row_today = None
try:
    row_tomorrow = daily_aggregate[daily_aggregate['date'] == pd.to_datetime(date_tomorrow).date()].iloc[0]
except Exception:
    row_tomorrow = None

def fmt_time(t):
    if t is None or pd.isna(t): return "—"
    if isinstance(t, datetime.time): return t.strftime('%H:%M')
    try: return pd.to_datetime(t).time().strftime('%H:%M')
    except: return "—"

def fmt_price_time(row, price_key, time_key):
    if row is not None and price_key in row and time_key in row:
        return f"{row[price_key]:.2f} p/kWh @ {fmt_time(row[time_key])}"
    return "—"

def diff_indicator(value):
    if pd.isna(value): return "—"
    color = "green" if value < 0 else "red" if value > 0 else "gray"
    arrow = "▼" if value < 0 else "▲" if value > 0 else "•"
    sign = "+" if value > 0 else ""
    return f"<span style='color:{color}; font-weight:bold;'>{arrow} {sign}{value:.1f}%</span>"

def weighted_card_block(row, label):
    if row is None:
        return ""

    price = f"{row['weighted_agile_price']:.2f}"
    diff_yda = diff_indicator(row['relative_diff_to_yda'])
    diff_fixed = diff_indicator(row['relative_diff_to_fixed'])
    return f"""
    <div style='margin-top:0.5em;'>
        <div style='font-size:0.9em; color:#555;'>{label}</div>
        <div style='font-size:2.2em; font-weight:600; color:#000;'>{price} <span style='font-size:0.5em;'>p/kWh</span></div>
        <div style='display:flex; justify-content:center; gap:1em; font-size:0.9em;'>
            <div>{diff_yda}<br><span style='font-size:0.7em; color:#555;'>vs YDA</span></div>
            <div>{diff_fixed}<br><span style='font-size:0.7em; color:#555;'>vs Fixed</span></div>
        </div>
    </div>
    """

card_style = "background-color: #f0f0f0; color: #000; padding: 1.5em; border-radius: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: center; min-height: 180px; display: flex; flex-direction: column; justify-content: center;"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div style='{card_style}'><h4>Today</h4><p><strong>Lowest:</strong> {fmt_price_time(row_today, 'cheapest_price', 'cheapest_time')}</p><p><strong>Highest:</strong> {fmt_price_time(row_today, 'highest_price', 'highest_time')}</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='{card_style}'><h4>Tomorrow</h4><p><strong>Lowest:</strong> {fmt_price_time(row_tomorrow, 'cheapest_price', 'cheapest_time')}</p><p><strong>Highest:</strong> {fmt_price_time(row_tomorrow, 'highest_price', 'highest_time')}</p></div>", unsafe_allow_html=True)
#with col3:
#    st.markdown(f"<div style='{card_style}'><h4>Weighted Average</h4>{weighted_card_block(row_today, 'Today')}{weighted_card_block(row_tomorrow, 'Tomorrow')}</div>", unsafe_allow_html=True)
with col3:
    # Generate the inner HTML strings first
    content_today = weighted_card_block(row_today, 'Today')
    content_tomorrow = weighted_card_block(row_tomorrow, 'Tomorrow')

    st.markdown(f"""
    <div style='{card_style}'>
        <h4 style='margin-bottom:0;'>Weighted Average</h4>
        {content_today}
        {content_tomorrow}
    </div>
    """, unsafe_allow_html=True)


# Optimiser
optimal_start, optimal_end, optimal_cost = optimal_dishwasher_start_time(tariffs_visuals, DISHWASHER_CONSUMPTION, date_today, date_tomorrow, selected_finish_time)
with col4:
    if optimal_start:
        # Define UK Timezone
        uk_zone = ZoneInfo("Europe/London")
        
        # Convert UTC start/end times to UK local time (handles BST automatically)
        s_str = optimal_start.astimezone(uk_zone).strftime('%H:%M')
        e_str = optimal_end.astimezone(uk_zone).strftime('%H:%M')
        content = f"<h4>Dishwasher Optimiser</h4><p><strong>Start:</strong> {s_str}</p><p><strong>Finish:</strong> {e_str}</p><p><strong>Cost:</strong> {optimal_cost:.2f} p</p>"
    else:
        content = f"<h4>Dishwasher Optimiser</h4><p><strong>Status:</strong> <span style='color:red'>Unavailable</span></p><p style='font-size:0.8em'>{optimal_cost if optimal_cost else 'Check 23:00-07:00 data'}</p>"
    st.markdown(f"<div style='{card_style}'>{content}</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# Electricity Chart
# ---------------------------
st.markdown("### Future Agile Prices (from now onwards)")
tariffs_future = tariffs_visuals[tariffs_visuals['valid_from'] >= now_utc].reset_index(drop=True)
if not tariffs_future.empty:
    try:
        tariffs_future['valid_from_local'] = tariffs_future['valid_from'].dt.tz_convert('Europe/London')
    except:
        tariffs_future['valid_from_local'] = pd.to_datetime(tariffs_future['valid_from']).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    
    def color_group(v):
        if v < 15.0 : return '#00A86B'
        elif v < (fixed_price - standing_charge_diff_std): return 'green'
        elif v < high_threshold: return 'orange'
        else: return 'red'
    
    tariffs_future['color'] = tariffs_future['agile_price'].apply(color_group)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tariffs_future['valid_from_local'],
        y=tariffs_future['agile_price'],
        marker_color=tariffs_future['color'],
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Agile: %{y:.2f} p/kWh<extra></extra>'
    ))
    fig.add_hline(y=(fixed_price - standing_charge_diff_std), line_dash="dash", line_color="darkred", annotation_text=f"Fixed: {(fixed_price - standing_charge_diff_std):.1f} p")
    fig.update_layout(xaxis_title='Time', yaxis_title='Price (p/kWh)', bargap=0.1, height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No future electricity data available.")


# ---------------------------
# Wind Generation Forecast
# ---------------------------
st.markdown("---")
st.markdown("### Wind Generation Forecast")

with st.spinner("Fetching wind generation forecast..."):
    wind_df = fetch_wind_forecast(now_utc, wind_days_ahead)

if not wind_df.empty:
    if not wind_df.empty:
        wind_df = wind_df[
            wind_df["forecast_datetime"] >= now_utc
        ].copy()

    wind_df["forecast_datetime_local"] = wind_df["forecast_datetime"].dt.tz_convert("Europe/London")

    fig_wind = go.Figure()

    fig_wind.add_trace(go.Scatter(
        x=wind_df["forecast_datetime_local"],
        y=wind_df["Wind_Forecast"],
        mode="lines",
        line=dict(width=3),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Wind: %{y:.0f} MW<extra></extra>"
    ))

    fig_wind.update_layout(
        xaxis_title="Time",
        yaxis_title="Forecast Wind Generation (MW)",
        height=350,
        margin=dict(t=20, b=20),
    )

    add_daytime_shading(
        fig_wind,
        now_utc.tz_convert("Europe/London"),
        wind_df["forecast_datetime_local"].max(),
)

    st.plotly_chart(fig_wind, use_container_width=True)
else:
    st.info("No wind forecast data available for the selected period.")


# ---------------------------
# Gas Chart
# ---------------------------
st.markdown("---")
st.markdown("### Daily Tracker gas price (last 30 days)")

# Gas Dates: 30 days ago to Yesterday (as per snippet logic for 'to' date)
gas_date_start = (date.today() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
gas_period_from = f'{gas_date_start}T00:00'
gas_period_to = f'{date_yesterday}T23:30'

with st.spinner("Fetching gas tariffs..."):
    gas_raw = fetch_tariffs(DEFAULT_GAS_URL, gas_period_from, gas_period_to)

if not gas_raw.empty:
    # Gas trackers are usually daily (00:00 to 00:00). 
    # We group by date in case the API returns partial slots, taking the max price for the day.
    gas_raw['date_str'] = gas_raw['date'].astype(str)
    gas_daily = gas_raw.groupby('date_str').agg({
        'value_inc_vat': 'max',  # Use max to be safe/conservative, usually uniform for the day
        'valid_from': 'first'
    }).reset_index()
    gas_daily['date_obj'] = pd.to_datetime(gas_daily['date_str'])
    
    # Sort just in case
    gas_daily = gas_daily.sort_values('date_obj')

    # Assign Colors based on threshold
    # Dark blue (#00008B) if above threshold, Light blue (#87CEFA) if below
    gas_daily['color'] = np.where(
        gas_daily['value_inc_vat'] > gas_fixed_price,
        '#00008B', # Dark Blue
        '#87CEFA'  # Light Blue
    )

    fig_gas = go.Figure()
    fig_gas.add_trace(go.Bar(
        x=gas_daily['date_obj'],
        y=gas_daily['value_inc_vat'],
        marker_color=gas_daily['color'],
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Gas Price: %{y:.2f} p/kWh<extra></extra>'
    ))

    # Red dotted line for Fixed Gas Price
    fig_gas.add_hline(
        y=gas_fixed_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Fixed Gas: {gas_fixed_price:.2f} p",
        annotation_position="top right"
    )

    fig_gas.update_layout(
        xaxis_title='Date',
        yaxis_title='Price (p/kWh)',
        bargap=0.2,
        height=400,
        margin=dict(t=20, b=20),
        xaxis=dict(
            tickformat='%b-%d', # CHANGED: Displays 'Dec-13'
            tickangle=45
        )
    )

    st.plotly_chart(fig_gas, use_container_width=True)