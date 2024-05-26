import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# this generate 'ModuleNotFoundError'
# import japanize_matplotlib

def simulate_10y_cost_mycar(
        car_price: float, inspection_cost: float, option_insurance_cost: float, gas_price: np.array,
        car_tax: float, fuel_efficiency: float, mileage: float, parking_fare: float) -> list[float]:
    cost_10y = []
    total_mileage = 0
    thresh_engoil_replace = 0
    for year in range(1, 11):
        cost_10y.append(car_price if year == 1 else 0)
        cost_1y = 0
        total_mileage += mileage
        thresh_engoil_replace += mileage
        # 自動車税
        cost_1y += car_tax
        # ガソリン代
        cost_1y += (mileage / fuel_efficiency)*gas_price[year-1]
        # 駐車場代
        cost_1y += parking_fare*12
        # 5,000km走行ごとにエンジンオイル交換
        if thresh_engoil_replace >= 5000:
            thresh_engoil_replace -= 5000
            cost_1y += ENGOIL_PRICE

        # 車検
        if is_used_car:
            if year >= 2 and (year-2) % 2 == 0:
                cost_1y += inspection_cost
            if year >= 3:
                inspection_cost = inspection_cost*1.05
        else:
            if year >= 3 and (year-3) % 2 == 0:
                cost_1y += inspection_cost
            if year >= 3:
                inspection_cost = inspection_cost*1.05


        # 任意保険
        cost_1y += option_insurance_price
        # タイヤ交換
        if year == 5:
            cost_1y += TIRE_PRICE
        cost_10y[year-1] += cost_1y
    return cost_10y

# Define constants or functions needed for calculations
CAR_TYPES = ['Sedan', 'SUV', 'Truck']  # Example types, add more as needed
LIVING_AREAS = ['Urban', 'Suburban', 'Rural']  # Example areas, add more as needed
CAR_TAX_TABLE = pd.DataFrame([
    ['~1,000', 1000, 2.5],
    ['1,000 ~ 1,500', 1500, 3.05],
    ['1,500 ~ 2,000', 2000, 3.6],
    ['2,000 ~ 2,500', 2500, 4.35],
    ['2,500 ~ 3,000', 3000, 5.0]
    ], columns = ['disp_range', 'disp_cc', 'price'])
ENGOIL_PRICE = 0.5
TIRE_PRICE = 5




st.title('維持費シミュレーション')
with st.form(key='my_form'):
    car_price = st.number_input('購入価格（万円）:', min_value=100.0, format='%f')
    disp_cc = st.selectbox('排気量(cc)', CAR_TAX_TABLE['disp_range'])
    is_used_car = st.checkbox('中古車', key='used_car')
    fuel_efficiency = st.number_input('燃費（km/ℓ）', min_value=10.0, format='%f')
    gas_price_initial = 0.0001 * st.number_input('初期ガソリン代（円）', value=170.0, min_value=0.0)
    option_insurance_price = st.number_input('任意保険年額（万円）', value = 6.0, min_value=0.0, format='%f')
    mileage_per_year = st.number_input('想定年間走行距離（km）:', value = 1000.0, min_value=0.0, format='%f')
    parking_fare = st.number_input('駐車場月額（万円）:', min_value=0.0, format='%f')

    submitted = st.form_submit_button(label='シミュレーション')

def calc_car_tax(disp_cc: int) -> float:
    return CAR_TAX_TABLE['price'][disp_cc == CAR_TAX_TABLE['disp_range']][0]

# gas price fluctuations with Brownian motion
# These values (the mean and standard deviation of return of gas price)
# are calculated in another program
GAS_RETURN_MU = 0.002087169210979362 * 12
GAS_RETURN_STD = 0.03462509428964654 * np.sqrt(12) * 0.5
def bm_1step(x0, mu=GAS_RETURN_MU, sigma=GAS_RETURN_STD):
    x = x0 * (1 + mu + sigma * np.random.standard_normal(1))
    return x[0]
T = 10  # シミュレーション期間（年数）
N = 1000  # シミュレーションパスの数
S0 = gas_price_initial*10000
gas_10y_multi = np.zeros((T, N))
gas_10y_multi[0] = S0
# Brownian motion
np.random.seed(0)
for t in range(1, T):
    Z = np.random.standard_normal(N)
    gas_10y_multi[t] = gas_10y_multi[t-1] * (1 + GAS_RETURN_MU + GAS_RETURN_STD * Z)
gas_10y_multi = gas_10y_multi / 10000



if submitted:
    car_tax = calc_car_tax(disp_cc)
    inspection_cost = 7
    # Call the simulation function

    cost_10y = simulate_10y_cost_mycar(
        car_price = car_price, inspection_cost = inspection_cost,
        option_insurance_cost = option_insurance_price, gas_price = np.array([gas_price_initial]*10),
        car_tax = car_tax, fuel_efficiency = fuel_efficiency, mileage = mileage_per_year,
        parking_fare = parking_fare)

    # Visualize the results
    df = pd.DataFrame({'Year': range(1, 11), 'cost': cost_10y})

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
    axes[0].bar(np.arange(1, 11)-0.2, df['cost'], alpha=0.8, label='車購入', width=0.4, color='skyblue')
    axes[0].plot(np.arange(1, 11), df['cost'], alpha=0.8, color='skyblue')
    axes[0].set_title('Annual costs')
    # axes[0].set_ylim(0, 40)

    axes[1].bar(np.arange(1, 11)-0.2, np.cumsum(df['cost']), alpha=0.8, width=0.4, color='skyblue')
    axes[1].plot(np.arange(1, 11), np.cumsum(df['cost']), alpha=0.8, color='skyblue')
    # fig.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.05))
    axes[1].set_title('Cumulative costs')

    axes[1].set_xticks(np.arange(1, 11))
    axes[1].set_xticklabels(np.arange(1, 11))

    fig.supxlabel('Year')
    fig.supylabel('cost (10 thousand)')
    st.pyplot(fig)

    st.markdown('## ガソリン価格の変動を加味したシミュレーション')
    np.random.seed(0)
    cost_10y_multi = np.array(
        [simulate_10y_cost_mycar(
            car_price = car_price, inspection_cost = inspection_cost,
            option_insurance_cost = option_insurance_price, gas_price = gas_10y_multi[:, i],
            car_tax = car_tax, fuel_efficiency = fuel_efficiency, mileage = mileage_per_year,
            parking_fare = parking_fare) for i in range(N)])
    cost_10y_multi = cost_10y_multi.T
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    gas_10y_multi = gas_10y_multi * 10000

    axes[0].plot(np.arange(1, 11), gas_10y_multi, color = 'black', alpha=0.05)
    mean_gas = gas_10y_multi.mean(axis=1)
    # CI are too small so omitted this.
    # std_error_gas = gas_10y_multi.std(axis=1) / np.sqrt(gas_10y_multi.shape[1])
    # confidence_interval_gas = 1.96 * std_error_gas  # 95%信頼区間のための1.96
    # upper_bound_gas = mean_gas + confidence_interval_gas
    # lower_bound_gas = mean_gas - confidence_interval_gas

    # # 95%信頼区間のプロット
    # axes[0].fill_between(np.arange(1,11), lower_bound_gas, upper_bound_gas, color='blue', alpha=0.1)
    # axes[0].plot(np.arange(1,11), upper_bound_gas, color='blue', linewidth=2, linestyle='--', label='95% Confidence Interval')
    # axes[0].plot(np.arange(1,11), lower_bound_gas, color='blue', linewidth=2, linestyle='--')
    axes[0].plot(np.arange(1, 11), mean_gas.T, linewidth=3, color='red', label='Average')

    axes[0].set_title('gas price simulation with Brownian motion')


    axes[1].plot(np.arange(1, 11), cost_10y_multi, linewidth=1, color='black', alpha=0.05)
    # 各系列の平均を計算し、太い線でプロット

    mean_series = cost_10y_multi.mean(axis=1)
    # 95%信頼区間の計算
    std_error = cost_10y_multi.std(axis=1) / np.sqrt(cost_10y_multi.shape[1])
    confidence_interval = 1.96 * std_error  # 95%信頼区間のための1.96
    upper_bound = mean_series + confidence_interval
    lower_bound = mean_series - confidence_interval

    # # 95%信頼区間のプロット
    # axes[1].fill_between(np.arange(1,11), lower_bound, upper_bound, color='skyblue', alpha=0.2)
    # axes[1].plot(np.arange(1,11), upper_bound, color='blue', linewidth=2, linestyle='--', label='95% Confidence Interval')
    # axes[1].plot(np.arange(1,11), lower_bound, color='blue', linewidth=2, linestyle='--')

    axes[1].plot(np.arange(1, 11), mean_series.T, linewidth=3, color='red', label='Average')

    # グラフの装飾
    axes[1].set_title('with gasoline fluctuations')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('cost (10 thousand)')
    axes[1].legend()
    st.pyplot(fig)