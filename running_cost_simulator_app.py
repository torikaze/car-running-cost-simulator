import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import japanize_matplotlib


def simulate_10y_cost_mycar(
        car_price: float, inspection_cost: float, option_insurance_cost: float, gas_price: float,
        car_tax: float, fuel_efficiency: float, mileage: float, parking_fare: float) -> list[float]:
    cost_10y = []
    total_mileage = 0
    thresh_engoil_replace = 0
    new_gas_price = gas_price
    for year in range(1, 11):
        cost_10y.append(car_price if year == 1 else 0)
        cost_1y = 0
        total_mileage += mileage
        thresh_engoil_replace += mileage
        # 自動車税
        cost_1y += car_tax
        # ガソリン代
        cost_1y += (mileage / fuel_efficiency)*new_gas_price
        # new_gas_price = gbm_1step(mu=mu, sigma=sigma, x0=new_gas_price*10000) / 10000

        # 駐車場代
        cost_1y += parking_fare*12
        # 5,000km走行ごとにエンジンオイル交換
        if thresh_engoil_replace >= 5000:
            thresh_engoil_replace -= 5000
            cost_1y += ENGOIL_PRICE

        # 車検
        if is_used_car:
            if year >= 3 and (year-3) % 2 == 0:
                cost_1y += inspection_cost
            if year >= 3:
                inspection_cost = inspection_cost*1.05
        else:
            if year >= 2 and (year-2) % 2 == 0:
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
    gas_price_initial = 0.001 * st.number_input('初期ガソリン代（円）', value=170.0, min_value=0.0)
    option_insurance_price = st.number_input('任意保険年額（万円）', value = 6.0, min_value=0.0, format='%f')
    mileage_per_year = st.number_input('想定年間走行距離（km）:', value = 1000.0, min_value=0.0, format='%f')
    parking_fare = st.number_input('駐車場代（万円）:', min_value=0.0, format='%f')

    submitted = st.form_submit_button(label='シミュレーション')

def calc_car_tax(disp_cc: int) -> float:
    return CAR_TAX_TABLE['price'][disp_cc == CAR_TAX_TABLE['disp_range']][0]


if submitted:
    car_tax = calc_car_tax(disp_cc)
    inspection_cost = 7
    # Call the simulation function
    cost_10y = simulate_10y_cost_mycar(
        car_price = car_price, inspection_cost = inspection_cost,
        option_insurance_cost = option_insurance_price, gas_price = gas_price_initial,
        car_tax = car_tax, fuel_efficiency = fuel_efficiency, mileage = mileage_per_year,
        parking_fare = parking_fare)

    # Visualize the results
    df = pd.DataFrame({'Year': range(1, 11), 'cost': cost_10y})
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
    axes[0].bar(np.arange(1, 11)-0.2, df['cost'], alpha=0.8, label='車購入', width=0.4, color='skyblue')
    axes[0].plot(np.arange(1, 11), df['cost'], alpha=0.8, color='skyblue')
    axes[0].set_title('毎年の費用')
    # axes[0].set_ylim(0, 40)

    axes[1].bar(np.arange(1, 11)-0.2, np.cumsum(df['cost']), alpha=0.8, width=0.4, color='skyblue')
    axes[1].plot(np.arange(1, 11), np.cumsum(df['cost']), alpha=0.8, color='skyblue')
    # fig.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.05))
    axes[1].set_title('毎年の累積費用')

    axes[1].set_xticks(np.arange(1, 11))
    axes[1].set_xticklabels(np.arange(1, 11))

    fig.supxlabel('年数')
    fig.supylabel('費用（万円）')
    st.pyplot(fig)