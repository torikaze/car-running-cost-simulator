import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

@st.cache_data
def simulate_cost_mycar(
        car_price: float, inspection_cost: float, option_insurance_price: float, gas_price: np.array,
        car_tax: float, fuel_efficiency: float, mileage: float, parking_fare: float, is_used_car: bool, 
        engoil_price: float, tire_price: float, insurance_rate: float, engoil_interval: float, tire_interval: int, 
        simulation_years: int) -> dict:
    cost_breakdown = {
        'year': [],
        'car_price': [],
        'car_tax': [],
        'gas_cost': [],
        'parking_fee': [],
        'engoil_cost': [],
        'inspection_cost': [],
        'insurance_cost': [],
        'tire_cost': [],
        'total_cost': []
    }
    
    total_mileage = 0
    thresh_engoil_replace = 0
    inspection_cost_current = inspection_cost
    insurance_price_current = option_insurance_price
    
    for year in range(1, simulation_years + 1):
        total_mileage += mileage
        thresh_engoil_replace += mileage
        
        # 各項目のコスト初期化
        year_car_price = car_price if year == 1 else 0
        year_car_tax = car_tax
        year_gas_cost = (mileage / fuel_efficiency) * gas_price[year-1]
        year_parking_fee = parking_fare * 12
        year_engoil_cost = 0
        year_inspection_cost = 0
        year_insurance_cost = insurance_price_current
        year_tire_cost = 0
        
        # エンジンオイル交換（指定km毎）
        if thresh_engoil_replace >= engoil_interval:
            thresh_engoil_replace -= engoil_interval
            year_engoil_cost = engoil_price

        # 車検費用
        if is_used_car:
            if year >= 3 and (year-3) % 2 == 0:
                year_inspection_cost = inspection_cost_current
            if year >= 3:
                inspection_cost_current = inspection_cost_current * 1.05
        else:
            if year >= 4 and (year-4) % 2 == 0:
                year_inspection_cost = inspection_cost_current
            if year >= 4:
                inspection_cost_current = inspection_cost_current * 1.05
        # 保険料の増減
        if year > 1:
            insurance_price_current = insurance_price_current * (1 + insurance_rate / 100)

        # タイヤ交換（指定年数毎）
        if year > 1 and (year - 1) % tire_interval == 0:
            year_tire_cost = tire_price
        
        # 年間総コスト
        year_total = (year_car_price + year_car_tax + year_gas_cost + year_parking_fee + 
                     year_engoil_cost + year_inspection_cost + year_insurance_cost + year_tire_cost)
        
        # 結果を記録
        cost_breakdown['year'].append(year)
        cost_breakdown['car_price'].append(year_car_price)
        cost_breakdown['car_tax'].append(year_car_tax)
        cost_breakdown['gas_cost'].append(year_gas_cost)
        cost_breakdown['parking_fee'].append(year_parking_fee)
        cost_breakdown['engoil_cost'].append(year_engoil_cost)
        cost_breakdown['inspection_cost'].append(year_inspection_cost)
        cost_breakdown['insurance_cost'].append(year_insurance_cost)
        cost_breakdown['tire_cost'].append(year_tire_cost)
        cost_breakdown['total_cost'].append(year_total)
    
    return cost_breakdown

def calc_car_tax(disp_cc: str, car_tax_table: pd.DataFrame) -> float:
    return car_tax_table['price'][disp_cc == car_tax_table['disp_range']].iloc[0]

def simulate_gas_prices(gas_price_initial: float, years: int, n_simulations: int) -> np.array:
    """Simulate gas prices using Brownian motion"""
    gas_return_mu = 0.002087169210979362 * 12
    gas_return_std = 0.03462509428964654 * np.sqrt(12) * 0.5
    
    s0 = gas_price_initial * 10000
    gas_multi = np.zeros((years, n_simulations))
    gas_multi[0] = s0
    
    np.random.seed(0)
    for t in range(1, years):
        z = np.random.standard_normal(n_simulations)
        gas_multi[t] = gas_multi[t-1] * (1 + gas_return_mu + gas_return_std * z)
    
    return gas_multi / 10000

def calculate_cost_simulations(car_price: float, inspection_cost: float, option_insurance_price: float,
                             gas_prices: np.array, car_tax: float, fuel_efficiency: float, 
                             mileage_per_year: float, parking_fare: float, is_used_car: bool,
                             engoil_price: float, tire_price: float, insurance_rate: float, 
                             engoil_interval: float, tire_interval: int, simulation_years: int, n_simulations: int) -> np.array:
    """Simulate costs for multiple gas price scenarios"""
    np.random.seed(0)
    cost_multi = np.array([
        simulate_cost_mycar(
            car_price=car_price, inspection_cost=inspection_cost,
            option_insurance_price=option_insurance_price, gas_price=gas_prices[:, i],
            car_tax=car_tax, fuel_efficiency=fuel_efficiency, mileage=mileage_per_year,
            parking_fare=parking_fare, is_used_car=is_used_car, 
            engoil_price=engoil_price, tire_price=tire_price, insurance_rate=insurance_rate,
            engoil_interval=engoil_interval, tire_interval=tire_interval, simulation_years=simulation_years
        )['total_cost'] for i in range(n_simulations)
    ])
    return cost_multi.T

def create_annual_cost_chart(df: pd.DataFrame) -> go.Figure:
    """Create chart for annual costs"""
    fig = go.Figure()
    
    # 積み上げ棒グラフのためのデータ準備
    cost_categories = [
        ('車両購入費', 'car_price', 'rgb(31, 119, 180)'),      
        ('自動車税', 'car_tax', 'rgb(255, 127, 14)'),          
        ('ガソリン代', 'gas_cost', 'rgb(44, 160, 44)'),        
        ('駐車場代', 'parking_fee', 'rgb(214, 39, 40)'),       
        ('オイル交換費', 'engoil_cost', 'rgb(148, 103, 189)'), 
        ('車検費', 'inspection_cost', 'rgb(140, 86, 75)'),     
        ('保険料', 'insurance_cost', 'rgb(227, 119, 194)'),    
        ('タイヤ交換費', 'tire_cost', 'rgb(127, 127, 127)')    
    ]
    
    for name, column, color in cost_categories:
        fig.add_trace(go.Bar(
            x=df['year'],
            y=df[column],
            name=name,
            marker_color=color,
            hovertemplate=f'{name}: %{{y:.2f}} 万円<extra></extra>'
        ))
    
    fig.update_layout(
        title='年間費用',
        xaxis_title='年',
        yaxis_title='費用（万円）',
        barmode='stack',
        hovermode='x unified'
    )
    return fig

def create_cumulative_cost_chart(df: pd.DataFrame) -> go.Figure:
    """Create chart for cumulative costs"""
    fig = go.Figure()
    
    # 累積費用の内訳計算
    cumulative_df = df.copy()
    cost_columns = ['car_price', 'car_tax', 'gas_cost', 'parking_fee', 
                   'engoil_cost', 'inspection_cost', 'insurance_cost', 'tire_cost']
    
    for col in cost_columns:
        cumulative_df[f'cumulative_{col}'] = cumulative_df[col].cumsum()
    
    # 積み上げ棒グラフのためのデータ準備
    cost_categories = [
        ('車両購入費', 'cumulative_car_price', 'rgb(255, 99, 132)'),
        ('自動車税', 'cumulative_car_tax', 'rgb(54, 162, 235)'),
        ('ガソリン代', 'cumulative_gas_cost', 'rgb(255, 205, 86)'),
        ('駐車場代', 'cumulative_parking_fee', 'rgb(75, 192, 192)'),
        ('オイル交換費', 'cumulative_engoil_cost', 'rgb(153, 102, 255)'),
        ('車検費', 'cumulative_inspection_cost', 'rgb(255, 159, 64)'),
        ('保険料', 'cumulative_insurance_cost', 'rgb(255, 99, 255)'),
        ('タイヤ交換費', 'cumulative_tire_cost', 'rgb(199, 199, 199)')
    ]
    
    for name, column, color in cost_categories:
        fig.add_trace(go.Bar(
            x=cumulative_df['year'],
            y=cumulative_df[column],
            name=name,
            marker_color=color,
            hovertemplate=f'{name}累積: %{{y:.2f}} 万円<extra></extra>'
        ))
    
    fig.update_layout(
        title='累積費用',
        xaxis_title='年',
        yaxis_title='費用（万円）',
        barmode='stack',
        hovermode='x unified'
    )
    return fig

def create_gas_price_chart(gas_prices: np.array, simulation_years: int, n_display: int = 100) -> go.Figure:
    """Create chart for gas price fluctuations"""
    mean_gas = gas_prices.mean(axis=1)
    std_error_gas = gas_prices.std(axis=1)
    confidence_interval_gas = 1.96 * std_error_gas
    upper_bound_gas = mean_gas + confidence_interval_gas
    lower_bound_gas = mean_gas - confidence_interval_gas

    fig = go.Figure()
    # Draw paths (limit display count)
    for i in range(min(n_display, gas_prices.shape[1])):
        fig.add_trace(go.Scatter(
            x=np.arange(1, simulation_years + 1),
            y=gas_prices[:, i],
            mode='lines',
            line=dict(color='black', width=1),
            opacity=0.05,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Mean line
    fig.add_trace(go.Scatter(
        x=np.arange(1, simulation_years + 1),
        y=mean_gas,
        mode='lines+markers',
        name='平均',
        line=dict(color='red', width=3),
        hovertemplate='%{x}年目<br>平均ガソリン価格: %{y:.0f} 円'
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=np.arange(1, simulation_years + 1),
        y=upper_bound_gas,
        mode='lines',
        name='95% 信頼区間上限',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate='%{x}年目<br>上限: %{y:.0f} 円'
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(1, simulation_years + 1),
        y=lower_bound_gas,
        mode='lines',
        name='95% 信頼区間下限',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate='%{x}年目<br>下限: %{y:.0f} 円',
        fill='tonexty',
        fillcolor='rgba(0,0,255,0.1)'
    ))
    
    fig.update_layout(
        title='ガソリン価格の変動シミュレーション',
        xaxis_title='年',
        yaxis_title='ガソリン価格（円）',
    )
    return fig

def create_cost_fluctuation_chart(cost_multi: np.array, simulation_years: int, y_range: tuple, n_display: int = 100) -> go.Figure:
    """Create chart for cost fluctuations"""
    mean_series = cost_multi.mean(axis=1)
    std_error = cost_multi.std(axis=1)
    confidence_interval = 1.96 * std_error
    upper_bound = mean_series + confidence_interval
    lower_bound = mean_series - confidence_interval

    fig = go.Figure()
    # Draw paths
    for i in range(min(n_display, cost_multi.shape[1])):
        fig.add_trace(go.Scatter(
            x=np.arange(1, simulation_years + 1),
            y=cost_multi[:, i],
            mode='lines',
            line=dict(color='black', width=1),
            opacity=0.05,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Mean line
    fig.add_trace(go.Scatter(
        x=np.arange(1, simulation_years + 1),
        y=mean_series,
        mode='lines+markers',
        name='平均',
        line=dict(color='red', width=3),
        hovertemplate='%{x}年目<br>平均費用: %{y:.2f} 万円'
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=np.arange(1, simulation_years + 1),
        y=upper_bound,
        mode='lines',
        name='95% 信頼区間上限',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate='%{x}年目<br>上限: %{y:.2f} 万円'
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(1, simulation_years + 1),
        y=lower_bound,
        mode='lines',
        name='95% 信頼区間下限',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate='%{x}年目<br>下限: %{y:.2f} 万円',
        fill='tonexty',
        fillcolor='rgba(135,206,235,0.2)'
    ))
    
    fig.update_layout(
        title='ガソリン価格の変動を考慮した年間費用',
        xaxis_title='年',
        yaxis_title='費用(万円)',
        yaxis_range=y_range,
    )
    return fig

def setup_ui_components():
    """Setup UI components and get user inputs"""
    # Constants
    car_tax_table = pd.DataFrame([
        ['~1,000', 1000, 2.5],
        ['1,000 ~ 1,500', 1500, 3.05],
        ['1,500 ~ 2,000', 2000, 3.6],
        ['2,000 ~ 2,500', 2500, 4.35],
        ['2,500 ~ 3,000', 3000, 5.0]
    ], columns=['disp_range', 'disp_cc', 'price'])

    # Title and detailed settings
    st.title('車の維持費シミュレーション')

    # Sidebar inputs
    st.sidebar.header('シミュレーション設定')
    simulation_years = st.sidebar.number_input('シミュレーション期間（年）:', min_value=1, max_value=20, value=10, step=1)
    car_price = st.sidebar.number_input('購入価格（万円）:', min_value=0.0, format='%f')
    disp_cc = st.sidebar.selectbox('排気量(cc)', car_tax_table['disp_range'])
    is_used_car = st.sidebar.checkbox('中古車', key='used_car')
    fuel_efficiency = st.sidebar.number_input('燃費（km/ℓ）', min_value=10.0, format='%f')
    gas_price_initial = 0.0001 * st.sidebar.number_input('初期ガソリン代（円）', value=170.0, min_value=0.0)
    option_insurance_price = st.sidebar.number_input('任意保険年額（万円）', value=6.0, min_value=0.0, format='%f')
    mileage_per_year = st.sidebar.number_input('想定年間走行距離（km）:', value=1000.0, min_value=0.0, format='%f')
    parking_fare = st.sidebar.number_input('駐車場月額（万円）:', min_value=0.0, format='%f')
    inspection_cost = st.sidebar.number_input('初回車検費用（万円）:', value=7.0, min_value=0.0, format='%f')

    # 詳細設定のexpander
    with st.sidebar.expander("詳細設定"):
        engoil_price = st.number_input('エンジンオイル交換費（万円）:', value=0.5, min_value=0.0, format='%f')
        engoil_interval = st.number_input('エンジンオイル交換距離（km）:', value=5000.0, min_value=1000.0, format='%f')
        tire_price = st.number_input('タイヤ交換費（万円）:', value=5.0, min_value=0.0, format='%f')
        tire_interval = st.number_input('タイヤ交換頻度（年）:', value=5, min_value=1, max_value=10)
        insurance_rate = st.number_input('保険料増減率（%）', min_value=-50.0, max_value=50.0, value=0.0, step=0.5)

    with st.expander("シミュレーションの補足"):
        st.markdown(
            f'''
            - 普通車を想定
            - 経年劣化による部品交換・修繕費用は、車検費用を増加させることで考慮
            '''
        )

    return {
        'car_tax_table': car_tax_table,
        'simulation_years': simulation_years,
        'engoil_price': engoil_price,
        'engoil_interval': engoil_interval,
        'tire_price': tire_price,
        'tire_interval': tire_interval,
        'inspection_cost': inspection_cost,
        'car_price': car_price,
        'disp_cc': disp_cc,
        'is_used_car': is_used_car,
        'fuel_efficiency': fuel_efficiency,
        'gas_price_initial': gas_price_initial,
        'option_insurance_price': option_insurance_price,
        'insurance_rate': insurance_rate,
        'mileage_per_year': mileage_per_year,
        'parking_fare': parking_fare
    }

def main():
    """Main processing"""
    # Get UI settings and user inputs
    config = setup_ui_components()
    
    # Simulation settings
    T = config['simulation_years']  # Simulation period (years)
    N = 2000  # Number of simulations
    
    # Simulate gas prices
    gas_multi = simulate_gas_prices(config['gas_price_initial'], T, N)
    
    # Calculate car tax
    car_tax = calc_car_tax(config['disp_cc'], config['car_tax_table'])
    
    # Basic cost calculation (fixed gas price)
    cost_breakdown = simulate_cost_mycar(
        car_price=config['car_price'], 
        inspection_cost=config['inspection_cost'],
        option_insurance_price=config['option_insurance_price'], 
        gas_price=np.array([config['gas_price_initial']] * T),
        car_tax=car_tax, 
        fuel_efficiency=config['fuel_efficiency'], 
        mileage=config['mileage_per_year'],
        parking_fare=config['parking_fare'],
        is_used_car=config['is_used_car'],
        engoil_price=config['engoil_price'],
        tire_price=config['tire_price'],
        insurance_rate=config['insurance_rate'],
        engoil_interval=config['engoil_interval'],
        tire_interval=config['tire_interval'],
        simulation_years=T
    )
    
    # Cost calculation with fluctuating gas prices
    cost_multi = calculate_cost_simulations(
        config['car_price'], config['inspection_cost'], config['option_insurance_price'],
        gas_multi, car_tax, config['fuel_efficiency'], config['mileage_per_year'],
        config['parking_fare'], config['is_used_car'], config['engoil_price'], 
        config['tire_price'], config['insurance_rate'], config['engoil_interval'],
        config['tire_interval'], T, N
    )
    
    # Save results to session state
    st.session_state['result'] = cost_breakdown['total_cost']

    if st.session_state['result']:
        # Create dataframe
        df = pd.DataFrame(cost_breakdown)
        df['cumulative_cost'] = df['total_cost'].cumsum()

        # Display basic charts
        fig1 = create_annual_cost_chart(df)
        fig2 = create_cumulative_cost_chart(df)
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("#### テーブル表示")
        
        # コスト内訳の表示用データフレーム作成
        display_df = df.copy()
        display_df = display_df.rename(columns={
            'year': '年',
            'car_price': '車両購入費',
            'car_tax': '自動車税',
            'gas_cost': 'ガソリン代',
            'parking_fee': '駐車場代',
            'engoil_cost': 'オイル交換費',
            'inspection_cost': '車検費',
            'insurance_cost': '保険料',
            'tire_cost': 'タイヤ交換費',
            'total_cost': '年間費用',
            'cumulative_cost': '累積費用'
        })
        st.dataframe(display_df, height=300)

        # Gas price fluctuation simulation
        st.markdown('## ガソリン価格の変動を加味したシミュレーション')
        with st.expander("補足"):
            st.markdown(
                '''
                - 変動要因は2004年1月~2024年3月の月次レギュラーガソリン価格の変化率の平均と標準偏差を使用
                - ブラウン運動によってガソリン価格のばらつきを表現
                '''
            )
        gas_multi_display = gas_multi * 10000

        # Gas price fluctuation chart
        fig_gas = create_gas_price_chart(gas_multi_display, T)
        st.plotly_chart(fig_gas, use_container_width=True)

        # Cost fluctuation chart
        # Display range slider
        y_min_annualcost, y_max_annualcost = st.slider(
            '縦軸の表示範囲', 
            float(np.min(cost_multi) * 0.9), 
            float(np.max(cost_multi) * 1.1), 
            (float(np.min(cost_multi) * 0.9), float(np.max(cost_multi) * 1.1)), 
            0.5
        )
        fig_cost = create_cost_fluctuation_chart(
            cost_multi, 
            T,
            (y_min_annualcost, y_max_annualcost)
        )
        st.plotly_chart(fig_cost, use_container_width=True)

if __name__ == "__main__":
    main()