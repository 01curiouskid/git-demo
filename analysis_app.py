# --- 1. IMPORTS AND SETUP ---
import streamlit as st
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import io
from contextlib import redirect_stdout

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Commodity Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. DATABASE CONNECTION (with Caching) ---
# --- 3. DATABASE CONNECTION (with Caching) ---
@st.cache_resource
def get_db_engine():
    """Establishes and returns a SQLAlchemy engine, cached for performance."""
    try:
        # Use st.secrets to get the credentials
        DB_HOST = st.secrets["database"]["host"]
        DB_PORT = st.secrets["database"]["port"]
        DB_NAME = st.secrets["database"]["dbname"]
        DB_USER = st.secrets["database"]["user"]
        DB_PASS = st.secrets["database"]["password"]

        encoded_pass = quote_plus(DB_PASS)
        db_url = f"postgresql+psycopg2://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)
        with engine.connect() as conn: pass # Test connection
        return engine
    except Exception as e:
        st.error(f"Database connection failed. Please check your secrets configuration. Error: {e}")
        return None

engine = get_db_engine()

# --- 4. CACHED DATA FETCHING FUNCTIONS (Corrected) ---
# FIX: These functions now return the raw, un-stripped names to match the working Jupyter code's logic.
@st.cache_data
def get_states(_engine):
    df = pd.read_sql("SELECT DISTINCT state_name FROM commodities_market_analysis ORDER BY state_name;", _engine)
    return df['state_name'].tolist()

@st.cache_data
def get_districts(_engine, state):
    query = "SELECT DISTINCT district_name FROM commodities_market_analysis WHERE state_name = %(state)s ORDER BY district_name;"
    df = pd.read_sql(query, _engine, params={'state': state})
    return df['district_name'].tolist()

@st.cache_data
def get_markets(_engine, district):
    query = "SELECT DISTINCT market_name FROM commodities_market_analysis WHERE district_name = %(district)s ORDER BY market_name;"
    df = pd.read_sql(query, _engine, params={'district': district})
    return df['market_name'].tolist()

@st.cache_data
def get_commodities(_engine, market):
    query = "SELECT DISTINCT commodity FROM commodities_market_analysis WHERE market_name = %(market)s ORDER BY commodity;"
    df = pd.read_sql(query, _engine, params={'market': market})
    return df['commodity'].tolist()

# --- 5. ANALYSIS FUNCTIONS ---

def perform_exploratory_data_analysis(df):
    st.header("Phase 2: Exploratory Data Analysis", divider='rainbow')
    price_cols = sorted([col for col in df.columns if 'price' in col])
    arrival_cols = sorted([col for col in df.columns if 'arrival' in col and 'is_imputed' not in col])
    
    st.subheader("Modal Price Over Time")
    price_fig = go.Figure()
    for col in price_cols:
        market_name = col.replace('price_', '').title()
        price_fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'{market_name} Price'))
    price_fig.update_layout(legend_title='Markets', hovermode='x unified')
    st.plotly_chart(price_fig, use_container_width=True)

    st.subheader("Arrivals Over Time (Imputed)")
    arrival_fig = go.Figure()
    for col in arrival_cols:
        market_name = col.replace('arrival_', '').title()
        arrival_fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'{market_name} Arrival'))
    arrival_fig.update_layout(legend_title='Markets', hovermode='x unified')
    st.plotly_chart(arrival_fig, use_container_width=True)
    
    st.subheader("Daily Arrival vs. Yearly Average")
    ratio_fig = go.Figure()
    years = sorted(df.index.year.unique())
    if not years: return
    table_data_list = []
    all_yearly_ratios = {}
    for year in years:
        all_yearly_ratios[year] = {}
        df_year = df.loc[df.index.year == year]
        if df_year.empty: continue
        for col in arrival_cols:
            yearly_avg = df_year[col].mean()
            daily_ratio = df_year[col] / yearly_avg if yearly_avg > 0 else df_year[col] * 0
            all_yearly_ratios[year][col] = daily_ratio
            imputed_col_name = col.replace('arrival_', 'is_imputed_')
            if imputed_col_name in df_year.columns:
                temp_df = pd.DataFrame({'year': year, 'market': col.replace('arrival_', ''), 'reported_date': df_year.index,'arrival': df_year[col], 'isImputed': df_year[imputed_col_name],'yearly_average': yearly_avg, 'ratio': daily_ratio})
                table_data_list.append(temp_df)
    initial_year = years[0]
    for col in arrival_cols:
        if col in all_yearly_ratios.get(initial_year, {}):
            ratio_fig.add_trace(go.Scatter(x=all_yearly_ratios[initial_year][col].index, y=all_yearly_ratios[initial_year][col].values, mode='lines', name=col.replace('arrival_', '').title()))
    buttons = []
    for year in years:
        x_data = [all_yearly_ratios.get(year, {}).get(col, pd.Series(dtype='datetime64[ns]')).index for col in arrival_cols]
        y_data = [all_yearly_ratios.get(year, {}).get(col, pd.Series(dtype='float64')).values for col in arrival_cols]
        buttons.append(dict(method='update', label=str(year), args=[{'x': x_data, 'y': y_data}]))
    ratio_fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, direction="down", showactive=True, x=0.1, xanchor="left", y=1.15, yanchor="top")], legend_title='Markets', hovermode='x unified')
    ratio_fig.add_hline(y=1.0, line_width=2, line_dash="dash", line_color="red", annotation_text="Yearly Average")
    st.plotly_chart(ratio_fig, use_container_width=True)
    if table_data_list:
        st.dataframe(pd.concat(table_data_list, ignore_index=True))

def perform_correlation_analysis(df, target_market_name):
    st.header("Phase 3: Correlation Analysis", divider='rainbow')
    cols_for_corr = [col for col in df.columns if 'is_imputed' not in col]
    corr_matrix = df[cols_for_corr].corr()
    target_price_col = f'price_{target_market_name}'
    st.subheader(f"Correlation with Target Market Price ({target_market_name.title()})")
    if target_price_col in corr_matrix:
        st.dataframe(corr_matrix[[target_price_col]].sort_values(by=target_price_col, ascending=False))
    st.subheader("Full Correlation Heatmap")
    heatmap_fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu_r', zmin=-1, zmax=1, text=corr_matrix.round(2).values, texttemplate="%{text}", textfont={"size":8}))
    heatmap_fig.update_layout(width=800, height=800, xaxis_tickangle=-45)
    st.plotly_chart(heatmap_fig, use_container_width=True)

def perform_lag_analysis(df, target_market_name, source_market_names):
    st.header("Phase 4: Lag & Cross-Correlation Analysis", divider='rainbow')
    target_variable = f'price_{target_market_name}'
    if target_variable not in df.columns:
        st.error(f"Target variable '{target_variable}' not found. Skipping lag analysis.")
        return
    lags_to_check = range(-21, 22)
    st.subheader("Price Cross-Correlations")
    valid_sources_p = [m for m in source_market_names if f'price_{m}' in df.columns]
    if valid_sources_p:
        price_ccf_fig = make_subplots(rows=len(valid_sources_p), cols=1, subplot_titles=[f"Price in {m.title()}" for m in valid_sources_p])
        for i, market in enumerate(valid_sources_p, 1):
            source_variable = f'price_{market}'
            ccf_values = [df[target_variable].corr(df[source_variable].shift(lag)) for lag in lags_to_check]
            if all(np.isnan(ccf_values)): continue
            max_idx = np.nanargmax(np.abs(ccf_values))
            colors = ['cornflowerblue'] * len(ccf_values); colors[max_idx] = 'orange'
            price_ccf_fig.add_trace(go.Bar(x=list(lags_to_check), y=ccf_values, name=market, marker_color=colors), row=i, col=1)
        price_ccf_fig.update_layout(title_text=f"<b>Source Prices vs. Target Price ({target_market_name.title()})</b>", showlegend=False, height=250*len(valid_sources_p))
        st.plotly_chart(price_ccf_fig, use_container_width=True)
    st.subheader("Arrival Cross-Correlations")
    valid_sources_a = [m for m in source_market_names if f'arrival_{m}' in df.columns]
    if valid_sources_a:
        arrival_ccf_fig = make_subplots(rows=len(valid_sources_a), cols=1, subplot_titles=[f"Arrival in {m.title()}" for m in valid_sources_a])
        for i, market in enumerate(valid_sources_a, 1):
            source_variable = f'arrival_{market}'
            ccf_values = [df[target_variable].corr(df[source_variable].shift(lag)) for lag in lags_to_check]
            if all(np.isnan(ccf_values)): continue
            max_idx = np.nanargmax(np.abs(ccf_values))
            colors = ['cornflowerblue'] * len(ccf_values); colors[max_idx] = 'orange'
            arrival_ccf_fig.add_trace(go.Bar(x=list(lags_to_check), y=ccf_values, name=market, marker_color=colors), row=i, col=1)
        arrival_ccf_fig.update_layout(title_text=f"<b>Source Arrivals vs. Target Price ({target_market_name.title()})</b>", showlegend=False, height=250*len(valid_sources_a))
        st.plotly_chart(arrival_ccf_fig, use_container_width=True)

def perform_seasonality_analysis(df):
    st.header("Phase 5: Seasonality & Cycle Decomposition (FFT)", divider='rainbow')
    price_cols = [col for col in df.columns if 'price' in col]
    arrival_cols = [col for col in df.columns if 'arrival' in col and 'is_imputed' not in col]
    def create_fft_plot(cols, data_type):
        st.subheader(f"FFT Power Spectrum for {data_type.title()} (Entire Dataset)")
        fig = go.Figure()
        for col in cols:
            series = df[col].dropna(); N = len(series)
            if N < 2: continue
            yf = np.fft.rfft(series - series.mean()); xf = np.fft.rfftfreq(N, 1); power = np.abs(yf); periods = np.divide(1.0, xf, out=np.full_like(xf, np.inf), where=xf!=0); mask = (periods > 7) & (periods < 365*2); fig.add_trace(go.Scatter(x=periods[mask], y=power[mask], name=col, mode='lines'))
        fig.add_vline(x=365, line_dash="dash", line_color="red"); fig.add_vline(x=182.5, line_dash="dash", line_color="green")
        st.plotly_chart(fig, use_container_width=True)
    create_fft_plot(price_cols, 'prices')
    create_fft_plot(arrival_cols, 'arrivals')

def perform_granger_causality_analysis(df, target_market_name, source_market_names, max_lag=7):
    st.header("Phase 6: Granger Causality Analysis", divider='rainbow')
    target_variable = f'price_{target_market_name}'
    if target_variable not in df.columns: st.error(f"Target '{target_variable}' not found. Skipping Granger analysis."); return
    for market in source_market_names:
        price_variable = f'price_{market}'; arrival_variable = f'arrival_{market}'
        if price_variable in df.columns and arrival_variable in df.columns:
            st.subheader(f"Causality: {market.title()} on {target_market_name.title()}")
            
            f = io.StringIO()
            with redirect_stdout(f):
                grangercausalitytests(df[[target_variable, price_variable]].dropna(), maxlag=max_lag, verbose=True)
            s = f.getvalue()
            st.text_area(f"Price → Price Causality Results", s, height=300)

            f = io.StringIO()
            with redirect_stdout(f):
                grangercausalitytests(df[[target_variable, arrival_variable]].dropna(), maxlag=max_lag, verbose=True)
            s = f.getvalue()
            st.text_area(f"Arrival → Price Causality Results", s, height=300)

# --- 6. MAIN APPLICATION LOGIC ---

st.title("📈 Commodity Market Analysis Dashboard")
st.write("This tool fetches live data to perform a detailed time-series analysis for a selected commodity and market.")

with st.sidebar:
    st.header("Analysis Parameters")
    if engine is None: st.sidebar.error("Database connection failed."); st.stop()
    
    states = get_states(engine)
    selected_state = st.selectbox("1. Select State", states, index=None, placeholder="Choose a state...")
    districts, markets, commodities = [], [], []
    if selected_state:
        districts = get_districts(engine, selected_state)
    selected_district = st.selectbox("2. Select District", districts, index=None, placeholder="Choose a district...", disabled=not selected_state)
    if selected_district:
        markets = get_markets(engine, selected_district)
    selected_market = st.selectbox("3. Select Target Market", markets, index=None, placeholder="Choose a market...", disabled=not selected_district)
    if selected_market:
        commodities = get_commodities(engine, selected_market)
    selected_commodity = st.selectbox("4. Select Commodity", commodities, index=None, placeholder="Choose a commodity...", disabled=not selected_market)
    selected_imputation = st.radio("5. Select Arrival Imputation Method", ('akima', 'Rolling Window Mean'), horizontal=True, disabled=not selected_commodity)
    run_button = st.button("🚀 Run Full Analysis", use_container_width=True, disabled=not selected_commodity)

if run_button:
    if not all([selected_state, selected_district, selected_market, selected_commodity]):
        st.warning("Please complete all selections in the sidebar to run the analysis.")
    else:
        with st.spinner("Running full analysis pipeline... This may take a moment."):
            try:
                # --- PHASE 1: Data Fetching and Processing ---
                st.header("Phase 1: Data Preparation", divider='rainbow')
                s_dist = selected_district.strip(); s_mkt = selected_market.strip(); s_comm = selected_commodity.strip()
                with st.expander("See District Market Summary"):
                    summary_query = "SELECT market_name, mp_last_year, mp_last_3_years, mp_last_5_years FROM commodities_market_analysis WHERE district_name = %(district)s AND commodity = %(commodity)s;"
                    summary_df = pd.read_sql(summary_query, engine, params={'district':selected_district, 'commodity':selected_commodity})
                    st.dataframe(summary_df)
                correlated_query = "SELECT * FROM commodities_market_analysis WHERE market_name = %(market)s AND district_name = %(district)s AND commodity = %(commodity)s;"
                target_info_df = pd.read_sql(correlated_query, engine, params={'market':selected_market, 'district':selected_district, 'commodity':selected_commodity})
                target_market_code = target_info_df.iloc[0]['market_code']
                all_market_codes = {target_market_code}
                for i in range(1, 6):
                    code_col = f'top_{i}_market_code'
                    if code_col in target_info_df.columns and pd.notna(target_info_df.iloc[0][code_col]): all_market_codes.add(str(target_info_df.iloc[0][code_col]).split('_')[0])
                market_codes_tuple = tuple([int(code) for code in all_market_codes])
                map_query = "SELECT market_code, market_name FROM commodities_market_analysis WHERE market_code IN %(codes)s;"
                map_df = pd.read_sql(map_query, engine, params={'codes':market_codes_tuple})
                id_to_name_map = {row['market_code']: row['market_name'].strip().replace(" ", "").lower() for index, row in map_df.iterrows()}
                target_market_clean_name = id_to_name_map[target_market_code]
                source_market_clean_names = [name for id, name in id_to_name_map.items() if id != target_market_code]
                daily_data_query = "SELECT reported_date, arrival, modal_price, der_market_id FROM market_wise_daily WHERE commodity_name = %(commodity)s AND der_market_id IN %(codes)s;"
                daily_df_raw = pd.read_sql(daily_data_query, engine, params={'commodity':selected_commodity, 'codes':market_codes_tuple})
                st.write(f"Fetched {len(daily_df_raw)} daily records for {len(all_market_codes)} markets.")
                daily_df_raw.rename(columns={'modal_price': 'price'}, inplace=True)
                daily_df = daily_df_raw.groupby(['reported_date', 'der_market_id']).agg(arrival=('arrival', lambda x: x.sum(min_count=1)), price=('price', 'mean')).reset_index()
                daily_df['is_imputed'] = daily_df['arrival'].isna()
                if selected_imputation == 'akima': imputation_lambda = lambda x: x.interpolate(method='akima').ffill().bfill()
                else: imputation_lambda = lambda x: x.fillna(x.rolling(window=7, min_periods=1).mean()).ffill().bfill()
                daily_df['arrival'] = daily_df.groupby('der_market_id')['arrival'].transform(imputation_lambda)
                master_df = daily_df.pivot_table(index='reported_date', columns='der_market_id', values=['arrival', 'price', 'is_imputed'])
                master_df = master_df.reindex(sorted(master_df.columns, key=lambda x: (x[1], x[0])), axis=1)
                master_df.columns = [f'{val}_{id_to_name_map.get(int(market_id), str(market_id))}' for val, market_id in master_df.columns]
                master_df = master_df.sort_index().asfreq('D')
                for col in [c for c in master_df.columns if 'is_imputed' in c]: master_df[col] = master_df[col].fillna(True).astype(bool)
                for col in [c for c in master_df.columns if 'is_imputed' not in c]: master_df[col] = master_df[col].ffill().bfill()
                st.subheader("Master Data Preview")
                st.dataframe(master_df.head())
                @st.cache_data
                def convert_df_to_excel(df):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=True, sheet_name='MasterData')
                    return output.getvalue()
                excel_data = convert_df_to_excel(master_df)
                st.download_button(label="📥 Download Master Data as Excel", data=excel_data, file_name=f"master_data_{s_dist}_{s_mkt}_{s_comm}.xlsx".replace(" ", "_"), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

                # --- RUN ALL ANALYSIS PHASES ---
                perform_exploratory_data_analysis(master_df)
                perform_correlation_analysis(master_df, target_market_clean_name)
                perform_lag_analysis(master_df, target_market_clean_name, source_market_clean_names)
                perform_seasonality_analysis(master_df) 
                # perform_granger_causality_analysis(master_df, target_market_clean_name, source_market_clean_names)
                st.success("✅✅✅ Full analysis complete! ✅✅✅")
            except Exception as e:
                st.error(f"An error occurred during the analysis: {e}")
else:
    st.info("Please make your selections in the sidebar and click 'Run Full Analysis' to begin.")

