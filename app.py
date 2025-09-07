import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, render_template, send_file
import yfinance as yf
import pandas as pd
import os
import io
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from matplotlib.ticker import FuncFormatter
from currency_symbols import CurrencySymbols # Needed for currency symbols

# Set template folder as current directory
import os
template_dir = os.path.abspath('.')  # points to directory containing app.py and HTML files
app = Flask(__name__, template_folder=template_dir)

CURRENCY_SYMBOLS_MAP = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "CHF": "CHF",
    "CAD": "C$",
    "AUD": "A$",
    "INR": "₹",
    "RUB": "₽",
    "KRW": "₩",
    "BRL": "R$",
    "MXN": "Mex$",
    "ZAR": "R",
    "SGD": "S$",
    "HKD": "HK$",
    "NZD": "NZ$",
    "SEK": "kr",
    "NOK": "kr",
    "DKK": "kr",
    "PLN": "zł",
    "TRY": "₺",
    "THB": "฿",
    "MYR": "RM",
    "IDR": "Rp",
    "PHP": "₱",
    "VND": "₫",
    "ILS": "₪",
    "SAR": "﷼",
    "AED": "د.إ",
    "EGP": "£",
    "NGN": "₦",
    "ARS": "$",
    "CLP": "$",
    "COP": "$",
    "PEN": "S/",
    "BDT": "৳",
    "PKR": "₨",
    "LKR": "Rs",
    "CZK": "Kč",
    "HUF": "Ft",
    "RON": "lei",
    "UAH": "₴",
    "KZT": "₸",
    "KWD": "د.ك",
    "QAR": "﷼",
    # Additional currencies from the comprehensive list
    "BHD": ".د.ب",
    "OMR": "﷼",
    "JOD": "د.ا",
    "MAD": "MAD",
    "TND": "د.ت",
    "DZD": "د.ج",
    "ETB": "Br",
    "KES": "KSh",
    "UGX": "USh",
    "TZS": "TSh",
    "GHS": "₵",
    "ZMW": "ZK",
    "BWP": "P",
    "NAD": "N$",
    "MUR": "₨",
    "SCR": "₨",
    "ISK": "kr",
    "HRK": "kn",
    "RSD": "дин.",
    "BGN": "лв",
    "MKD": "ден",
    "ALL": "L",
    "MDL": "L",
    "GEL": "₾",
    "AMD": "֏",
    "AZN": "₼",
    "BYN": "Br",
    "MNT": "₮",
    "KPW": "₩",
    "MMK": "Ks",
    "KHR": "៛",
    "LAK": "₭",
    "BND": "B$",
    "FJD": "FJ$",
    "PGK": "K",
    "VUV": "VT",
    "WST": "WS$",
    "TOP": "T$",
    "AFN": "؋",
    "IRR": "﷼",
    "IQD": "ع.د",
    "LBP": "ل.ل",
    "SYP": "£S",
    "YER": "﷼",
    "LYD": "ل.د",
    "SDG": "ج.س",
    "SOS": "Sh.So.",
    "DJF": "Fdj",
    "ERN": "Nfk",
    "CVE": "Esc",
    "GMD": "D",
    "XOF": "CFA",  # West African CFA Franc
    "SLL": "Le",
    "LRD": "L$",
    "MWK": "MK",
    "MZN": "MT",
    "AOA": "Kz",
    "CDF": "FC",
    "XAF": "FCFA",  # Central African CFA Franc
    "XPF": "₣",     # CFP Franc
    "XCD": "EC$",   # East Caribbean Dollar
    "TWD": "NT$",   # Taiwan Dollar (New Taiwan Dollar)
    "BOB": "Bs",    # Bolivian Boliviano
    "UYU": "$U",    # Uruguayan Peso
    "PYG": "₲",     # Paraguayan Guarani
    "HTG": "G",     # Haitian Gourde
    "DOP": "RD$",   # Dominican Peso
    "JMD": "J$",    # Jamaican Dollar
    "TTD": "TT$",   # Trinidad and Tobago Dollar
    "BSD": "B$",    # Bahamian Dollar
    "BBD": "Bds$",  # Barbadian Dollar
    "BZD": "BZ$",   # Belize Dollar
    "GTQ": "Q",     # Guatemalan Quetzal
    "HNL": "L",     # Honduran Lempira
    "NIO": "C$",    # Nicaraguan Córdoba
    "CRC": "₡",     # Costa Rican Colón
    "PAB": "B/.",   # Panamanian Balboa
    "SVC": "$",     # Salvadoran Colón
    "CUP": "$",     # Cuban Peso
    "GYD": "G$",    # Guyanese Dollar
    "SRD": "$",     # Surinamese Dollar
    "AWG": "ƒ",     # Aruban Florin
    "ANG": "ƒ",     # Netherlands Antillean Guilder
    "XEU": "₠",     # European Currency Unit (obsolete)
    "STD": "Db",    # São Tomé and Príncipe Dobra
    "SHP": "£",     # Saint Helena Pound
    "FKP": "£",     # Falkland Islands Pound
    "GIP": "£",     # Gibraltar Pound
    "GGP": "£",     # Guernsey Pound
    "JEP": "£",     # Jersey Pound
    "IMP": "£",     # Isle of Man Pound
    "SZL": "L",     # Swazi Lilangeni
    "LSL": "L",     # Lesotho Loti
    "MGA": "Ar",    # Malagasy Ariary
    "KMF": "CF",    # Comorian Franc
    "RWF": "₣",     # Rwandan Franc
    "BIF": "₣",     # Burundian Franc
    "GNF": "₣",     # Guinean Franc
    "SLE": "Le",    # Sierra Leonean Leone (new)
    "CIV": "CFA",   # Ivory Coast (uses XOF)
    "MLI": "CFA",   # Mali (uses XOF)
    "BFA": "CFA",   # Burkina Faso (uses XOF)
    "NER": "CFA",   # Niger (uses XOF)
    "SEN": "CFA",   # Senegal (uses XOF)
    "TGO": "CFA",   # Togo (uses XOF)
    "BEN": "CFA",   # Benin (uses XOF)
    "CMR": "FCFA",  # Cameroon (uses XAF)
    "CAF": "FCFA",  # Central African Republic (uses XAF)
    "TCD": "FCFA",  # Chad (uses XAF)
    "COG": "FCFA",  # Republic of the Congo (uses XAF)
    "GNQ": "FCFA",  # Equatorial Guinea (uses XAF)
    "GAB": "FCFA",  # Gabon (uses XAF)
}

# --- Currency symbol helper ---
def get_currency_display(info):
    currency_code = info.get("currency", "")
    if currency_code in CURRENCY_SYMBOLS_MAP:
        symbol = CURRENCY_SYMBOLS_MAP[currency_code]
        return f"{currency_code} ({symbol})" if symbol else currency_code
    else:
        # fallback: try library or just code
        try:
            from currency_symbols import CurrencySymbols
            symbol = CurrencySymbols.get_symbol(currency_code)
            if symbol:
                return f"{currency_code} ({symbol})"
        except ImportError:
            pass
        return currency_code or ""

# --- Helper function to format HTML tables with currency in header ---
def format_table_html_with_currency(df, currency_display, classes="financial-table"):
    """Convert dataframe to HTML with currency in the first column header"""
    df_copy = df.copy()
    df_copy.index.name = None  # Remove index name
    html_table = df_copy.to_html(classes=classes, border=1)
    # Replace the first empty <th></th> with "Currency" + currency code/symbol
    html_table = html_table.replace('<th></th>', f'<th>Currency {currency_display}</th>', 1)
    return html_table

def format_cashflow_table_html_with_currency(df, currency_display, classes="financial-table"):
    """Convert cashflow dataframe to HTML with currency replacing Parameter header"""
    df_copy = df.copy()
    html_table = df_copy.to_html(classes=classes, index=False, border=1)
    # Replace Parameter header with "Currency" + currency code/symbol
    html_table = html_table.replace('<th>Parameter</th>', f'<th>Currency {currency_display}</th>', 1)
    return html_table

# --- Helper Functions (refactored) ---
def human_format(num):
    try:
        num = float(num)
    except:
        return "N/A"
    if num == 0:
        return "0"
    magnitude = 0
    abs_num = abs(num)
    while abs_num >= 1000:
        magnitude += 1
        abs_num /= 1000.0
    suffix = ['', 'K', 'M', 'B', 'T']
    if magnitude < len(suffix):
        return f"{num/1000**magnitude:.1f}{suffix[magnitude]}"
    else:
        return f"{num:.2e}"

def format_number(value):
    return f"{value/1e9:.2f}B" if pd.notna(value) and abs(value) >= 1e9 else \
           f"{value/1e6:.2f}M" if pd.notna(value) and abs(value) >= 1e6 else \
           f"{value:.2f}" if pd.notna(value) else "N/A"

def assign_score(parameter, values):
    if parameter in ["Operating Cash Flow", "Free Cash Flow (FCF)", "Free Cash Flow"]:
        return 5 if all(v > 0 for v in values) else 3
    return 1

def score_cagr(value, metric):
    if value == "N/A" or value == "":
        return ""
    try:
        val = float(value.strip('%'))
    except:
        return ""
    if metric in ["Revenue", "Net Profit"]:
        if val < 0:
            return 1
        elif val < 2.5:
            return 2
        elif val < 5:
            return 3
        elif val < 10:
            return 4
        else:
            return 5
    if metric == "Gross Profit Margin %":
        return 5 if val > 20 else 4 if val > 15 else 3 if val > 10 else 2 if val > 5 else 1
    if metric == "Operating Margin %":
        return 5 if val > 12 else 4 if val > 10 else 3 if val > 5 else 2 if val > 3 else 1
    if metric == "Net Profit Margin %":
        return 5 if val > 12 else 4 if val > 10 else 3 if val > 5 else 2 if val > 3 else 1
    return ""

weight_map = {
    "Revenue": 0.1,
    "Net Profit": 0.1,
    "Gross Profit Margin %": 0.125,
    "Operating Margin %": 0.1,
    "Net Profit Margin %": 0.075
}

def get_executives(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    try:
        info = ticker.info
    except:
        info = {}
    executives = []
    try:
        key_execs = info.get("companyOfficers", [])
        for exec in key_execs:
            executives.append({
                "Name": exec.get("name", "N/A"),
                "Title": exec.get("title", "N/A")
            })
    except:
        pass
    return executives

def get_valuation_measures(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    try:
        info = ticker.info
    except:
        info = {}
    metrics_map = {
        "Market Cap (intraday)": "marketCap",
        "Enterprise Value": "enterpriseValue",
        "Trailing P/E": "trailingPE",
        "Forward P/E": "forwardPE",
        "PEG Ratio (5 yr expected)": "pegRatio",
        "Price/Sales (ttm)": "priceToSalesTrailing12Months",
        "Price/Book (mrq)": "priceToBook",
        "Enterprise Value/Revenue": "enterpriseToRevenue",
        "Enterprise Value/EBITDA": "enterpriseToEbitda"
    }
    data = []
    for label, key in metrics_map.items():
        val = info.get(key, "N/A")
        if isinstance(val, (int, float)):
            if val >= 1e9:
                val_str = f"{val/1e9:.2f}B"
            elif val >= 1e6:
                val_str = f"{val/1e6:.2f}M"
            else:
                val_str = f"{val:.2f}"
        else:
            val_str = val
        data.append({"Measure": label, "Value": val_str})
    df = pd.DataFrame(data)
    html_table = df.to_html(classes="financial-table", index=False, border=1)
    return html_table

def score_current_ratio(val):
    if pd.isna(val):
        return ""
    if val < 1.0:
        return 1
    elif val < 1.2:
        return 2
    elif val < 1.5:
        return 3
    elif val < 2.0:
        return 4
    else:
        return 5

def score_debt_to_equity(val):
    if pd.isna(val):
        return ""
    if val > 2.0:
        return 1
    elif val > 1.5:
        return 2
    elif val > 1.0:
        return 3
    elif val > 0.5:
        return 4
    else:
        return 5

def score_net_debt_to_equity(val):
    if pd.isna(val):
        return ""
    if val > 2.0:
        return 1
    elif val > 1.5:
        return 2
    elif val > 1.0:
        return 3
    elif val > 0.5:
        return 4
    else:
        return 5

def get_series(df, possible_names):
    for name in df.index:
        for target in possible_names:
            if name.replace(" ", "").lower() == target.replace(" ", "").lower():
                return df.loc[name]
    return pd.Series([0]*len(df.columns), index=df.columns)

def compute_ttm(quarterly_series):
    if quarterly_series.empty:
        return 0
    return quarterly_series.iloc[-1]

def format_columns(df):
    if df is not None and not df.empty:
        df.columns = [col.strftime('%d-%m-%Y') if hasattr(col, 'strftime') else col for col in df.columns]
    return df

def get_latest_data(annual_df, quarterly_df):
    annual_df = format_columns(annual_df)
    quarterly_df = format_columns(quarterly_df)
    if quarterly_df is not None and not quarterly_df.empty:
        latest_quarter = quarterly_df.iloc[:, 0]
        latest_quarter.name = 'Latest'
        annual_df = annual_df.join(latest_quarter, how='outer')
    cols = [c for c in annual_df.columns if c != 'Latest']
    cols.sort(key=lambda x: pd.to_datetime(x, format='%d-%m-%Y', errors='coerce'))
    if 'Latest' in annual_df.columns:
        cols.append('Latest')
    return annual_df[cols]

COLORS = ['#1a4e8c', '#5a95d1', '#3b73b9']

INCOME_MAP = {
    "Total Revenue": "Total Revenue",
    "Cost Of Revenue": "Cost Of Revenue",
    "Gross Profit": "Gross Profit",
    "Operating Expense": "Operating Expense",
    "Operating Income": "Operating Income",
    "EBITDA": "EBITDA",
    "Net Income": "Net Income"
}

BALANCE_MAP = {
    "Cash And Cash Equivalents": "Cash And Cash Equivalents",
    "Current Assets": "Current Assets",
    "Current Liabilities": "Current Liabilities",
    "Accounts Receivable": "Accounts Receivable",
    "Accounts Payable": "Accounts Payable",
    "Inventory": "Inventory",
    "Total Debt": "Total Debt",
    "Total Assets": "Total Assets"
}

CASH_MAP = {
    "Operating Cash Flow": "Operating Cash Flow",
    "Free Cash Flow": "Free Cash Flow",
    "Capital Expenditure": "Capital Expenditure",
    "Depreciation And Amortization": "Depreciation And Amortization",
    "Issuance Of Debt": "Issuance Of Debt",
    "Repayment Of Debt": "Repayment Of Debt",
    "End Cash Position": "End Cash Position"
}

def df_to_numeric(df):
    return df.fillna(0).infer_objects()

def add_bar_labels(ax, bars, fmt="%.1fB"):
    for bar in bars:
        h = bar.get_height()
        if h != 0:
            ax.annotate(fmt % (h / 1e9),
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0,3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

def add_line_labels(ax, x_labels, y_values, fmt="%.1fB", divide_by_billion=True):
    for x, y in zip(x_labels, y_values):
        if y != 0:
            v = y / 1e9 if divide_by_billion else y
            ax.annotate(fmt % v,
                        xy=(x, y),
                        xytext=(0,5), textcoords="offset points",
                        ha='center', fontsize=8)

def build_summary(mapping, annual_df, quarterly_df, years_list, ttm_label):
    summary = pd.DataFrame(index=list(mapping.keys()), columns=[str(y) for y in years_list] + [ttm_label])
    for name, col in mapping.items():
        row = [annual_df[col][y] if (col in annual_df.columns and y in annual_df.index) else None for y in years_list]
        ttm_val = quarterly_df[col].iloc[0] if quarterly_df is not None and col in quarterly_df.columns else None
        row.append(ttm_val)
        summary.loc[name] = row
    return summary

def plot_grouped_bar_to_base64(title, data_rows, labels, x_labels, stacked=False):
    width = 0.35
    ind = np.arange(len(x_labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor('white')
    ax.grid(True, linestyle=':', color='gray', alpha=0.7)
    
    def y_axis_formatter(x, pos):
        if x >= 1e9:
            return f'{int(x/1e9)}B'
        elif x >= 1e6:
            return f'{int(x/1e6)}M'
        elif x >= 1e3:
            return f'{int(x/1e3)}K'
        else:
            return f'{int(x)}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    
    if stacked:
        bottom_vals = np.zeros(len(x_labels))
        for i, row in enumerate(data_rows):
            bar = ax.bar(x_labels, row, width, bottom=bottom_vals, color=COLORS[i % len(COLORS)], label=labels[i])
            add_bar_labels(ax, bar)
            bottom_vals += row
    else:
        for i, row in enumerate(data_rows):
            bars = ax.bar(ind + i*width, row, width, color=COLORS[i % len(COLORS)], label=labels[i])
            add_bar_labels(ax, bars)
        ax.set_xticks(ind + width/2)
        ax.set_xticklabels(x_labels)
    
    ax.set_title(title)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(labels))
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{encoded}"

def plot_line_chart_to_base64(title, data_rows, labels, x_labels, percentage=False):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor('white')
    ax.grid(True, linestyle=':', color='gray', alpha=0.7)
    
    def y_axis_formatter(x, pos):
        if percentage:
            return f'{int(x)}%'
        if x >= 1e9:
            return f'{int(x/1e9)}B'
        elif x >= 1e6:
            return f'{int(x/1e6)}M'
        elif x >= 1e3:
            return f'{int(x/1e3)}K'
        else:
            return f'{int(x)}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    
    for i, row in enumerate(data_rows):
        fmt = "%.1f%%" if percentage else "%.1f"
        ax.plot(x_labels, row, marker='o', color=COLORS[i % len(COLORS)], label=labels[i])
        add_line_labels(ax, x_labels, row, fmt=fmt, divide_by_billion=not percentage)
    
    ax.set_title(title)
    if percentage:
        ax.set_ylabel('Percentage')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(labels))
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{encoded}"

def fetch_financials_summary(ticker: str, years: int):
    stock = yf.Ticker(ticker)
    
    income = stock.financials.T
    balance = stock.balance_sheet.T
    cashflow = stock.cashflow.T
    
    income_q = stock.quarterly_financials.T
    balance_q = stock.quarterly_balance_sheet.T
    cashflow_q = stock.quarterly_cashflow.T
    
    for df in [income, balance, cashflow]:
        df.index = pd.to_datetime(df.index).year
    
    for df in [income_q, balance_q, cashflow_q]:
        df.index = pd.to_datetime(df.index).year
    
    current_year = datetime.now().year
    years_list = list(range(current_year - years, current_year))
    ttm_label = f"{current_year} (TTM)"
    
    income_summary = build_summary(INCOME_MAP, income, income_q, years_list, ttm_label)
    balance_summary = build_summary(BALANCE_MAP, balance, balance_q, years_list, ttm_label)
    cashflow_summary = build_summary(CASH_MAP, cashflow, cashflow_q, years_list, ttm_label)
    
    income_num = df_to_numeric(income_summary)
    balance_num = df_to_numeric(balance_summary)
    cashflow_num = df_to_numeric(cashflow_summary)
    
    x_labels = income_summary.columns.tolist()
    
    charts = {
        "income": [
            plot_grouped_bar_to_base64("Revenue vs Gross Profit",
             [income_num.loc['Total Revenue'], income_num.loc['Gross Profit']],
             ['Total Revenue', 'Gross Profit'], x_labels),
            plot_line_chart_to_base64("Operating Income & EBITDA Trend",
             [income_num.loc['Operating Income'], income_num.loc['EBITDA']],
             ['Operating Income', 'EBITDA'], x_labels),
            plot_grouped_bar_to_base64("Net Income vs Operating Expense",
             [income_num.loc['Net Income'], income_num.loc['Operating Expense']],
             ['Net Income', 'Operating Expense'], x_labels),
            plot_line_chart_to_base64("Profit Margins (%)",
             [(income_num.loc['Net Income'] / income_num.loc['Total Revenue']) * 100],
             ['Net Profit Margin (%)'], x_labels, percentage=True)
        ],
        "balance": [
            plot_grouped_bar_to_base64("Cash + Current Assets vs Current Liabilities",
             [balance_num.loc['Cash And Cash Equivalents'] + balance_num.loc['Current Assets'],
              balance_num.loc['Current Liabilities']],
             ['Cash + Current Assets', 'Current Liabilities'], x_labels),
            plot_grouped_bar_to_base64("Accounts Receivable vs Accounts Payable",
             [balance_num.loc['Accounts Receivable'], balance_num.loc['Accounts Payable']],
             ['Accounts Receivable', 'Accounts Payable'], x_labels),
            plot_line_chart_to_base64("Inventory Trend",
             [balance_num.loc['Inventory']], ['Inventory'], x_labels),
            plot_grouped_bar_to_base64("Debt vs Other Assets",
             [balance_num.loc['Total Debt'], balance_num.loc['Total Assets'] - balance_num.loc['Total Debt']],
             ['Total Debt', 'Other Assets'], x_labels)
        ],
        "cashflow": [
            plot_grouped_bar_to_base64("Operating Cash Flow vs Free Cash Flow",
             [cashflow_num.loc['Operating Cash Flow'], cashflow_num.loc['Free Cash Flow']],
             ['Operating Cash Flow', 'Free Cash Flow'], x_labels),
            plot_line_chart_to_base64("Capital Expenditure vs Depreciation And Amortization",
             [cashflow_num.loc['Capital Expenditure'], cashflow_num.loc['Depreciation And Amortization']],
             ['Capital Expenditure', 'Depreciation And Amortization'], x_labels),
            plot_grouped_bar_to_base64("Issuance Of Debt vs Repayment Of Debt",
             [cashflow_num.loc['Issuance Of Debt'], cashflow_num.loc['Repayment Of Debt']],
             ['Issuance Of Debt', 'Repayment Of Debt'], x_labels, stacked=True),
            plot_line_chart_to_base64("Cash Position Trend",
             [cashflow_num.loc['End Cash Position']], ['Cash Position'], x_labels)
        ]
    }
    
    return income_summary, balance_summary, cashflow_summary, charts

def format_summary_df(df):
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if col == 'CAGR':
            continue
        df_formatted[col] = df_formatted[col].apply(lambda x: human_format(x) if pd.notna(x) else "N/A")
    return df_formatted

# --- Main app route ---
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        ticker_symbol = request.form.get("ticker").upper()
        num_years = int(request.form.get("years"))
        
        ticker = yf.Ticker(ticker_symbol)
        try:
            info = ticker.info
        except:
            info = {}
        
        company_name = info.get("longName", ticker_symbol)
        description = info.get('longBusinessSummary', 'N/A')
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        website = info.get("website", "N/A")
        executives = get_executives(ticker_symbol)
        valuation_html = get_valuation_measures(ticker_symbol)
        
        logo_url = info.get("logo_url")
        if not logo_url and website:
            try:
                domain = website.replace("https://", "").replace("http://", "").split("/")[0]
                logo_url = f"https://logo.clearbit.com/{domain}"
            except:
                logo_url = None
        
        currency_display = get_currency_display(info)  # Get currency symbol/code
        
        income_statement = ticker.financials
        ttm_income = ticker.quarterly_financials.sum(axis=1)
        
        if income_statement is None or income_statement.empty:
            table_html = None
            table_df = pd.DataFrame()
            years_str = []
        else:
            income_statement = income_statement.sort_index(axis=1)
            current_year = datetime.now().year
            requested_years = list(range(current_year - num_years, current_year))
            available_years = []
            
            for year in requested_years:
                year_timestamp = pd.Timestamp(year=year, month=12, day=31)
                if year_timestamp in income_statement.columns:
                    available_years.append(year_timestamp)
            
            if len(available_years) < num_years:
                available_years = income_statement.columns[-min(num_years, len(income_statement.columns)):]
            
            years_str = [y.strftime('%Y') for y in available_years]
            ttm_label = f"{current_year} (TTM)"
            years_str.append(ttm_label)
            
            def safe_row(name):
                series = pd.Series([0] * (len(years_str) - 1), index=years_str[:-1])
                if name in income_statement.index:
                    s = income_statement.loc[name]
                    s.index = [y.strftime('%Y') for y in s.index]
                    series.update(s)
                if name in ttm_income.index:
                    series[years_str[-1]] = ttm_income[name]
                return series
            
            revenue = safe_row("Total Revenue")
            gross_profit = safe_row("Gross Profit")
            operating_income = safe_row("Operating Income")
            net_profit = safe_row("Net Income")
            
            table_df = pd.DataFrame(columns=years_str)
            
            rows = [
                ("Revenue", revenue),
                ("Gross Profit", gross_profit),
                ("Operating Income", operating_income),
                ("Net Profit", net_profit),
                ("Gross Profit Margin %", (gross_profit / revenue * 100).replace([float("inf"), -float("inf")], 0)),
                ("Operating Margin %", (operating_income / revenue * 100).replace([float("inf"), -float("inf")], 0)),
                ("Net Profit Margin %", (net_profit / revenue * 100).replace([float("inf"), -float("inf")], 0)),
            ]
            
            score_col = {}
            weight_col = {}
            weighted_score_col = {}
            
            for row_name, series in rows:
                if "Margin" in row_name:
                    table_df.loc[row_name, years_str] = series.apply(lambda x: f"{x:.2f}%" if x != 0 else "N/A")
                else:
                    table_df.loc[row_name, years_str] = series.apply(format_number)
                
                first_val = series.iloc[0]
                last_val = series.iloc[-2] if len(series) > 1 else series.iloc[0]
                n_years = len(series) - 1
                
                if pd.notna(first_val) and pd.notna(last_val) and first_val > 0 and last_val > 0 and n_years > 1:
                    cagr = (last_val / first_val) ** (1 / (n_years - 1)) - 1
                    cagr_str = f"{cagr*100:.2f}%"
                    table_df.loc[row_name, "CAGR"] = cagr_str
                else:
                    table_df.loc[row_name, "CAGR"] = "N/A"
                
                score = score_cagr(table_df.loc[row_name, "CAGR"], row_name)
                score_col[row_name] = score
                weight = weight_map.get(row_name, 0)
                weight_col[row_name] = weight
                weighted_score_col[row_name] = score * weight if score != "" else ""
            
            table_df["Score"] = pd.Series(score_col)
            table_df["Weight"] = pd.Series(weight_col)
            table_df["Weighted Score"] = pd.Series(weighted_score_col)
            table_df = table_df[years_str + ["CAGR", "Score", "Weight", "Weighted Score"]]
            
            # Use new formatting function with currency in header
            table_html = format_table_html_with_currency(table_df, currency_display)
        
        # --- Balance Sheet Metrics ---
        balance_sheet = ticker.balance_sheet
        quarterly_balance = ticker.quarterly_balance_sheet
        balance_sheet = balance_sheet.reindex(sorted(balance_sheet.columns), axis=1)
        quarterly_balance = quarterly_balance.reindex(sorted(quarterly_balance.columns), axis=1)
        
        current_assets = get_series(balance_sheet, ["Total Current Assets", "Current Assets"])
        current_liabilities = get_series(balance_sheet, ["Total Current Liabilities", "Current Liabilities"])
        equity = get_series(balance_sheet, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity"])
        short_term_debt = get_series(balance_sheet, ["Short Long Term Debt", "Short Term Debt"])
        long_term_debt = get_series(balance_sheet, ["Long Term Debt"])
        cash_and_equiv = get_series(balance_sheet, ["Cash", "Cash And Cash Equivalents", "CashAndCashEquivalents"])
        
        total_debt = short_term_debt + long_term_debt
        net_debt = total_debt - cash_and_equiv
        current_ratio = current_assets / current_liabilities
        debt_to_equity = total_debt / equity
        net_debt_to_equity = net_debt / equity
        
        current_assets_ttm = compute_ttm(get_series(quarterly_balance, ["Total Current Assets", "Current Assets"]))
        current_liabilities_ttm = compute_ttm(get_series(quarterly_balance, ["Total Current Liabilities", "Current Liabilities"]))
        equity_ttm = compute_ttm(get_series(quarterly_balance, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity"]))
        short_term_debt_ttm = compute_ttm(get_series(quarterly_balance, ["Short Long Term Debt", "Short Term Debt"]))
        long_term_debt_ttm = compute_ttm(get_series(quarterly_balance, ["Long Term Debt"]))
        cash_and_equiv_ttm = compute_ttm(get_series(quarterly_balance, ["Cash", "Cash And Cash Equivalents", "CashAndCashEquivalents"]))
        
        total_debt_ttm = short_term_debt_ttm + long_term_debt_ttm
        net_debt_ttm = total_debt_ttm - cash_and_equiv_ttm
        
        current_ratio_ttm = current_assets_ttm / current_liabilities_ttm if current_liabilities_ttm != 0 else float('nan')
        debt_to_equity_ttm = total_debt_ttm / equity_ttm if equity_ttm != 0 else float('nan')
        net_debt_to_equity_ttm = net_debt_ttm / equity_ttm if equity_ttm != 0 else float('nan')
        
        metrics = pd.DataFrame({
            "Current Assets": current_assets,
            "Current Liabilities": current_liabilities,
            "Current Ratio": current_ratio,
            "Debt": total_debt,
            "Net Debt": net_debt,
            "Equity": equity,
            "Debt-to-Equity Ratio": debt_to_equity,
            "Net Debt-to-Equity Ratio": net_debt_to_equity
        })
        
        metrics = metrics.T
        metrics.columns = [str(c.year) for c in metrics.columns]
        
        current_year_ttm = ttm_label
        metrics[current_year_ttm] = [
            current_assets_ttm,
            current_liabilities_ttm,
            current_ratio_ttm,
            total_debt_ttm,
            net_debt_ttm,
            equity_ttm,
            debt_to_equity_ttm,
            net_debt_to_equity_ttm
        ]
        
        selected_years = years_str[:-1]
        columns_to_keep = selected_years + [current_year_ttm]
        metrics = metrics.reindex(columns=columns_to_keep)
        
        score_map = {
            "Current Ratio": score_current_ratio,
            "Debt-to-Equity Ratio": score_debt_to_equity,
            "Net Debt-to-Equity Ratio": score_net_debt_to_equity,
        }
        
        weights = {
            "Current Ratio": 0.05,
            "Debt-to-Equity Ratio": 0.075,
            "Net Debt-to-Equity Ratio": 0.05,
        }
        
        score_list = []
        weighted_score_list = []
        
        for row in metrics.index:
            if row in score_map:
                val = metrics.loc[row, ttm_label]
                try:
                    num_val = float(val)
                except:
                    num_val = None
                
                if num_val is not None:
                    score_val = score_map[row](num_val)
                else:
                    score_val = ""
                
                score_list.append(score_val)
                
                if score_val == "":
                    weighted_score_list.append("")
                else:
                    weighted_score_list.append(round(score_val * weights[row], 3))
            else:
                score_list.append("")
                weighted_score_list.append("")
        
        metrics["Score"] = score_list
        metrics["Weight"] = [weights.get(r, "") for r in metrics.index]
        metrics["Weighted Score"] = weighted_score_list
        
        for col in metrics.columns[:-3]:
            metrics[col] = metrics[col].apply(lambda x: human_format(x) if pd.notna(x) else "N/A")
        
        # Use new formatting function with currency in header
        metrics_html = format_table_html_with_currency(metrics, currency_display)
        
        # --- Cash Flow ---
        cashflow = ticker.cashflow
        ttm_cashflow = ticker.quarterly_cashflow.sum(axis=1)
        
        rows_to_keep = ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditures']
        available_rows = []
        
        for row in rows_to_keep:
            if row in cashflow.index:
                available_rows.append(row)
            elif row == 'Capital Expenditures' and 'Capital Expenditure' in cashflow.index:
                available_rows.append('Capital Expenditure')
        
        cashflow_filtered = cashflow.loc[available_rows]
        cashflow_filtered.columns = [str(col.year) for col in cashflow_filtered.columns]
        cashflow_filtered = cashflow_filtered.sort_index(axis=1, ascending=True)
        
        final_rows = []
        weight_mapping = {'Operating Cash Flow': 0.15, 'Free Cash Flow': 0.15, 'Capital Expenditures': 0.025, 'Capital Expenditure': 0.025}
        
        for param in available_rows:
            years = years_str[:-1]
            values = cashflow_filtered.loc[param, years].tolist()
            values.append(ttm_cashflow[param] if param in ttm_cashflow.index else 0)
            years.append(current_year_ttm)
            
            formatted_values = [human_format(v) for v in values]
            
            if param in ['Capital Expenditures', 'Capital Expenditure']:
                trend = values[-1] - values[0]
                if trend < 0:
                    score = 5
                elif abs(trend) < 1e7:
                    score = 3
                else:
                    score = 1
            else:
                score = assign_score(param, values)
            
            weight = weight_mapping.get(param, 0)
            weighted_score = round(score * weight, 3)
            
            row_dict = {"Parameter": param}
            for y, val in zip(years, formatted_values):
                row_dict[y] = val
            row_dict.update({"CAGR": "-", "Score": score, "Weight": weight, "Weighted Score": weighted_score})
            final_rows.append(row_dict)
        
        cashflow_table = pd.DataFrame(final_rows)
        
        # Use new formatting function with currency replacing Parameter header
        cashflow_html = format_cashflow_table_html_with_currency(cashflow_table, currency_display)
        
        # --- Unified Scorecard ---
        scorecard_columns = list(table_df.columns[:-3]) + ['Score', 'Weight', 'Weighted Score']
        additional_cols = []
        
        for col in metrics.columns:
            if col not in scorecard_columns and col not in ['Score', 'Weight', 'Weighted Score']:
                additional_cols.append(col)
        
        for col in cashflow_table.columns:
            if col not in scorecard_columns and col not in ['Score', 'Weight', 'Weighted Score', 'Parameter', 'CAGR']:
                additional_cols.append(col)
        
        final_columns = list(dict.fromkeys(scorecard_columns + additional_cols))
        
        financial_params = ['Revenue', 'Net Profit', 'Gross Profit Margin %', 'Operating Margin %', 'Net Profit Margin %']
        financial_slice = table_df.loc[financial_params, final_columns].copy()
        
        balance_params = ['Current Ratio', 'Debt-to-Equity Ratio', 'Net Debt-to-Equity Ratio']
        balance_columns = [col for col in final_columns if col in metrics.columns]
        balance_slice = metrics.loc[balance_params, balance_columns].copy()
        
        cashflow_params = cashflow_table['Parameter'].tolist()
        cashflow_df = cashflow_table.set_index('Parameter')
        available_cashflow_cols = [col for col in final_columns if col in cashflow_df.columns]
        cashflow_slice = cashflow_df.loc[cashflow_params, available_cashflow_cols].copy()
        
        for df_ in (financial_slice, balance_slice, cashflow_slice):
            for c in ['Score', 'Weight', 'Weighted Score']:
                if c not in df_.columns:
                    df_[c] = ""
        
        scorecard_df = pd.concat([financial_slice, balance_slice, cashflow_slice], sort=False)
        
        non_ttm_year = str(pd.Timestamp.now().year)
        ttm_year = f"{non_ttm_year} (TTM)"
        if non_ttm_year in scorecard_df.columns and ttm_year in scorecard_df.columns:
            scorecard_df = scorecard_df.drop(columns=[non_ttm_year])
        
        sum_weighted_score = pd.to_numeric(scorecard_df['Weighted Score'], errors='coerce').sum()
        sum_score = pd.to_numeric(scorecard_df['Score'], errors='coerce').sum()
        sum_weight = pd.to_numeric(scorecard_df['Weight'], errors='coerce').sum()
        
        sum_row = {col: '-' for col in scorecard_df.columns}
        sum_row['Weighted Score'] = round(sum_weighted_score, 3)
        sum_row['Score'] = round(sum_score, 3)
        sum_row['Weight'] = round(sum_weight, 3)
        
        scorecard_df.loc['Total Weighted Score'] = pd.Series(sum_row)
        
        def format_cell(x):
            if isinstance(x, str) and (x == 'N/A' or x == '-' or x == ''):
                return x
            try:
                num = float(str(x).replace('%', ''))
                if isinstance(x, str) and '%' in str(x):
                    return f"{num:.2f}%"
                return human_format(num)
            except:
                return x
        
        readable_cols = [c for c in scorecard_df.columns if c not in ['Score', 'Weight', 'Weighted Score']]
        scorecard_df.loc[:, readable_cols] = scorecard_df.loc[:, readable_cols].applymap(format_cell)
        scorecard_df.loc['Total Weighted Score', readable_cols] = '-'
        
        # Use new formatting function with currency in header
        scorecard_html = format_table_html_with_currency(scorecard_df, currency_display, classes='financial-table')
        
        income_summary, balance_summary, cashflow_summary, charts = fetch_financials_summary(ticker_symbol, num_years)
        
        income_summary_hr = format_summary_df(income_summary)
        balance_summary_hr = format_summary_df(balance_summary)
        cashflow_summary_hr = format_summary_df(cashflow_summary)
        
        # Use new formatting function for summary tables
        income_summary_html = format_table_html_with_currency(income_summary_hr, currency_display, classes="financial-table")
        balance_summary_html = format_table_html_with_currency(balance_summary_hr, currency_display, classes="financial-table")
        cashflow_summary_html = format_table_html_with_currency(cashflow_summary_hr, currency_display, classes="financial-table")
        
        download_url = f"/download_excel?ticker={ticker_symbol}"
        
        return render_template("results.html",
                               company_name=company_name,
                               ticker_symbol=ticker_symbol,
                               description=description,
                               sector=sector,
                               industry=industry,
                               website=website,
                               executives=executives,
                               valuation_html=valuation_html,
                               table_html=table_html,
                               balance_sheet_html=metrics_html,
                               cashflow_html=cashflow_html,
                               scorecard_html=scorecard_html,
                               logo_url=logo_url,
                               download_url=download_url,
                               income_summary_html=income_summary_html,
                               balance_summary_html=balance_summary_html,
                               cashflow_summary_html=cashflow_summary_html,
                               income_charts=charts["income"],
                               balance_charts=charts["balance"],
                               cashflow_charts=charts["cashflow"]
                               )
    
    return render_template("index.html")

@app.route("/download_excel")
def download_excel():
    ticker_symbol = request.args.get('ticker', '').upper()
    if not ticker_symbol:
        return "Ticker symbol missing", 400
    
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        income_statement = get_latest_data(ticker.financials, ticker.quarterly_financials)
    except:
        income_statement = pd.DataFrame()
    
    try:
        balance_sheet = get_latest_data(ticker.balance_sheet, ticker.quarterly_balance_sheet)
    except:
        balance_sheet = pd.DataFrame()
    
    try:
        cash_flow = get_latest_data(ticker.cashflow, ticker.quarterly_cashflow)
    except:
        cash_flow = pd.DataFrame()
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if not income_statement.empty:
            income_statement.to_excel(writer, sheet_name="Income Statement")
        if not balance_sheet.empty:
            balance_sheet.to_excel(writer, sheet_name="Balance Sheet")
        if not cash_flow.empty:
            cash_flow.to_excel(writer, sheet_name="Cash Flow")
    
    output.seek(0)
    filename = f"{ticker_symbol}_financials.xlsx"
    
    return send_file(output, as_attachment=True, download_name=filename,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)