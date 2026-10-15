import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.dates import DateFormatter
import streamlit as st
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

def dates_and_periods(closing, first_bond_year, maturity):
    yc, mc, dc = closing.year, closing.month, closing.day
    yby, mby, dby = first_bond_year.year, first_bond_year.month, first_bond_year.day
    ym, mm, dm = maturity.year, maturity.month, maturity.day

    # helper functions to adjust dates
    d1_adj = lambda d1 : 30 if d1 == 31 else d1
    d2_adj = lambda d1, d2 : 30 if d1 == 30 and d2 == 31 else d2

    stub_period = (360 * (yby - yc) + 30 * (mby - mc) + (d2_adj(dc, dby) - d1_adj(dc))) / 360
    # The logic is to adjust for the stub period
    n = ym - yby + (0 if isinstance(stub_period, int) and not isinstance(stub_period, bool) else 1)

    return stub_period, n

def solve_model_arbitrary_revenue(n, stub_period, R, r, capi = True):
    # n and stub_period come from dates_and_periods
    # R is the revenue constraint (an arbitrary revenue curve, input as a vector)
    # r is the coupon rate (float)
    # capi is True if we are respecting IRS limit, False otherwise
    
    # Create our principal, interest, and capitalized interest arrays
    principal = np.zeros(n)
    interest = np.zeros(n)
    cap_int = np.zeros(n)

    # Calculate the last year's principal and interest
    principal[-1] = max(0,np.floor((R[-1] / (1 + r)) / 5000) * 5000)
    interest[-1] = principal[-1] * r
    cap_int[-1] = min(0, R[-1] - interest[-1])
    
    # The for loop calculates principal, interest, and cap-I
    for i in range(n-1):
        principal[-i-2] = max(0, np.floor(((R[-i-2] - interest[-i-1]) / (1 + r)) / 5000) * 5000)
        interest[-i-2] = interest[-i-1] + principal[-i-2] * r
        cap_int[-i-2] = min(0, R[-i-2] - interest[-i-2])

    # Fix the stub period
    principal[0] = max(0,np.floor(((R[0] - interest[1] * stub_period) / (1 + r *
                         stub_period)) / 5000) * 5000)
    interest[0] = (interest[1] + principal[0] * r) * stub_period
    cap_int[0] = min(0, R[0] - interest[0])
    
    # If we care, we check against the IRS limit for cap-I
    if capi:
        total_capi = - np.cumsum(cap_int)
        IRS_limit = 3 * r * np.sum(principal)
        for i in range(n-1):
            if total_capi[i] > IRS_limit:
                cap_int[i] = - max(IRS_limit - total_capi[i-1], 0)
    
    return principal, interest, cap_int

def pricing(r, y, n):
    #r is the coupon rate (we assume annual coupon payments for simplicity)
    #y is the yield
    #n is the number of bond years to maturity
    price = 100 * ((1+y)**(-n) + r/y - r/(y*(1+y)**n))
    return int(price * 1000) / 1000

def sources_and_uses(principal, interest, capi, price, coi, dsrf):
    # principal, interest, and capi come from our solve_model function
    # price is the dollar price of the bonds
    # coi is the cost of issuance as a percentage of par
    # dsrf is a boolean determining whether we have
    
    par = np.sum(principal)
    premium_discount = par * (1 - price / 100)
    
    coi_amt = par * coi
    debt_service = principal + interest
    reserve_amt = min(0.1 * par, 1.25 * np.average(debt_service), np.max(debt_service)) if dsrf else 0
    total_capi = -np.sum(capi)
    proceeds = par + premium_discount - coi_amt - reserve_amt - total_capi

    sources = {"Par": par, "Premium (Discount)": premium_discount}
    uses = {"Net Proceeds": proceeds,
            "Cost of Issuance": coi_amt,
            "Debt Service Reserve Fund": reserve_amt,
            "Capitalized Interest": total_capi}

    return sources, uses

# Helper function for formatting our dataframes later
def style_final_row(df):
    def make_bold(row):
        # Check if the current row's index is the same as the last index of the DF
        is_last = row.name == df.index[-1]
        return ['font-weight: bold' if is_last else '' for _ in row]
    return df.style.format("{:,.0f}").apply(make_bold, axis=1)

# This section begins the streamlit
st.markdown("# Simple Bond Modeling App, v2")
st.markdown('''This is a simple app for putting together fully amortizing muni
    bond models. The _Inputs_ section below includes all the parameters for the
    model, and the _Outputs_ section summarizes the results. This app is designed to
    size bonds around a revenue curve, rather than around a desired project
    fund. I built this because I wanted a faster alternative for quickly mocking
    up bond models than building them in Excel. This interactive app lets the
    user see how the results change as the parameters change quickly and easily.  
    Play around with the parameters below - including revenue available to pay the
    bonds - to see how the sources and uses change.''')
st.markdown(''' This is version two, which
    added date-based inputs, arbitrary revenue inputs, a debt service coverage
    ratio, automatically calculated capitalized interest, and arbitrary ongoing
    expenses.''')
st.markdown("## Inputs")
st.divider()

@st.fragment
def plot_debt_service():

    # This section creates the Streamlit interactive elements
    inputcol1, inputcol2 = st.columns(2, gap='medium')

    with inputcol1:
        # Input dates and checkboxes
        closing = st.date_input("Closing Date", format = "MM/DD/YYYY")
        # init_byr = date(date.today().year, date.today().month + 6, 1)
        init_byr = date.today().replace(day=1) + relativedelta(months=+6)
        init_mty = init_byr + relativedelta(years=+30)
        maturity = st.date_input("Final Maturity", value = init_mty,
                                 format = "MM/DD/YYYY")
        init_year = closing.year if maturity.month > closing.month else closing.year + 1
        first_bond_year = date(init_year, maturity.month, maturity.day)
        dscr = st.number_input("Debt Service Coverage Ratio", min_value = 1.,
                               value=1.25, step=.01) 
        with st.popover("Input Ongoing Expenses?"):
            stub_period, n = dates_and_periods(closing, first_bond_year, maturity)
            bond_years = [first_bond_year + relativedelta(years=+i) for i in range(n)]
            init_fees = 1_500 * np.ones(n)
            fee_table = pd.DataFrame({"Bond Year Ending": bond_years,
                                      "Fees": init_fees})
            fees = st.data_editor(fee_table, height ="stretch", column_config={
                                  "Bond Year Ending":
                                  st.column_config.DateColumn(disabled=True,
                                  format="MM/DD/YYYY"),
                                  "Fees": 
                                  st.column_config.NumberColumn(required=True,
                                  format="dollar", step = 0.01)
                                  })

    with inputcol2:
        # Percentage Sliders and DSCR
        r = st.slider("Coupon Rate (Percent)", min_value = 0., max_value = 20.0, value = 5.0, step
                      = .125, format="%.3f%%") # Don't forget to divide by 100!
        y = st.slider("Yield (Percent)", min_value = 0., max_value = 20.0, value = 5.0, step
                      = .125, format="%.3f%%") # Don't forget to divide by 100!
        coi = st.slider("Total Cost of Issuance (Percent)", min_value = 0.,
                        max_value = 20.0, value = 2.5, step = .125, format="%.3f%%")
        dsrf = st.checkbox("Include Reserve Fund?", value=True)
        capi_toggle = st.checkbox("Respect IRS Cap-I Limit?", value=True)

    # Builds the Revenue Input table
    st.markdown("#### Initial Revenues")   
    init_revenue = 1_000_000 * np.ones(n)
    revenue_table = pd.DataFrame({"Bond Year Ending" : bond_years,
                                  "Revenue Available" : init_revenue})

    R = st.data_editor(revenue_table, height = "stretch", column_config={
                       "Bond Year Ending":
                       st.column_config.DateColumn(disabled=True,
                       format="MM/DD/YYYY"),
                       "Revenue Available": 
                       st.column_config.NumberColumn(required=True,
                       format="dollar", step = 0.01)
                       })

    st.divider()
    st.markdown("## Outputs")

    # This section solves the model
    revenues_for_model = (R["Revenue Available"].to_numpy()) / dscr - fees["Fees"].to_numpy()
    principal, interest, capi = solve_model_arbitrary_revenue(n, stub_period,
         revenues_for_model, r/100, capi=capi_toggle) 

    # This section creates the plot
    fig, ax = plt.subplots()
    x = R['Bond Year Ending']
    width = 270 # width of the bars on the chart
    bottom = np.zeros(n) # keeps track of how high each bar is so far after adding the principal

    debt_service = {"Principal": principal,
                    "Interest": interest,
                    "Capitalized Interest": capi,
                    "Ongoing Expenses": fees["Fees"].to_numpy()
                    }
    for label, amount in debt_service.items():
        p = ax.bar(x, amount, width, label=label, bottom = bottom)
        bottom += amount

    ax.set_ylim(0, max(R["Revenue Available"]) * 1.1)
    plt.gcf().autofmt_xdate()
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.set_xlabel("Bond Year")
    ax.set_ylabel("Total Debt Service ($)")
    ax.legend()

    # This section displays sources and uses and the debt service table

    st.pyplot(fig)

    # This section prints the sources and uses, and the debt service table
    st.markdown("### Sources & Uses and Debt Service")
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        sources, uses = sources_and_uses(principal, interest, capi, pricing(r/100,
                                         y/100, n), coi/100, dsrf) 
        sources_df = pd.DataFrame.from_dict(sources, orient="index",
                     columns=['Amount'])
        uses_df = pd.DataFrame.from_dict(uses, orient="index",
                  columns=['Amount'])
        sources_df.loc['Total'] = sources_df['Amount'].sum()
        uses_df.loc['Total'] = uses_df['Amount'].sum()
        #st.table(sources_df.style.format("${:,.0f}").apply(style_final_row, axis=1))
        st.table(sources_df.style.format("${:,.0f}"))
        st.table(uses_df.style.format("${:,.0f}"))

    with col2:
        ds_df = pd.DataFrame.from_dict(debt_service)
        ds_df['Total Debt Service'] = ds_df.sum(axis=1)
        st.table(ds_df.style.format("${:,.0f}"))

    blank_row = pd.DataFrame(index=[0], columns=sources_df.columns)
    s_and_u = pd.concat([sources_df,blank_row, uses_df])
    return ds_df.to_csv().encode("utf-8"), s_and_u.to_csv().encode("utf-8")

# Call the fragment function
# plot_debt_service()
ds_df, s_and_u = plot_debt_service()

# Separate fragment for download buttons
@st.fragment
def download_buttons():
    st.download_button(label="Download Sources and Uses Table", data=s_and_u,
                       file_name="sources_and_uses.csv", mime="text/csv",
                       icon=":material/download:")
    st.download_button(label="Download Debt Service Table", data=ds_df,
                       file_name="debt_service.csv", mime="text/csv",
                       icon=":material/download:")


# Call the fragment function
download_buttons()
