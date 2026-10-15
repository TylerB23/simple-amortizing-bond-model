import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import streamlit as st
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

####################################################
# BOND MODEL CALCULATION FUNCTIONS
####################################################

def dates_and_periods(closing, first_bond_year, maturity):
    """
    This function calculates the total number of annual periods in the model and the length of its initial stub period,
    given the closing date, first bond year ending date, and the maturity date.

    Args:
        closing (datetime.date): the date the bond closes.
        first_bond_year (datetime.date): the date of the end of the first bond year (i.e., first principal payment)
        maturity (datetime.date): the date of the final payment on the bond.

    Returns:
        tuple[float, int]: a tuple containing an integer and a float. The first is the fraction of a year contained in
        the first period, from closing to first_bond_year. The second is the total number of periods in the bond model,
        including the stub period.
    """
    closing_year, closing_month, closing_day = closing.year, closing.month, closing.day
    bond_year, bond_month, bond_day = first_bond_year.year, first_bond_year.month, first_bond_year.day
    mat_year, mat_month, mat_day = maturity.year, maturity.month, maturity.day

    # helper functions to adjust dates
    d1_adj = lambda d1 : 30 if d1 == 31 else d1
    d2_adj = lambda d1, d2 : 30 if d1 == 30 and d2 == 31 else d2

    stub_period = (360 * (bond_year - closing_year) + 30 * (bond_month - closing_month) +
                   (d2_adj(closing_day, bond_day) - d1_adj(closing_day))) / 360
    num_periods = mat_year - bond_year + (0 if isinstance(stub_period, int) and not isinstance(stub_period, bool) else 1)

    return stub_period, num_periods

def solve_model_arbitrary_revenue(num_periods, stub_period, rev_vector, cpn, capi = True):
    """
    Solves a bond model with an arbitrary revenue curve.
    Args:
        num_periods (int): the number of periods in the bond model. Calculate with `dates_and_periods`.
        stub_period (int): the fraction of a year for the first period. Calculate with `dates_and_periods`.
        rev_vector (list[float]): list containing revenue amounts available per period.
        cpn (float): coupon rate on the bond.
        capi (bool): True if model should restrict capitalized interest to IRS 3-year limit (default); false otherwise.

    Returns:
        tuple[list[float], list[float], list[float]]: a tuple containing three lists of floats. The first
        contains the principal paid in each period. The second contains the interest payments. The third contains
        the budgeted capitalized interest.
    """
    principal = np.zeros(num_periods)
    interest = np.zeros(num_periods)
    cap_int = np.zeros(num_periods)

    principal[-1] = max(0, np.floor((rev_vector[-1] / (1 + cpn)) / 5000) * 5000)
    interest[-1] = principal[-1] * cpn
    cap_int[-1] = min(0, rev_vector[-1] - interest[-1])

    for i in range(num_periods-2, -1, -1):
        principal[i] = max(0, np.floor(((rev_vector[i] - interest[i+1]) / (1 + cpn)) / 5000) * 5000)
        interest[i] = interest[i+1] + principal[i] * cpn
        cap_int[i] = min(0, rev_vector[i] - interest[i])

    principal[0] = max(0, np.floor(((rev_vector[0] - interest[1] * stub_period) / (1 + cpn * stub_period)) / 5000) * 5000)
    interest[0] = (interest[1] + principal[0] * cpn) * stub_period
    cap_int[0] = min(0, rev_vector[0] - interest[0])

    if capi:
        total_capi = -1 * np.cumsum(cap_int)
        irs_limit = 3 * cpn * np.sum(principal)
        for i in range(num_periods - 1):
            if total_capi[i] > irs_limit:
                cap_int[i] = - max(irs_limit - total_capi[i - 1], 0)

    return principal, interest, cap_int

def pricing(cpn, yld, num_periods):
    """ Calculates bond price, assuming semiannual compounding.
    Args:
        cpn (float): the coupon rate.
        yld (float): the yield to worst.
        num_periods (float): the number of periods in the model.

    Returns:
        (float): the price of the bond, per $100 face.
    """
    lamda = (1+yld/2)**(-1)
    price = 100 * lamda**(2 * num_periods) + (100 * cpn / 2) * (lamda / (1 - lamda)) * (1 - lamda**(2 * num_periods))
    return price

def sources_and_uses(principal, interest, capi, price, coi, dsrf):
    """
    Computes the sources and uses of funds for a bond issuance based on the
    provided inputs.

    The method calculates various financial components such as par amount,
    premium/discount, cost of issuance, debt service reserve fund,
    capitalized interest, and net proceeds. It returns the sources
    and uses of funds in a structured manner.

    Args:
        principal (array-like): Principal amounts paid each period. Generate with `solve_model_arbitrary_revenue`.
        interest (array-like): Interest payments for each period. Generate with `solve_model_arbitrary_revenue`.
        capi (array-like): Capitalized interest amounts for each period. Generate with `solve_model_arbitrary_revenue`.
        price (float): Dollar price of the bonds per $100 face. Generate with `pricing`.
        coi (float): Cost of issuance expressed as a percentage of par amount.
        dsrf (bool): A boolean flag indicating whether a debt service reserve fund is used.

    Returns:
        tuple[dict[str, float], dict[str, float]]: A tuple containing two dictionaries.
        The first dictionary contains the sources of funds with corresponding amounts,
        and the second dictionary contains the uses of funds with corresponding amounts.
    """
    par = np.sum(principal)
    premium_discount = par * (price / 100 - 1)

    coi_amt = par * coi
    debt_service = principal + interest
    reserve_amt = min(0.1 * par, 1.25 * np.average(debt_service), np.max(debt_service)) if dsrf else 0
    total_capi = -1 * np.sum(capi)
    proceeds = par + premium_discount - coi_amt - reserve_amt - total_capi

    sources = {"Par": par,
               "Premium (Discount)": premium_discount}
    uses = {"Net Proceeds": proceeds,
            "Cost of Issuance": coi_amt,
            "Debt Service Reserve Fund": reserve_amt,
            "Capitalized Interest": total_capi}

    return sources, uses

# Helper function for formatting our dataframes later
def _style_final_row(df):
    def make_bold(row):
        # Check if the current row's index is the same as the last index of the DF
        is_last = row.name == df.index[-1]
        return ['font-weight: bold' if is_last else '' for _ in row]
    return df.style.format("{:,.0f}").apply(make_bold, axis=1)

####################################################
# STREAMLIT
####################################################

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

    with inputcol1: # dates, DSCR toggle, and expenses
        closing = st.date_input("Closing Date", format = "MM/DD/YYYY")
        init_byr = date.today().replace(day=1) + relativedelta(months=+6)
        init_mty = init_byr + relativedelta(years=+30)
        maturity = st.date_input("Final Maturity", value = init_mty, format = "MM/DD/YYYY")
        init_year = closing.year if maturity.month > closing.month else closing.year + 1
        first_bond_year = date(init_year, maturity.month, maturity.day)
        dscr = st.number_input("Debt Service Coverage Ratio", min_value=1., value=1.25, step=.01)
        with st.popover("Input Ongoing Expenses?"):
            stub_period, num_periods = dates_and_periods(closing, first_bond_year, maturity)
            bond_years = [first_bond_year + relativedelta(years=+i) for i in range(num_periods)]
            init_fees = 1_500 * np.ones(num_periods)
            fee_table = pd.DataFrame({"Bond Year Ending": bond_years, "Fees": init_fees})
            fees = st.data_editor(fee_table, height ="stretch", column_config={
                                  "Bond Year Ending": st.column_config.DateColumn(disabled=True, format="MM/DD/YYYY"),
                                  "Fees": st.column_config.NumberColumn(required=True, format="dollar", step = 0.01)
                                  })

    with inputcol2: # coupon rate, yield rate, COI checkbox, DSRF checkbox, Cap-I checkbox
        cpn = st.slider("Coupon Rate (Percent)", min_value=0., max_value=20.0, value=5.0, step=.125, format="%.3f%%")
        yld = st.slider("Yield (Percent)", min_value=0., max_value=20.0, value=5.0, step=.125, format="%.3f%%")
        coi = st.slider("Total Cost of Issuance (Percent)", min_value=0., max_value=20.0, value=2.5, step=.125, format="%.3f%%")
        dsrf = st.checkbox("Include Reserve Fund?", value=True)
        capi_toggle = st.checkbox("Respect IRS Cap-I Limit?", value=True)

    st.markdown("#### Initial Revenues")
    init_revenue = 1_000_000 * np.ones(num_periods)
    revenue_table = pd.DataFrame({"Bond Year Ending" : bond_years,
                                  "Revenue Available" : init_revenue})

    rev_table = st.data_editor(revenue_table, height ="stretch", column_config={
                       "Bond Year Ending": st.column_config.DateColumn(disabled=True, format="MM/DD/YYYY"),
                       "Revenue Available": st.column_config.NumberColumn(required=True, format="dollar", step=0.01)
                       })

    st.divider()

    st.markdown("## Outputs")

    revenues_for_model = (rev_table["Revenue Available"].to_numpy()) / dscr - fees["Fees"].to_numpy()
    principal, interest, capi = solve_model_arbitrary_revenue(num_periods, stub_period,
                                                              revenues_for_model, cpn / 100, capi=capi_toggle)

    fig, ax = plt.subplots()
    dates = rev_table['Bond Year Ending']
    width = 270
    bottom = np.zeros(num_periods) # keeps track of how high each bar is so far after adding the principal

    debt_service = {"Principal": principal,
                    "Interest": interest,
                    "Capitalized Interest": capi,
                    "Ongoing Expenses": fees["Fees"].to_numpy()
                    }
    for label, amount in debt_service.items():
        p = ax.bar(dates, amount, width, label=label, bottom=bottom)
        bottom += amount

    ax.set_ylim(0, max(rev_table["Revenue Available"]) * 1.1)
    fig.autofmt_xdate()
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.set_xlabel("Bond Year")
    ax.set_ylabel("Total Debt Service ($)")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.8),
              ncol=2, fancybox=True, shadow=True)

    st.pyplot(fig)

    st.markdown("### Sources & Uses and Debt Service")
    sources, uses = sources_and_uses(principal, interest, capi, pricing(cpn / 100, yld / 100, num_periods), coi / 100,
                                     dsrf)
    sources_df = pd.DataFrame.from_dict(sources, orient="index", columns=['Amount'])
    uses_df = pd.DataFrame.from_dict(uses, orient="index", columns=['Amount'])
    sources_df.loc['Total'] = sources_df['Amount'].sum()
    uses_df.loc['Total'] = uses_df['Amount'].sum()
    st.table(sources_df.style.format("${:,.0f}"))
    st.table(uses_df.style.format("${:,.0f}"))

    ds_df = pd.DataFrame.from_dict(debt_service)
    ds_df['Total Debt Service'] = ds_df.sum(axis=1)
    st.table(ds_df.style.format("${:,.0f}"))

    blank_row = pd.DataFrame(index=[0], columns=sources_df.columns)
    s_and_u = pd.concat([sources_df,blank_row, uses_df])
    return ds_df.to_csv().encode("utf-8"), s_and_u.to_csv().encode("utf-8")

####################################################
# EXECUTION
####################################################

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

download_buttons()