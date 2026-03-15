import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from matplotlib import ticker
import streamlit as st
import pandas as pd

def solve_model(n, R, r, g):
    # n is the number of periods
    # R is the revenue constraint
    # r is the coupon rate
    # g is the growth rate for the revenue curve

    # Create our principal  and interest arrays
    principal = np.zeros(n)
    interest = np.zeros(n)

    # this lambda expression makes it easy to fill in the revenues
    growth_rate = lambda i, g : (1+g)**i

    # Create the revenue array
    revenue = np.asarray([growth_rate(i,g) for i in np.arange(n)]) * R

    # Calculate the last year's principal and interest
    # I'm creating the arrays backwards because it makes the indexing easier
    principal[0] = np.floor((revenue[-1] / (1 + r)) / 5000) * 5000
    interest[0] = principal[0] * r
    for i in range(n-1):
        principal[i+1] = np.floor(((revenue[-i-1] - interest[i]) / (1 + r)) / 5000) * 5000
        interest[i+1] = interest[i] + principal[i+1] * r

    return np.flip(principal), np.flip(interest)

def pricing(r, y, n):
    #r is the coupon rate (we assume annual coupon payments for simplicity)
    #y is the yield
    #n is the number of bond years to maturity
    price = 100 * ((1+y)**(-n) + r/y - r/(y*(1+y)**n))
    return int(price * 1000) / 1000

def sources_and_uses(principal, interest, price, coi, dsrf):
    # principal and interest come from our solve_model function
    # price is the dollar price of the bonds
    # coi is the cost of issuance as a percentage of par
    # dsrf is a boolean determining whether we have
    
    par = np.sum(principal)
    premium_discount = par * (1 - price / 100)
    
    coi_amt = par * coi
    debt_service = principal + interest
    reserve_amt = min(0.1 * par, 1.25 * np.average(debt_service), np.max(debt_service)) if dsrf else 0
    proceeds = par + premium_discount - coi_amt - reserve_amt

    sources = {"Par": par, "Premium (Discount)": premium_discount}
    uses = {"Net Proceeds": proceeds, "Cost of Issuance": coi_amt, "Debt Service Reserve Fund": reserve_amt}

    return sources, uses

# Helper function for formatting our dataframes later
def style_final_row(df):
    def make_bold(row):
        # Check if the current row's index is the same as the last index of the DF
        is_last = row.name == df.index[-1]
        return ['font-weight: bold' if is_last else '' for _ in row]
    return df.style.format("{:,.0f}").apply(make_bold, axis=1)

# This section begins the streamlit
st.markdown("# Simple Bond Modeling App")
st.markdown('''This is a simple app for putting together fully amortizing muni
    bond models. The _Inputs_ section below includes all the parameters for the
    model, and the _Outputs_ section summarizes the results. This app is designed to
    size bonds around a revenue curve, rather than around a desired project
    fund. I built this because I wanted a faster alternative for quickly mocking
    up bond models than building them in Excel. This interactive app lets the
    user see how the results change as the parameters change quickly and easily.  
    Play around with the parameters below - including revenue available to pay the
    bonds - to see how the sources and uses change.''')
st.markdown("## Inputs")

st.divider()

@st.fragment
def plot_debt_service():

    # This section creates the Streamlit interactive elements
    n = st.slider("Maturity Length (Years)", min_value = 1, max_value = 40,
                  value = 30, step = 1)
    r = st.slider("Coupon Rate (Percent)", min_value = 0., max_value = 20.0, value = 5.0, step
                  = .125, format="%.3f%%") # Don't forget to divide by 100!
    y = st.slider("Yield (Percent)", min_value = 0., max_value = 20.0, value = 5.0, step
                  = .125, format="%.3f%%") # Don't forget to divide by 100!
    R = st.number_input("Base Year Revenue", min_value = 0, value = 1_000_000, step
                        = 10_000) # Don't forget to cast this to a float!
    g = st.slider("Revenue Growth Rate (Percent)", min_value = -5., max_value=5.,
                  value = 0., step=0.125, format="%.3f%%") # Don't forget to divide by 100!
    coi = st.slider("Total Cost of Issuance (Percent)", min_value = 0.,
                    max_value = 20.0, value = 2.5, step = .125, format="%.3f%%")
    dsrf = st.checkbox("Include Reserve Fund?", value=True)

    st.divider()
    st.markdown("## Outputs")

    # This section solves the model
    principal, interest = solve_model(n, R, r/100, g/100) 

    # This section creates the plot
    fig, ax = plt.subplots()
    n = len(principal)
    x = np.arange(1,n+1,1)
    width = 0.6 # width of the bars on the chart
    bottom = np.zeros(n) # keeps track of how high each bar is so far after adding the principal

    debt_service = {"Principal": principal, "Interest": interest}
    for label, amount in debt_service.items():
        p = ax.bar(x, amount, width, label=label, bottom = bottom)
        bottom += amount

    ax.set_ylim(0, max(principal + interest) * 1.2)
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
        sources, uses = sources_and_uses(principal, interest, pricing(r/100, y/100, n), coi/100, dsrf) 
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

# Call the fragment function
plot_debt_service(),