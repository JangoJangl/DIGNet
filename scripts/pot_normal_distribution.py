import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os
import pandas as pd



if __name__ == "__main__":

    x = np.arange(-10,10.1,0.01)
    guassian_1 = stats.norm(-5,1).pdf(x)
    gaussian_2 = stats.norm(5,1).pdf(x)
    y = 0.25*guassian_1 + 0.75*gaussian_2
    m_projection = stats.norm(1, 3).pdf(x)
    i_projection = gaussian_2

    fig_M = go.Figure(data=go.Scatter(x=x, y=y, name='True Distribution'))
    # fig_M.update_layout(template='simple_white', title='M-Projection')
    fig_M.update_xaxes(title_text="X")
    fig_M.update_yaxes(title_text='Density',  # axis label
                     showline=True,  # add line at x=0
                     linecolor='black',  # line color
                     linewidth=1.4,  # line size
                     ticks='outside',  # ticks outside axis
                     tickwidth=1.4,  # tick width
                     tickcolor='black',  # tick color
                     dtick=0.1
                     )
    fig_M.update_layout(shapes=[go.layout.Shape(
        type='rect',
        xref='paper',
        yref='paper',
        x0=0,
        y0=0,
        x1=1.0,
        y1=1.0,
        line={'width': 1, 'color': 'black'})],
        plot_bgcolor='white',)
    fig_M.add_trace(go.Scatter(x=x, y=m_projection, name='M-Projection',line = dict(dash='dash')))
    fig_M.show()

    fig_I = go.Figure(data=go.Scatter(x=x, y=y, name='True Distribution'))
    # fig_I.update_layout(template='simple_white', title='I-
    fig_I.update_xaxes(title_text="X")
    fig_I.update_yaxes(title_text='Density',  # axis label
                       showline=True,  # add line at x=0
                       linecolor='black',  # line color
                       linewidth=1.4,  # line size
                       ticks='outside',  # ticks outside axis
                       tickwidth=1.4,  # tick width
                       tickcolor='black',  # tick color
                       dtick=0.1
                       )
    fig_I.update_layout(shapes=[go.layout.Shape(
        type='rect',
        xref='paper',
        yref='paper',
        x0=0,
        y0=0,
        x1=1.0,
        y1=1.0,
        line={'width': 1, 'color': 'black'})],
        plot_bgcolor='white',)
    fig_I.add_trace(go.Scatter(x=x, y=i_projection, name='I-Projection',line = dict(dash='dash')))
    fig_I.show()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    fig_M.write_image("plots/fig_M.pdf")
    fig_I.write_image("plots/fig_I.pdf")
