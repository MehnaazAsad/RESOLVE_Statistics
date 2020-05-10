import numpy as np
import random
from bokeh.layouts import column, row
from bokeh.models import CustomJS, Slider
from bokeh.plotting import ColumnDataSource, figure, output_file, show

def behroozi10(logmstar, theta):
    """ 
    This function calculates the B10 stellar to halo mass relation 
    using the functional form 
    """
    M_1, Mstar_0, beta, delta = theta[:4]
    gamma = 1.56
    second_term = (beta*np.log10((10**logmstar)/(10**Mstar_0)))
    third_term_num = (((10**logmstar)/(10**Mstar_0))**delta)
    third_term_denom = (1 + (((10**logmstar)/(10**Mstar_0))**(-gamma)))
    logmh = M_1 + second_term + (third_term_num/third_term_denom) - 0.5

    return logmh

global logmstar

mstar_min = np.round(np.log10((10**8)/2.041),1) 
mstar_max = np.round(np.log10((10**12)/2.041),1) 
logmstar = np.linspace(mstar_min, mstar_max, 500)
theta = [12.35, 10.72, 0.44, 0.57, 0.15]
logmh = behroozi10(logmstar, theta)

source = ColumnDataSource(data=dict(x=logmh, y=logmstar))

plot = figure(plot_width=950, plot_height=600, x_range=(10.4, 16))
plot.line('x', 'y', source=source, line_width=5, line_alpha=0.6)
plot.xaxis.axis_label = "Halo Mass"
plot.yaxis.axis_label = "Stellar Mass"

# Param sliders
mhalo_slider = Slider(start=10, end=13, value=theta[0], step=.2, 
    title="Characteristic halo mass")
mstellar_slider = Slider(start=8, end=11, value=theta[1], step=.2, 
    title="Characteristic stellar mass")
lowslope_slider = Slider(start=0.2, end=0.7, value=theta[2], step=.05, 
    title="Low mass slope")
highslope_slider = Slider(start=0.4, end=0.8, value=theta[3], step=.05, 
    title="High mass slope")
# scatter_slider = Slider(start=0.1, end=0.5, value=theta[4], step=.01, 
#     title="Scatter")


callback = CustomJS(args=dict(source=source, mhalo=mhalo_slider, 
    mstellar=mstellar_slider, lowslope=lowslope_slider, 
    highslope=highslope_slider),
    code="""
    const data = source.data;
    const M_1 = mhalo.value;
    const Mstar_0 = mstellar.value;
    const beta = lowslope.value;
    const delta = highslope.value;
    const gamma = -1.56;
    const x = data['x'];
    const y = data['y'];
    for (var i = 0; i < y.length; i++) {
        const second_term = beta * Math.log10((Math.pow(10,y[i]))/(Math.pow(10,Mstar_0)));
        const third_term_num = Math.pow(((Math.pow(10,y[i]))/(Math.pow(10,Mstar_0))),delta);
        const third_term_denom = 1 + (Math.pow(((Math.pow(10, y[i]))/(Math.pow(10,Mstar_0))),gamma));
        x[i] = M_1 + second_term + (third_term_num/third_term_denom) - 0.5
    }
    source.change.emit();
""")

mhalo_slider.js_on_change('value', callback)
mstellar_slider.js_on_change('value', callback)
lowslope_slider.js_on_change('value', callback)
highslope_slider.js_on_change('value', callback)
# scatter_slider.js_on_change('value', callback2)

layout = row(
    plot,
    column(mhalo_slider, mstellar_slider, lowslope_slider, highslope_slider),
)

output_file("../../../../../Interactive_Models/behroozi10_smhm.html", 
    title="Behroozi10 SMHM")

show(layout)

