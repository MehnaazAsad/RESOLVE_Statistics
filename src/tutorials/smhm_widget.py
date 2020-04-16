'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.widgets import Slider, Button, RadioButtons

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=15)
rc('text', usetex=True)

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

fig, ax = plt.subplots(figsize=(10,10))
plt.subplots_adjust(left=0.25, bottom=0.4)
plt.xlabel(r'$M_{h}$', fontsize=20)
plt.ylabel(r'$M_{*}$', fontsize=20)
l, = plt.plot(logmh, logmstar, lw=2)
# ax.margins(x=logmh.min())

axcolor = 'lightgoldenrodyellow'
axmhalo = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axmstellar = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axlowslope = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
axhighslope = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
axscatter = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor=axcolor)

smhalo = Slider(axmhalo, 'Characteristic halo mass', 10.0, 13.0, valinit=theta[0])
smstellar = Slider(axmstellar, 'Characteristic stellar mass', 8.0, 11.0, 
    valinit=theta[1])
slowslope = Slider(axlowslope, 'Low mass slope', 0.2, 0.7, valinit=theta[2])
shighslope = Slider(axhighslope, 'High mass slope', 0.4, 0.8, valinit=theta[3])
sscatter = Slider(axscatter, 'Scatter', 0.1, 0.3, valinit=theta[4])

def update(val):
    mhalo = smhalo.val
    mstellar = smstellar.val
    lowslope = slowslope.val
    highslope = shighslope.val
    scatter = sscatter.val
    theta = [mhalo, mstellar, lowslope, highslope, scatter]
    logmh = behroozi10(logmstar, theta)
    l.set_xdata(logmh)
    l.set_ydata(logmstar)
    fig.canvas.draw_idle()


smhalo.on_changed(update)
smstellar.on_changed(update)
slowslope.on_changed(update)
shighslope.on_changed(update)
sscatter.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    smhalo.reset()
    smstellar.reset()
    slowslope.reset()
    shighslope.reset()
    sscatter.reset()
button.on_clicked(reset)

plt.show()
'''

import numpy as np
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

plot = figure(plot_width=950, plot_height=600)
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
# scatter_slider = Slider(start=0.1, end=0.3, value=theta[4], step=.01, 
#     title="Scatter")

callback = CustomJS(args=dict(source=source, mhalo=mhalo_slider, mstellar=mstellar_slider, lowslope=lowslope_slider, highslope=highslope_slider),
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
# scatter_slider.js_on_change('value', callback)

layout = row(
    plot,
    column(mhalo_slider, mstellar_slider, lowslope_slider, highslope_slider),
)

output_file("../../../Interactive_Models/behroozi10_smhm.html", title="Behroozi10 SMHM")

show(layout)
