from matplotlib.legend_handler import HandlerBase

class AnyObjectHandler(HandlerBase): 
    def create_artists(self, legend, orig_handle, x0, y0, width, height, 
        fontsize, trans):
        if orig_handle[3]:
            topcap_r = plt.Line2D([x0,x0+width*0.2], [0.8*height, 0.8*height], 
                linestyle='-', color='darkred') 
            body_r = plt.Line2D([x0+width*0.1, x0+width*0.1], \
                [0.2*height, 0.8*height], linestyle='-', color='darkred')
            bottomcap_r = plt.Line2D([x0, x0+width*0.2], \
                [0.2*height, 0.2*height], linestyle='-', color='darkred')
            topcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
                [0.8*height, 0.8*height], linestyle='-', color='darkblue') 
            body_b = plt.Line2D([x0+width*0.5, x0+width*0.5], \
                [0.2*height, 0.8*height], linestyle='-', color='darkblue')
            bottomcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
                [0.2*height, 0.2*height], linestyle='-', color='darkblue')
            return [topcap_r, body_r, bottomcap_r, topcap_b, body_b, bottomcap_b]
        l1 = plt.Line2D([x0, x0+width], [0.7*height, 0.7*height], 
            linestyle=orig_handle[2], color=orig_handle[0]) 
        l2 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height],  
            linestyle=orig_handle[2],
            color=orig_handle[1])
        return [l1, l2] 

fig1= plt.figure(figsize=(10,10))

maxis_red_data, phi_red_data, err_red_data = red_data[0], red_data[1], \
    red_data[2]
maxis_blue_data, phi_blue_data, err_blue_data = blue_data[0], blue_data[1], \
    blue_data[2]
lower_err = phi_red_data - err_red_data
upper_err = phi_red_data + err_red_data
lower_err = phi_red_data - lower_err
upper_err = upper_err - phi_red_data
asymmetric_err = [lower_err, upper_err]
ered = plt.errorbar(maxis_red_data,phi_red_data,yerr=asymmetric_err,
    color='darkred',fmt='s', ecolor='darkred',markersize=5,capsize=5,
    capthick=0.5,label='data',zorder=10)
lower_err = phi_blue_data - err_blue_data
upper_err = phi_blue_data + err_blue_data
lower_err = phi_blue_data - lower_err
upper_err = upper_err - phi_blue_data
asymmetric_err = [lower_err, upper_err]
eblue = plt.errorbar(maxis_blue_data,phi_blue_data,yerr=asymmetric_err,
    color='darkblue',fmt='s', ecolor='darkblue',markersize=5,capsize=5,
    capthick=0.5,label='data',zorder=10)
for idx in range(len(result[0][0])):
    plt.plot(result[0][0][idx],result[0][1][idx],color='indianred',
        linestyle='-',alpha=0.3,zorder=0,label='model')
for idx in range(len(result[0][2])):
    plt.plot(result[0][2][idx],result[0][3][idx],color='cornflowerblue',
        linestyle='-',alpha=0.3,zorder=0,label='model')
for idx in range(len(result[1][0])):
    plt.plot(result[1][0][idx],result[1][1][idx],color='indianred',
        linestyle='-',alpha=0.3,zorder=0)
for idx in range(len(result[1][2])):
    plt.plot(result[1][2][idx],result[1][3][idx],color='cornflowerblue',
        linestyle='-',alpha=0.3,zorder=0)
for idx in range(len(result[2][0])):
    plt.plot(result[2][0][idx],result[2][1][idx],color='indianred',
        linestyle='-',alpha=0.3,zorder=0)
for idx in range(len(result[2][2])):
    plt.plot(result[2][2][idx],result[2][3][idx],color='cornflowerblue',
        linestyle='-',alpha=0.3,zorder=0)
for idx in range(len(result[3][0])):
    plt.plot(result[3][0][idx],result[3][1][idx],color='indianred',
        linestyle='-',alpha=0.3,zorder=0)
for idx in range(len(result[3][2])):
    plt.plot(result[3][2][idx],result[3][3][idx],color='cornflowerblue',
        linestyle='-',alpha=0.3,zorder=0)
for idx in range(len(result[4][0])):
    plt.plot(result[4][0][idx],result[4][1][idx],color='indianred',
        linestyle='-',alpha=0.3,zorder=0)
for idx in range(len(result[4][2])):
    plt.plot(result[4][2][idx],result[4][3][idx],color='cornflowerblue',
        linestyle='-',alpha=0.3,zorder=0)
# REMOVED BEST FIT ERROR
plt.errorbar(maxis_bf_red,phi_bf_red,
    color='darkred',fmt='-s',ecolor='darkred',markersize=3,lw=3,
    capsize=5,capthick=0.5,label='best fit',zorder=10)
plt.errorbar(maxis_bf_blue,phi_bf_blue,
    color='darkblue',fmt='-s',ecolor='darkblue',markersize=3,lw=3,
    capsize=5,capthick=0.5,label='best fit',zorder=10)
plt.ylim(-5,-1)
if mf_type == 'smf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
elif mf_type == 'bmf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
plt.legend([("darkred", "darkblue", "-", False), \
    ("indianred","cornflowerblue", "-", False), (0, 0, False, True)],\
    ["model", "best fit", "data"], handler_map={tuple: AnyObjectHandler()},\
    loc='best', prop={'size': 20})
# plt.legend([("darkred", "darkblue", "-"), \
#     ("indianred","cornflowerblue", "-")],\
#     ["model", "best fit", "data"], handler_map={tuple: AnyObjectHandler()},\
#     loc='best', prop={'size': 20})
# plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
plt.annotate(r'$\boldsymbol\chi ^2 \approx$ {0}'.format(np.round(bf_chi2,2)), 
    xy=(0.1, 0.1), xycoords='axes fraction', bbox=dict(boxstyle="square", 
    ec='k', fc='lightgray', alpha=0.5), size=15)
plt.show()