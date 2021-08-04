from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import copy
import math
import matplotlib
import matplotlib.artist as art
import matplotlib.font_manager
import matplotlib.textpath
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import scipy
import scipy.ndimage
import scipy.stats

VRSN_GT_1_2 = float('.'.join(matplotlib.__version__.split(os.extsep)[:2])) > 1.2

dtDflts = {
    'alpha'              : 0.6,
    'bin_count'          : 0.1,  # can be a (float) fraction of values
    'bottom_title_relpos': (0.5, 0.01),
    'conv_npts'          : 60,
    'dpi'                : 120,
    'exp_metric'         : True,  # plot/label experimental metric
    'fig_size'           : (10.0, 6.0),
    'file_fmt'           : 'png',
    'layout'             : {'tight': True, 'pad': 2.},
    'ln_clr'             : {'tk': 'k', 'f': ['lightblue', 'salmon', 'lightgreen', 'yellow'],
                            'e' : ['blue', 'red', 'darkgreen', 'k'], 'exp': 'magenta'},
    'ln_style'           : {'tk': '-'},  # 'tk' - ticks, 'b' - box, 'f' - face, 'e' - edge
    'ln_width'           : {'b': 2.0, 'tk': 1.5},
    'mark'               : {'f': '', 'cnv_a': 'o', 'cnv_b': 's', 'cnv_r': '*'},  # 'f' - flier point
    'mark_size'          : {'d1': 50, 'd2': 20, 'cnv': 30},
    'matrix_max_rows'    : 5,
    'patch_artist'       : True,
    'rand_seed'          : 20141010,
    'sbfig_margins'      : {'bottom': 0.2, 'left': 0.1},
    'sd_multiple'        : 10,
    'top_title_relpos'   : (0.5, 0.94),
    'txt_size'           : {'t': 12, 'dt': 8, 'tk': 12, 'a': 10},
    'whskr_reach'        : 1.5,
}


def boxplot(output_name, output_preds, posterior_wts, num_pts, seed, exp_obs, exp_std, num_bins, addtnl_uncrtnts):
    def mod_pred_dists(q, w, qsd, nBins, nSDMltpl=10, addl_uncertnts=[]):
        mu = np.mean(q)

        low = mu - nSDMltpl * qsd
        upp = mu + nSDMltpl * qsd
        hrange = (low, upp)
        gaussFltr = scipy.ndimage.gaussian_filter1d

        hgt, edges = np.histogram(q, bins=nBins, normed=True, range=hrange, weights=w)
        dy = (edges[-1] - edges[0]) / nBins

        lsHgt = [hgt]
        for _, fSD in addl_uncertnts:
            lsHgt.append(gaussFltr(lsHgt[-1], (mu * fSD / 100.) / dy, order=0, mode='constant'))
        return edges, hgt, lsHgt

    # end method 'mod_pred_dists'

    def get_pdf_ndx(pdf_name):
        '''pdf_ndx - return index for pdf_name'''

        if pdf_name.startswith('prior\nmodel'):
            # ...................... prior
            pdf_ndx = 0
        elif pdf_name.startswith('(A) mpu\nposterior'):
            # ...................... MPU posterior
            pdf_ndx = 1
        elif pdf_name.startswith('prior\npredictive'):
            # ...................... predictive prior
            pdf_ndx = 2
        elif pdf_name.startswith('posterior') or len(pdf_name.split()[0]) == 1:
            # ...................... posterior addtnlUncrtnts/predictive
            pdf_ndx = 3
        else:
            raise RuntimeError("Bad pdf_name = %s" % pdf_name)
        return pdf_ndx

    # end method 'get_pdf_ndx'

    dtMean = {}
    dtStdDev = {}

    # model prior
    edges, h_mod_prior, h_pred_prior = mod_pred_dists(output_preds, np.ones(output_preds.shape[0]), exp_std, num_bins)
    dy = edges[1] - edges[0]
    np.random.seed(seed + 1)
    npModelPrior = np.random.choice(edges[0:-1] + 0.5 * dy, num_pts, p=h_mod_prior / h_mod_prior.sum())

    dtMean['mod_prior'] = np.mean(npModelPrior)
    dtStdDev['mod_prior'] = np.std(npModelPrior)

    # mpu
    edges, h_mod_post, ls_h_pred_post = mod_pred_dists(output_preds, posterior_wts, exp_std, num_bins,
                                                    addl_uncertnts=addtnl_uncrtnts)
    h_pred_post = ls_h_pred_post[-1]
    dy = edges[1] - edges[0]  # bin size
    np.random.seed(seed + 3)
    mod_post = np.random.choice(edges[0:-1] + 0.5 * dy, num_pts, p=h_mod_post / h_mod_post.sum())
    dtMean['mod_post'] = np.mean(mod_post)
    dtStdDev['mod_post'] = np.std(mod_post)

    print h_pred_post
    np.random.seed(seed + 4)
    pred_post = np.random.choice(edges[0:-1] + 0.5 * dy, num_pts, p=h_pred_post / h_pred_post.sum())

    dtMean['pred_post'] = np.mean(pred_post)
    dtStdDev['pred_post'] = np.std(pred_post)

    lsBx = [npModelPrior, mod_post, pred_post]
    pp_label_1 = "model"
    lsXLbl = ['prior\nmodel', '(A) mpu\nposterior', 'posterior\n%s (A +%s%s)' % \
              (pp_label_1, ''.join(['\n+ %s' % tpUnc[0] for tpUnc in addtnl_uncrtnts]))]

    fp = matplotlib.font_manager.FontProperties()
    fp.set_size(dtDflts['txt_size']['t'])
    npBbox = matplotlib.textpath.TextPath((-100, -100), 'box plot', prop=fp).get_extents().get_points()

    fig = plt.figure(figsize=dtDflts['fig_size'])

    fig.subplots_adjust(**(dtDflts['subfig_margins']))
    pos = xrange(1, len(lsBx) + 1)
    if VRSN_GT_1_2:
        bxplt = plt.boxplot(lsBx, sym=dtDflts['mark']['f'], whis=dtDflts['whskr_reach'],
                            patch_artist=dtDflts['patch_artist'], positions=pos, medianprops={'c': 'k'})
    else:
        bxplt = plt.boxplot(lsBx, sym=dtDflts['mark']['f'], whis=dtDflts['whskr_reach'],
                            patch_artist=dtDflts['patch_artist'], positions=pos)

    for j, patch in enumerate(bxplt['boxes']):
        patch.set_facecolor(dtDflts['ln_clr']['f'][get_pdf_ndx(lsXLbl[j])])
        patch.set_edgecolor(dtDflts['ln_clr']['e'][get_pdf_ndx(lsXLbl[j])])
        patch.set_linewidth(dtDflts['ln_width']['b'])

    plt.xticks(pos, lsXLbl, rotation=90)

    fMinYPt = 1.0e+99
    fMaxYPt = -1.0e+99
    for whisker in bxplt['whiskers']:
        npY = whisker.get_ydata()
        fMinYPt = min(npY.min(), fMinYPt)
        fMaxYPt = max(npY.max(), fMaxYPt)
    delta_y = fMaxYPt - fMinYPt
    if dtDflts['whskr_reach'] == [5, 95]:
        # extend range to cover at least 2.5-sigma
        fMaxYPt = fMinYPt + 1.1 * delta_y
        fMinYPt = fMaxYPt - 1.2 * delta_y
        delta_y *= 1.2
    elif dtDflts['whskr_reach'] == [2.5, 97.5]:
        # extend range to cover at least 2.5-sigma
        fMaxYPt = fMinYPt + 1.05 * delta_y
        fMinYPt = fMaxYPt - 1.1 * delta_y
        delta_y *= 1.1
    fMaxYPt = fMaxYPt + 0.2 * delta_y  # reserve space for labels
    fMinYPt = fMinYPt - 0.03 * delta_y  # keep whisker tails within box
    delta_y = fMaxYPt - fMinYPt
    delta_fnt = 1.0 + (dtDflts['txt_size']['a'] - 8) / 20.0

    def sh_str(val):
        '''sh_str - convert val to a "short" string'''
        return "%.2f" % val

    mean = exp_obs
    sd = exp_std
    for idx in xrange(len(lsBx)):  # add the std. dev. values and the experiment mean for each box plot
        npX = bxplt['medians'][idx].get_xdata()
        plt.text(np.average(npX),
                 fMinYPt + (1.0 - 0.08 * delta_fnt) * delta_y,
                 (r'$\mu$=' + sh_str(np.mean(lsBx[idx])) + '\n' + r'$\sigma$=' + sh_str(np.std(lsBx[idx]))),
                 horizontalalignment='center', fontsize=dtDflts['txt_size']['a'], color='k')
        if idx == len(lsBx) - 1:  # if working with the posterior predictive, plot the "observed" mean and std dev
            plt.text(np.average(npX),
                     fMinYPt + (1.0 - 0.15 * delta_fnt) * delta_y,
                     (r'$\mu$=' + sh_str(mean) + '\n' + r'$\sigma$=' + sh_str(sd)),
                     horizontalalignment='center', fontsize=dtDflts['txt_size']['a'],
                     color=dtDflts['ln_clr']['exp'])
            plt.plot(npX, [mean, mean], color=dtDflts['ln_clr']['exp'], linewidth=dtDflts['ln_width']['b'])

    npCaps = bxplt['caps'][-1]
    plt.ylim(fMinYPt, fMaxYPt)
    plt.title(output_name)

    # Add histogram plot to the right using data from the last box plot
    l_ndx = 1
    subFig = plt.subplot(111)
    divider = make_axes_locatable(subFig)
    axHisty = divider.append_axes("right", (1 / 15.0) * dtDflts['fig_size'][0], pad=0.34, sharey=subFig)
    yy = np.ravel(zip(edges, edges))
    xx = np.ravel(zip([0.] + list(h_pred_post), list(h_pred_post) + [0.]))
    axHisty.step(xx, yy, lw=dtDflts['ln_width']['b'], c='k')
    axHisty.fill_betweenx(yy, xx, x2=0, color=dtDflts['ln_clr']['f'][3])
    xmax = 1.1 * np.max(h_pred_post)
    axHisty.set_xlim(0, xmax)
    axHisty.set_ylim(np.min(edges), np.max(edges))
    axHisty.plot([0.0, xmax], [mean, mean], color=dtDflts['ln_clr']['exp'],
                 linewidth=dtDflts['ln_width']['b'])
    xt = axHisty.get_xticks()
    axHisty.set_xticks([xt[0], xt[-1]])

    subFig.set_ylim(fMinYPt, fMaxYPt)

    art.setp(bxplt['whiskers'], color=dtDflts['ln_clr']['tk'], lw=dtDflts['ln_width']['tk'],
             ls=dtDflts['ln_style']['tk'])

    plt.show()
