import argparse
import sys
import numpy as np
import RIFT.lalsimutils as lalsimutils
import lal
import scipy.stats as stats
from ligo.lw import lsctables, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)

parser = argparse.ArgumentParser()
parser.add_argument("--inj-file", help="Name of XML file")
parser.add_argument("--inj-file-out", default="output-puffball", help="Name of XML file")
parser.add_argument("--puff-factor", default=4, type=float)
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
parser.add_argument("--downselect-parameter", action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range", action='append', type=str)
opts=  parser.parse_args()

# Extract parameter names
coord_names = opts.parameter 
if coord_names is None:
    sys.exit(0)
print(f"Co-ordinates to be puffed are: {coord_names}")

# To ensure the puffed values are within provided bounds
downselect_dict = {}
# Add some pre-built downselects, to avoid common out-of-range-error problems
downselect_dict['chi1'] = [0,1]
downselect_dict['chi2'] = [0,1]
downselect_dict['eta'] = [0,0.25]
downselect_dict['m1'] = [0,1e10]
downselect_dict['m2'] = [0,1e10]

if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = list(map(eval,opts.downselect_parameter_range))
else:
    dlist = []
    dlist_ranges = []
    opts.downselect_parameter =[]
if len(dlist) != len(dlist_ranges):
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]

# Load data
P_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj_file)

# Extract parameters
dat_out = []
for P in P_list:
    line_out = np.zeros(len(coord_names))
    for x in np.arange(len(coord_names)):
        fac=1
        if coord_names[x] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        line_out[x] = P.extract_param(coord_names[x])/fac     
    dat_out.append(line_out)

# relabel data
dat_out = np.array(dat_out)
X = dat_out[:,0:len(coord_names)]

# kde
kde = stats.gaussian_kde(X.T)
# broaden it
kde.covariance = float(opts.puff_factor)*kde.covariance 
# generate new samples
X_out = kde.resample(2*len(X)).T # final shape: (total_samples, params)


# Copy parameters back in.  MAKE SURE THIS IS POSSIBLE
P_out = []
for indx_P in np.arange(len(P_list)):
    include_item=True
    P = P_list[indx_P]
    for indx in np.arange(len(coord_names)):
        fac=1
        # sanity check restrictions, which may cause problems with the coordinate converters
        if coord_names[indx] == 'eta' and (X_out[indx_P,indx]>0.25) :
            continue
        if coord_names[indx] == 'delta_mc' and (X_out[indx_P,indx]>1) :
            continue
        if coord_names[indx] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        P_list[indx_P].assign_param( coord_names[indx], X_out[indx_P,indx]*fac)

    if np.isnan(P.m1) or np.isnan(P.m2):  # don't allow nan mass
        continue

    for param in downselect_dict:
        val = P.extract_param(param)
        if np.isnan(val):
            include_item=False   # includes check on m1,m2
            continue # stop trying to calculate with this parameter
        if param in ['mc','m1','m2','mtot']:
            val = val/ lal.MSUN_SI
        if val < downselect_dict[param][0] or val > downselect_dict[param][1]:
            include_item =False
    if include_item:
        P_out.append(P)

print(f"The number of exported points is {len(P_out)}, original number of points is {len(P_list)}")

# Export
lalsimutils.ChooseWaveformParams_array_to_xml(P_out,fname=opts.inj_file_out,fref=P.fref)
