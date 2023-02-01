import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import pickle
import tqdm
from rdkit.Chem import AllChem
from  rdkit import DataStructs
from sklearn.metrics import mean_squared_error

folder = "BOAW_bindb/"

df_left = pd.read_csv(folder +  "results/binding.csv")

model_tile = "BOAW Random Forest Model n = 10"

plt_file_name = "BOAW_RF10_mean_predictor"

print(df_left.columns)

total_df = pd.DataFrame(pickle.load(open("binding_data.p", "rb"))).reset_index()

######


def compare_same_dataset(mols):

    fps = [ ]

    to_remove = []

    for m in mols:

        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048))

    maxes = []

    i = 0

    for fp in fps:

        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)

        sims[i] = 0

        if max(sims) < 0.6:

            to_remove.append(i)

        else:

            maxes.append(np.mean(np.sort(sims)[::-1][0:min([10, len(fps)])]))

        i+=1

    m_sim = np.median(maxes)

    return to_remove,m_sim


IC50s = {}

for i, r in tqdm.tqdm([(i, r) for i, r in total_df.iterrows()]):

    IC50 = []
    mols = []

    for v, m in zip(r['IC50 (nM)'], r['Mols']):

        v = str(v)

        v = v.replace(" ", "")

        if (">" not in v) and ("<" not in v):

            v = float(v)

            if not np.isnan(v):
                IC50.append(np.log(v))
                mols.append(m)

    to_remove, m_sim = compare_same_dataset(mols)

    mols = np.delete(np.array(mols), to_remove)

    IC50 = np.delete(np.array(IC50), to_remove)

    IC50s[r['ID']] = IC50


######

vs = []

for i,r in df_left.iterrows():

    IC50_ = IC50s[r['file']]

    mean_errors =  np.array(IC50_) -  np.median(IC50_)

    rmse = mean_squared_error(IC50_, [np.median(IC50_) for j in IC50_])

    print(rmse)
    print(r["RMSE"])

    vs.append(r['RMSE'] / rmse)

print(vs)

df_left['RMSE'] = vs

x_ = np.linspace(min(df_left['RMSE']) , max(df_left['RMSE']),100)

sim_x = np.linspace(0.5,1,10)

colors= (sim_x - 0.5 )/ 0.5
print(colors)
colors = plt.cm.plasma(colors)

i = 1

for x in sim_x[1:]:

    errors = []
    lengths = []

    for error , sim,length in zip(df_left['RMSE'],df_left['mean_sim'],df_left['length']):

        if (sim <= x ) and ( sim >  sim_x[i-1]) :

            errors.append(error)
            lengths.append(length)

    if len(errors) > 10:

        kde = gaussian_kde(errors)

        plt.plot(x_,kde.pdf(x_),label =str(round(sim_x[i-1],2))  + " < mean sim =< " + str(round(x,2))  +" n series = "  + str(len(errors)) + " median length = " + str(int(np.median(lengths))) + "\nmedian RMSE/(task stddev) = " + str(round(np.mean(errors),2)) ,linewidth = 3, color = colors[i]  )

    i+=1

plt.title(model_tile + " Average Series Length = " + str(np.median(df_left['length'])))

plt.xlabel("RMSE/(RMSE no skill model)")
plt.ylabel("Density")
plt.legend()

plt.savefig(folder + "/" + plt_file_name + "_RMSE_vs_sim.png",bbox_inches='tight')
plt.savefig(folder + "/" + plt_file_name + "_RMSE_vs_sim.svg",format = "SVG",bbox_inches='tight')
plt.close()

############


len_x = np.linspace(min(df_left['length']) , max(df_left['length']),10)




len_x = [int(x) for x in len_x]

print(len_x)


x_ = np.linspace(min(df_left['RMSE']) , max(df_left['RMSE']),100)


colors = (sim_x - 0.5 )/ 0.5
print(colors)
colors = plt.cm.plasma(colors)

i = 1

for x in len_x[1:]:

    print(x,len_x[i - 1])

    errors = []
    lengths = []

    for error , length in zip(df_left['RMSE'],df_left['length']):

        if (length <= x ) and ( length >  len_x[i - 1]) :

            errors.append(error)
            lengths.append(length)

    if len(errors) > 10:

        kde = gaussian_kde(errors)

        plt.plot(x_,kde.pdf(x_),linewidth = 3, color = colors[i],label = str(len_x[i-1]) + " < length =< " + str(x) + "\nn series = "  + str(len(errors)) + "\nmedian RMSE/(task stddev) = " + str(round(np.mean(errors),2))  )

    i+=1



plt.title(model_tile + " Average Series Length = " + str(np.median(df_left['length'])))

plt.xlabel("RMSE/(no skill model)")
plt.ylabel("Density")
plt.legend()

plt.savefig(folder + "/" + plt_file_name + "_RMSE_vs_length.png",bbox_inches='tight')
plt.savefig(folder + "/" + plt_file_name + "_RMSE_vs_length.svg",format = "SVG",bbox_inches='tight')
plt.close()


###########


i = 1

errors = []
lengths = [ ]
xs_ = []

i = 1

for x in sim_x[1:]:

    print(x,sim_x[i-1])

    errors_ = []
    lengths_ = []

    for error , sim,length in zip(df_left['RMSE'],df_left['mean_sim'],df_left['length']):

        if (sim <= x ) and ( sim > sim_x[i-1]) :

            errors_.append(error)
            lengths_.append(length)

    if len(errors_) > 5:
        errors.append(errors_)
        lengths.append(lengths_)
        xs_.append(x)

    i+=1

f, axs = plt.subplots( len(errors),1, sharex=True,sharey=True)

i = 0

for x,errors_,lengths_ in zip(xs_,errors,lengths):

    axs[i].plot(errors_,lengths_,"o" , color = colors[i] ,label = str(round(sim_x[i],2)) + " < mean sim =< " + str(round(sim_x[i +1],2 )) +"\nn series = "  + str(len(errors_)) + "\nmedian RMSE/(task stddev) = " + str(round(np.mean(errors_),2)),alpha = 0.4 )
    axs[i].set_ylabel("Series Length")
    axs[i].set_xlabel("RMSE/(no skill model)")
    axs[i].set_ylim([0,200])
    axs[i].legend()

    i+=1



plt.savefig(folder + "/" + plt_file_name + "_RMSE.png",bbox_inches='tight')
plt.savefig(folder + "/" + plt_file_name + "_RMSE.svg",format = "SVG",bbox_inches='tight')

plt.close()
