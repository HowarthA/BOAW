import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

#df_x = pd.read_csv("../QM_Modelling/Binding_set_avSOAP_RF10/binding.csv").reset_index()

df_x = pd.read_csv("bindb_MF/results/binding.csv")

df_x = df_x[['file','RMSE','R2_of_means','length','RMSE_No_skill']]

df_y = pd.read_csv("BOAW_bindb_confs/results/binding.csv").reset_index()

#df_y = pd.read_csv("../QM_Modelling/Binding_set_MF_RF10/binding.csv")

df_y = df_y[['file','RMSE','R2_of_means','length','RMSE_No_skill']]

#print(np.average(df_x['RMSE_No_skill']))

#print(np.average(df_y['RMSE_No_skill']))


suffix_x = "MF_RF10"


suffix_y = "BOAW_RF10"


l = min([len(df_x),len(df_y)])

out_f = "BOAW_bindb_confs/" + suffix_y +"Vs" + suffix_x

df_t = df_x.merge(df_y,how = 'inner',on = 'file',suffixes=(suffix_x,suffix_y))

df_t = df_t.drop_duplicates()

RMSE_count = 0

R2_count = 0

for x,y in zip(df_t['R2_of_means' + str(suffix_x)] , df_t['R2_of_means' + str(suffix_y)]):

    if x > y:

        R2_count+=1

print("X better than Y in " + str(R2_count/ len(df_t )) + " by R2")

for x, y in zip(df_t['RMSE' + str(suffix_x)], df_t['RMSE' + str(suffix_y)]):

    if x < y:
        RMSE_count += 1

print("X better than Y better in " + str(RMSE_count /len(df_t ) ) + " by RMSE")

df_t['length' + str(suffix_x)] = (df_t['length' + str(suffix_x)] - min(df_t['length' + str(suffix_x)] ))/(max(df_t['length' + str(suffix_x)]) - min(df_t['length' + str(suffix_x)] ))
df_t=df_t.sort_values(by='length' + str(suffix_x), ascending=True)

plt.scatter(df_t['RMSE' + str(suffix_x)] ,df_t['RMSE' + str(suffix_y)],alpha = 0.8,c = [i for i in df_t['length' + str(suffix_x)]],cmap = 'viridis')

plt.plot([min(df_t['RMSE' + str(suffix_x)]),max(df_t['RMSE' + str(suffix_x)])], [min(df_t['RMSE' + str(suffix_x)]),max(df_t['RMSE' + str(suffix_x)])],linestyle = ":", color ='black',linewidth = 2)
plt.xlabel("RMSE" + suffix_x)
plt.ylabel("RMSE" + suffix_y)
plt.title("X performs better than Y in "+str(round(100* RMSE_count / len(df_t ),2)) + "% of cases")
plt.xlim([0,10])
plt.ylim([0,10])
plt.savefig( out_f + "_RMSE.png")
plt.close()

plt.scatter(df_t['R2_of_means' + str(suffix_x)] ,df_t['R2_of_means' + str(suffix_y)],alpha = 0.8,c = [i for i in df_t['length' + str(suffix_x)]],cmap = 'viridis')

plt.plot([min(df_t['R2_of_means' + str(suffix_x)]),max(df_t['R2_of_means' + str(suffix_x)])], [min(df_t['R2_of_means' + str(suffix_x)]),max(df_t['R2_of_means' + str(suffix_x)])],linestyle = ":", color ='black',linewidth = 2 )
plt.xlabel("R2_of_means" + suffix_x)
plt.ylabel("R2_of_means" + suffix_y)
plt.title("X performs better than Y in "+str( round(100* R2_count / len(df_t ),2)) + "% of cases")

plt.savefig(out_f + "_R2.png")

plt.close()

#### plot RMSE and RMSE/no skill KDES

x = np.linspace(0,max(df_x['RMSE']),1000)

RMSE_x_kde = gaussian_kde(df_x['RMSE'])

RMSE_y_kde = gaussian_kde(df_y['RMSE'])

plt.plot(x,RMSE_x_kde.pdf(x), label = suffix_x,color= "deepskyblue")
plt.plot(x,RMSE_y_kde.pdf(x), label = suffix_y,color = "crimson")

plt.xlabel("RMSE")
plt.ylabel("Density")
plt.legend()

plt.title("Comparing RMSE")

plt.savefig(out_f + "_RMSE_KDE.png")

plt.close()

RMSE_x_kde = gaussian_kde(df_x['RMSE_No_skill'])

RMSE_y_kde = gaussian_kde(df_y['RMSE_No_skill'])

x = np.linspace(0,max(df_x['RMSE_No_skill']),1000)

plt.plot(x,RMSE_x_kde.pdf(x), label = suffix_x,color='deepskyblue')
plt.plot(x,RMSE_y_kde.pdf(x), label = suffix_y,color = 'crimson')

plt.legend()
plt.xlabel("RMSE_No_skill")
plt.ylabel("Density")

plt.title("Comparing RMSE/(No skill)")

plt.savefig(out_f + "_RMSENoSkill_KDE.png")

plt.close()

