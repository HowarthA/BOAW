import pandas as pd
import os

folder = "bindb_SOAP/results/"

df_total = []

for f in os.listdir(folder):

    if f.endswith(".csv"):

        df  = pd.read_csv(folder  + f)
        for i ,r in df.iterrows():

            try:

                df_total.append([r['file'], int(r['length']), float(r['mean_sim']), float(r['R2_of_means']),float( r['RMSE']),float(r['std_errors']), float(r['RMSE_No_skill'])])
            except:

                print("broken" , r)


df_total = pd.DataFrame(df_total , columns=['file', 'length', 'mean_sim', 'R2_of_means', 'RMSE','std_errors', 'RMSE_No_skill'])
print(df_total)

df_total.to_csv(folder + "binding.csv")