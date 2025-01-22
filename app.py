import numpy as np
from cmath import pi, sqrt
import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from SGMPy_material import *
from SGMPy_section_v01 import *
import plotly.graph_objs as go


input_data = {}  

Sollecitazioni = {'G1+':{'N': -0.0, 'T': 0.0, 'Mf': 9.58, 'Mt': 0.0}, 'G1-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0}, # peso proprio
                  'G2+':{'N': 0.0, 'T': 0.0, 'Mf': 7.55, 'Mt': 0.0}, 'G2-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0}, # permanenti portati
                  'R+':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0}, 'R-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0},  # ritiro
                  'Mfat+':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0}, 'Mfat-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0}, # fatica
                  'MQ+':{ 'N': 0.0, 'T': 0.0, 'Mf': 1274, 'Mt': 0.0}, 'MQ-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0}, # mobili concentrati
                  'Md+':{'N': 0.0, 'T': 0.0, 'Mf': 138, 'Mt': 0.0}, 'Md-':{ 'N': 0.0, 'T': 0.0, 'Mf': 1200.0, 'Mt': 0.0}, # mobili distribuiti
                  'Mf+':{ 'N': 0.0, 'T': 0.0, 'Mf': 800.0, 'Mt': 0.0}, 'Mf-':{ 'N': 0.0, 'T': 0.0, 'Mf': 1200.0, 'Mt': 0.0}, # folla
                  'T+':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0}, 'T-':{ 'N': 0.0, 'T': 0.0, 'Mf': 1200.0, 'Mt': 0.0},   # termica
                  'C+':{ 'N': 0.0, 'T': 0.0, 'Mf': 10.0, 'Mt': 0.0}, 'C-':{ 'N': 0.0, 'T': 0.0, 'Mf': 1200.0, 'Mt': 0.0},   # cedimenti
                  'V+':{ 'N': 0.0, 'T': 0.0, 'Mf': 10.0, 'Mt': 0.0}, 'V-':{ 'N': 0.0, 'T': 0.0, 'Mf': 1200.0, 'Mt': 0.0},   # vento
        }

def combinazione(list_sigma):
   
   path = "coefficienti.xlsx"

   # list_sigma = [g1_sigma_plot,
   #             g2_sigma_plot,
   #             r_sigma_plot,
   #             fat_sigma_plot,
   #             ts_sigma_plot,
   #             udl_sigma_plot,
   #             folla_sigma_plot,
   #             t_sigma_plot,
   #             c_sigma_plot,
   #             v_sigma_plot,
   #             ]

   sigma_slu = list_sigma[0] #G1
   + list_sigma[0] #G2
   + list_sigma[1] #Ritiro
   + list_sigma[2] #Carichi mobili concentrati
   + list_sigma[3] #Carichi mobili distribuiti
   + list_sigma[4] #Carico folla
   + list_sigma[5] #temperatura
   + list_sigma[6] #cedimenti
   + list_sigma[7] #vento

   

   gamma = pd.read_excel(path, "SLU", index_col=0)
   psi = pd.read_excel(path, "SLE", index_col=0)
   #print(gamma.loc['g1', 'A1_sfav'])
   #print(psi)
   
   # Nslu_pos = (gamma.loc['g1', 'A1_sfav'] * Sollecitazioni['G2+']['N'] + 
   #            + gamma.loc['g2', 'A1_sfav'] * Sollecitazioni['G2+']['N']) 

   return


## START --> Sollecitazioni
# Convert dictionary to DataFrame
df = pd.DataFrame(Sollecitazioni).T.reset_index()
df.rename(columns={'index': 'Tipo'}, inplace=True)

# Editable table using st.data_editor
st.title("Edit Sollecitazioni")
st.write("Modify the values in the table below:")

# Display the editable data table
edited_df = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",  # Allow adding/removing rows
)

# Save changes and convert back to dictionary
updated_dict = edited_df.set_index('Tipo').T.to_dict()
st.json(updated_dict, expanded=False)

combinazione(Sollecitazioni)


input_section = { 'l': [12*1000], #lunghezza del concio
                 
                  "bPsup": [300], #larghezza piattabanda superiore
                  'tPsup': [20], #spessore piattabanda superiore
                  "brPsup": [0], # larghezza raddoppio piattabanda superiore
                  'trPsup': [0], #spessore raddoppio piattabanda superiore
                  "ha": [360], #altezza anima
                  "ta": [10], #spessore anima
                  "brPinf": [0], #larghezza raddoppio piattabanda inferiore
                  'trPinf': [0], #spessore raddoppio piattabanda inferiore
                  "bPinf": [350], #larghezza piattabanda inferiore
                  'tPinf': [20], #spessore piattabanda inferiore

                  "hcls": [190],
                  "h_predalle": [50],
                  "Bcls": [1000],

                  "phi_sup": [16], 
                  "int_sup": [100],
                  "int_inf": [100],
                  "phi_inf": [16], 
                  "int_inf": [100],

                  "n_inf": [15], 
                  "n_0": [6],
                  "n_r": [18], 
                  "n_c": [17],
                  }


# Convert dictionary to DataFrame
df_section = pd.DataFrame.from_dict(input_section, orient = "index") #.reset_index()


# Editable table using st.data_editor
st.title("Edit Sezione")
st.write("Modify the values in the table below:")

# Display the editable data table
edited_df = st.data_editor(
    df_section,
    use_container_width=True,
    num_rows="dynamic",  # Allow adding/removing rows
)

# convert back to dictionary
updated_dict = edited_df.to_dict()
st.json(updated_dict, expanded=False)



##INPUT
# Extract updated values from the table
Hcls = float(edited_df.loc["hcls"][0])
Bcls = float(edited_df.loc['Bcls'][0])
hpredall = float(edited_df.loc['h_predalle'][0])
phi_sup = float(edited_df.loc['phi_sup'][0])
int_sup = float(edited_df.loc['int_sup'][0])
phi_inf = float(edited_df.loc['phi_inf'][0])
int_inf = float(edited_df.loc['int_inf'][0])

bf = float(edited_df.loc['bPsup'][0])
tbf = float(edited_df.loc['tPsup'][0])

bf_r = float(edited_df.loc['brPsup'][0])
tbrf = float(edited_df.loc['trPsup'][0])

hw = float(edited_df.loc['ha'][0])
tw = float(edited_df.loc['ta'][0])

rbf_inf = float(edited_df.loc['brPinf'][0])
rtbf_inf = float(edited_df.loc['trPinf'][0])

binf = float(edited_df.loc['bPinf'][0])
tbf_inf = float(edited_df.loc['tPinf'][0])

ninf = float(edited_df.loc['n_inf'][0])
n0 = float(edited_df.loc['n_0'][0])
nr = float(edited_df.loc['n_r'][0])
nc = float(edited_df.loc['n_c'][0])



## COSTRUZIONE SOLETTA IN CALCESTRUZZO
clsSection = RectangularSection(Bcls, Hcls, [0, 0], material="C25/30")
pointG0 = [[-int_sup*i, 35] for i in range(0, int(Bcls*0.5/100))] + [[int_inf*i, 35] for i in range(1, int(Bcls*0.5/100))]
pointG1 = [[-int_inf*i, Hcls-35] for i in range(0, int(Bcls*0.5/100))] + [[int_inf*i, Hcls-35] for i in range(1, int(Bcls*0.5/100))]
b0 = renforcementBar(phi_sup, pointG0)
b1 = renforcementBar(phi_inf, pointG1)
# = rectangularCA(clsSection, [b0, b1])
#cplot = plotSection_ploty(c)
#cplot.show()


## COSTRUZIONE SEZIONE IN ACCIAIO 
gapCls = hpredall+Hcls
PlateSup = OrizontalPlate(bf, tbf, [0, gapCls], material="S355")
rPlateSup = OrizontalPlate(bf_r, tbrf, [0, gapCls+tbf], material="S355")
#vribs1 = V_rib(283, 300, 25, 6, [0, 70+16]) #cl4Dict={"Binst":75, "Be1":60}
wPlate1 = WebPlate(hw, tw, [0, gapCls+tbf+tbrf], 0, material="S355", cl4Dict=None)
rPlateInf = OrizontalPlate(rbf_inf, rtbf_inf, [0, gapCls+tbf+tbrf+hw], material="S355")
PlateInf = OrizontalPlate(binf, tbf_inf, [0, (gapCls+tbf+tbrf+hw+rtbf_inf)], material="S355")

# if tbf == 0 or tw == 0 or tbf_inf == 0:
#    st.warning("parametri essenziali non settati")

# if tbrf == 0 and rtbf_inf == 0:
#    listDict = [PlateSup, wPlate1, PlateInf]

# elif tbrf == 0 and rtbf_inf != 0:
#    listDict = [PlateSup, wPlate1, rPlateInf, PlateInf]

# elif tbrf != 0 and rtbf_inf == 0:
#    listDict = [PlateSup, rPlateSup, wPlate1, PlateInf]

# elif tbrf != 0 and rtbf_inf != 0:
#    listDict = [PlateSup,rPlateSup, wPlate1, rPlateInf, PlateInf]


listDict = [PlateSup,rPlateSup, wPlate1, rPlateInf, PlateInf]
Isection = builtSection(listDict)


#print(Isection)

SectionComposite_I = CompositeSection(Isection, clsSection, [b0, b1], n0)
cplot = plotSection_ploty(SectionComposite_I)
#plot section in STREAMLIT
st.plotly_chart(cplot , use_container_width=True)
#cplot.show()

listParams = ["A", "Ay", "Az", "Iy", "Iz", "It", "Pg", "ay", "az"]
dictProp = {}
table = {}

#Fase 1 - Only steel
dictProp["g1"] = Isection

#Fase 2 - G2 - t inf
SectionComposite_g2 = CompositeSection(Isection, clsSection, [b0, b1], ninf)
dictProp["g2"] = SectionComposite_g2


#Fase 3 - R - ritiro
SectionComposite_r = CompositeSection(Isection, clsSection, [b0, b1], nr)
dictProp["r"] = SectionComposite_r 

#Fase 4 - C - cedimenti
SectionComposite_c = CompositeSection(Isection, clsSection, [b0, b1], nc)
dictProp["c"] = SectionComposite_c

#Fase 5 - Qm - mobili
SectionComposite_m = CompositeSection(Isection, clsSection, [b0, b1], n0)
dictProp["mobili"] = SectionComposite_m

#Fase 6 - Fessurato
clsSection_F = RectangularSection(0.1, 0.1, [0, 0], material="C25/30")
SectionComposite_f = CompositeSection(Isection, clsSection_F, [b0, b1], 100000)
dictProp["fe"] = SectionComposite_f

# per print table
table["g1"] = { i :dictProp["g1"][i] for i in listParams}
table["g2"] = { i :dictProp["g2"][i] for i in listParams}
table["r"] = { i :dictProp["r"][i] for i in listParams}
table["c"] = { i :dictProp["c"][i] for i in listParams}
table["mobili"] = { i :dictProp["mobili"][i] for i in listParams}
table["fe"] = { i :dictProp["fe"][i] for i in listParams}

df_section_prop = pd.DataFrame.from_dict(table, orient = "index").T #.reset_index()
#st.write(df_section_prop)


## CALCOLO TENSIONI
tension = {'G1+':{'sigma': 0.0, 'tau': 0.0}, 'G1-':{'sigma': 0.0, 'tau': 0.0}, # peso proprio
                  'G2+':{'sigma': 0.0, 'tau': 0.0}, 'G2-':{'sigma': 0.0, 'tau': 0.0}, # permanenti portati
                  'R+':{'sigma': 0.0, 'tau': 0.0}, 'R-':{'sigma': 0.0, 'tau': 0.0},  # ritiro
                  'Mfat+':{'sigma': 0.0, 'tau': 0.0}, 'Mfat-':{'sigma': 0.0, 'tau': 0.0}, # fatica
                  'MQ+':{'sigma': 0.0, 'tau': 0.0}, 'MQ-':{'sigma': 0.0, 'tau': 0.0}, # mobili concentrati
                  'Md+':{'sigma': 0.0, 'tau': 0.0}, 'Md-':{ 'sigma': 0.0, 'tau': 0.0}, # mobili distribuiti
                  'Mf+':{ 'sigma': 0.0, 'tau': 0.0}, 'Mf-':{ 'sigma': 0.0, 'tau': 0.0}, # folla
                  'T+':{ 'sigma': 0.0, 'tau': 0.0}, 'T-':{ 'sigma': 0.0, 'tau': 0.0},   # termica
                  'C+':{ 'sigma': 0.0, 'tau': 0.0}, 'C-':{ 'sigma': 0.0, 'tau': 0.0},   # cedimenti
                  'V+':{ 'sigma': 0.0, 'tau': 0.0}, 'V-':{'sigma': 0.0, 'tau': 0.0},   # vento
        }

hi = [0, gapCls, gapCls, gapCls+tbf, gapCls+tbf+tbrf, gapCls+tbf+tbrf+hw, gapCls+tbf+tbrf+hw+rtbf_inf, gapCls+tbf+tbrf+hw+rtbf_inf+tbf_inf]



def tension(dictProp, Sollecitazioni, condition = "positive"):

   posList = ["G1+", "G2+", 'R+', 'Mfat+', 'MQ+', 'Md+', 'Mf+','T+', 'C+', 'V+']
   negList = ["G1-", "G2-", 'R-', 'Mfat-', 'MQ-', 'Md-', 'Mf-','T-', 'C-', 'V-']

   hi_plot = list(hi) + [hi[-1], hi[0], hi[0]]

   ## G1+
   if condition == "positive":
      N = Sollecitazioni[posList[0]]["N"]
      Mf = Sollecitazioni[posList[0]]["Mf"]
   else:
      N = Sollecitazioni[negList[0]]["N"]
      Mf = Sollecitazioni[negList[0]]["Mf"]

   g1_sigmaN = N*1000/dictProp["g1"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["g1"]["Pg"][1]
   g1_sigmaMf = (Mf*1000/dictProp["g1"]["Iy"])*hg #contributo per momento flettente
   g1_sigma = g1_sigmaN + g1_sigmaMf
   g1_sigma[0], g1_sigma[1] = 0, 0
   g1_sigma_plot = list(g1_sigma) + [0.0, 0.0, g1_sigma[0]]

   #st.write(g1_sigma_plot)
   
   ## G2+
   if condition == "positive":
      N = Sollecitazioni[posList[1]]["N"]
      Mf = Sollecitazioni[posList[1]]["Mf"]
   else:
      N = Sollecitazioni[negList[1]]["N"]
      Mf = Sollecitazioni[negList[1]]["Mf"]

   g2_sigmaN = N*1000/dictProp["g2"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["g2"]["Pg"][1]
   g2_sigmaMf = (Mf*1000**2/dictProp["g2"]["Iy"])*hg #contributo per momento flettente
   g2_sigma = g2_sigmaN + g2_sigmaMf
   g2_sigma[0], g2_sigma[1] =  g2_sigma[0]/ninf, g2_sigma[1]/ninf
   g2_sigma_plot = list(g2_sigma) + [0.0, 0.0, g2_sigma[0]]

   #st.write(g2_sigma_plot)

   ## R+ (CONTROLLARE)
   if condition == "positive":
      N = Sollecitazioni[posList[2]]["N"]
      Mf = Sollecitazioni[posList[2]]["Mf"]
   else:
      N = Sollecitazioni[negList[2]]["N"]
      Mf = Sollecitazioni[negList[2]]["Mf"]

   r_sigmaN = N*1000/dictProp["r"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["r"]["Pg"][1]
   r_sigmaMf = (Mf*1000**2/dictProp["r"]["Iy"])*hg #contributo per momento flettente
   r_sigma = r_sigmaN + r_sigmaMf
   r_sigma[0], r_sigma[1] =  r_sigma[0]/nr, r_sigma[1]/nr
   r_sigma_plot = list(r_sigma) + [0.0, 0.0, r_sigma[0]]

   ## Mfat+ Fatica (CONTROLLARE)
   if condition == "positive":
      N = Sollecitazioni[posList[3]]["N"]
      Mf = Sollecitazioni[posList[3]]["Mf"]
   else:
      N = Sollecitazioni[negList[3]]["N"]
      Mf = Sollecitazioni[negList[3]]["Mf"]

   fat_sigmaN = N*1000/dictProp["mobili"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["mobili"]["Pg"][1]
   fat_sigmaMf = (Mf*1000**2/dictProp["mobili"]["Iy"])*hg #contributo per momento flettente
   fat_sigma = fat_sigmaN + fat_sigmaMf
   fat_sigma[0], fat_sigma[1] =  fat_sigma[0]/n0, fat_sigma[1]/n0
   fat_sigma_plot = list(fat_sigma) + [0.0, 0.0, fat_sigma[0]]

   #st.write(c_sigma_plot)

   ## MQ+ Mobili concentrati
   if condition == "positive":
      N = Sollecitazioni[posList[4]]["N"]
      Mf = Sollecitazioni[posList[4]]["Mf"]
   else:
      N = Sollecitazioni[negList[4]]["N"]
      Mf = Sollecitazioni[negList[4]]["Mf"]

   ts_sigmaN = N*1000/dictProp["mobili"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["mobili"]["Pg"][1]
   ts_sigmaMf = (Mf*1000**2/dictProp["mobili"]["Iy"])*hg #contributo per momento flettente
   ts_sigma = ts_sigmaN + ts_sigmaMf
   ts_sigma[0], ts_sigma[1] =  ts_sigma[0]/n0, ts_sigma[1]/n0
   ts_sigma_plot = list(ts_sigma) + [0.0, 0.0, ts_sigma[0]]

   #st.write(ts_sigma_plot)

   ## Md+ Mobili distribuiti
   if condition == "positive":
      N = Sollecitazioni[posList[5]]["N"]
      Mf = Sollecitazioni[posList[5]]["Mf"]
   else:
      N = Sollecitazioni[negList[5]]["N"]
      Mf = Sollecitazioni[negList[5]]["Mf"]

   udl_sigmaN = N*1000/dictProp["mobili"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["mobili"]["Pg"][1]
   udl_sigmaMf = (Mf*1000**2/dictProp["mobili"]["Iy"])*hg #contributo per momento flettente
   udl_sigma = udl_sigmaN + udl_sigmaMf
   udl_sigma[0], udl_sigma[1] =  udl_sigma[0]/n0, udl_sigma[1]/n0
   udl_sigma_plot = list(udl_sigma) + [0.0, 0.0, udl_sigma[0]]

   ## Mf+ Mobili folla
   if condition == "positive":
      N = Sollecitazioni[posList[6]]["N"]
      Mf = Sollecitazioni[posList[6]]["Mf"]
   else:
      N = Sollecitazioni[negList[6]]["N"]
      Mf = Sollecitazioni[negList[6]]["Mf"]

   folla_sigmaN = N*1000/dictProp["mobili"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["mobili"]["Pg"][1]
   folla_sigmaMf = (Mf*1000**2/dictProp["mobili"]["Iy"])*hg #contributo per momento flettente
   folla_sigma = folla_sigmaN + folla_sigmaMf
   folla_sigma[0], folla_sigma[1] =  folla_sigma[0]/n0, folla_sigma[1]/n0
   folla_sigma_plot = list(folla_sigma) + [0.0, 0.0, folla_sigma[0]]

   ## T+ termica(CONTROLLARE)
   if condition == "positive":
      N = Sollecitazioni[posList[7]]["N"]
      Mf = Sollecitazioni[posList[7]]["Mf"]
   else:
      N = Sollecitazioni[negList[7]]["N"]
      Mf = Sollecitazioni[negList[7]]["Mf"]

   t_sigmaN = N*1000/dictProp["c"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["c"]["Pg"][1]
   t_sigmaMf = (Mf*1000**2/dictProp["c"]["Iy"])*hg #contributo per momento flettente
   t_sigma = t_sigmaN + t_sigmaMf
   t_sigma[0], t_sigma[1] =  t_sigma[0]/ninf, t_sigma[1]/ninf
   t_sigma_plot = list(t_sigma) + [0.0, 0.0, t_sigma[0]]

   ## C+ (CONTROLLARE)
   if condition == "positive":
      N = Sollecitazioni[posList[8]]["N"]
      Mf = Sollecitazioni[posList[8]]["Mf"]
   else:
      N = Sollecitazioni[negList[8]]["N"]
      Mf = Sollecitazioni[negList[8]]["Mf"]

   c_sigmaN = N*1000/dictProp["c"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["c"]["Pg"][1]
   c_sigmaMf = (Mf*1000**2/dictProp["c"]["Iy"])*hg #contributo per momento flettente
   c_sigma = c_sigmaN + c_sigmaMf
   c_sigma[0], c_sigma[1] =  c_sigma[0]/nc, c_sigma[1]/nc
   c_sigma_plot = list(c_sigma) + [0.0, 0.0, c_sigma[0]]

   ## V+ vento(CONTROLLARE)
   if condition == "positive":
      N = Sollecitazioni[posList[9]]["N"]
      Mf = Sollecitazioni[posList[9]]["Mf"]
   else:
      N = Sollecitazioni[negList[9]]["N"]
      Mf = Sollecitazioni[negList[9]]["Mf"]

   v_sigmaN = N*1000/dictProp["c"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["c"]["Pg"][1]
   v_sigmaMf = (Mf*1000**2/dictProp["c"]["Iy"])*hg #contributo per momento flettente
   v_sigma = v_sigmaN + v_sigmaMf
   v_sigma[0], v_sigma[1] =  v_sigma[0]/ninf, v_sigma[1]/ninf
   v_sigma_plot = list(v_sigma) + [0.0, 0.0, v_sigma[0]]

   #st.write(udl_sigma_plot)

   list_sigma = [g1_sigma_plot,
                 g2_sigma_plot,
                 r_sigma_plot,
                 fat_sigma_plot,
                 ts_sigma_plot,
                 udl_sigma_plot,
                 folla_sigma_plot,
                 t_sigma_plot,
                 c_sigma_plot,
                 v_sigma_plot,
                  ]

   sigma_tot_plot = np.sum(list_sigma, axis=0)
   
   list_sigma.append(sigma_tot_plot)
   
   list_fig = []
   for i_sigma in list_sigma:

      fig = go.Figure()

      #fig.add_trace(go.Scatter(x=g1_sigma_plot, y=hi_plot, fill='tozeroy', name = "tensione g1")) 
      #fig.add_trace(go.Scatter(x=g2_sigma_plot, y=hi_plot, fill='tozeroy', name = "tensione g2")) 
      fig.add_trace(go.Scatter(x=i_sigma, y=hi_plot, fill='tozeroy')) 

      fig.update_layout(
         title=dict(
            text="Tensione sulla sezione"
         ),
         xaxis=dict(
            title=dict(
                  text="tensione [MPa]"
            )
         ),
         yaxis=dict(
            title=dict(
                  text="ascissa sull'altezza della sezione [mm]"
            )
         ),
         # legend=dict(
         #    title=dict(
         #          text="Legend Title"
         #    )
         #),
         font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
         )
      )
      
      fig.update_layout(yaxis = dict(autorange="reversed"))

      list_fig.append(fig)


   return list_fig, list_sigma



## PLOT TENSION POSITIVE
tension_plot_plus, list_tension = tension(dictProp, Sollecitazioni)

# using naive method
# to convert lists to dictionary
test_keys = ["g1", "g2", "r", "Mf", "Mts", "Mudl", "Mfolla", "T", "C", "V", "totale"]
tension_table_print = {}
for i, key in enumerate(test_keys):
   tension_table_print[key] = list_tension[i]

st.title("Tensioni M+")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11  = st.tabs(test_keys)

with tab1:
   st.plotly_chart(tension_plot_plus[0], use_container_width=True, key= "tension_g1+")
with tab2:
   st.plotly_chart(tension_plot_plus[1], use_container_width=True, key= "tension_g2+")
with tab3:
   st.plotly_chart(tension_plot_plus[2], use_container_width=True, key= "tension_r+")
with tab4:
   st.plotly_chart(tension_plot_plus[3], use_container_width=True, key= "tension_Mf+")
with tab5:
   st.plotly_chart(tension_plot_plus[4], use_container_width=True, key= "tension_Mts+")
with tab6:
   st.plotly_chart(tension_plot_plus[5], use_container_width=True, key= "tension_Mudl+")
with tab7:
   st.plotly_chart(tension_plot_plus[6], use_container_width=True, key= "tension_Mfolla+")
with tab8:
   st.plotly_chart(tension_plot_plus[7], use_container_width=True, key= "tension_t+")
with tab9:
   st.plotly_chart(tension_plot_plus[8], use_container_width=True, key= "tension_c+")
with tab10:
   st.plotly_chart(tension_plot_plus[9], use_container_width=True, key= "tension_v+")
with tab11:
   st.plotly_chart(tension_plot_plus[10], use_container_width=True, key= "tension_tot+")

df_tension = pd.DataFrame.from_dict(tension_table_print, orient = "index").T #.reset_index()
st.write(df_tension)


## PLOT TENSION NEGATIVE
st.title("Tensioni M-")
tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20, tab21, tab22 = st.tabs(test_keys)

tension_plot_neg, list_tension_neg = tension(dictProp, Sollecitazioni, condition="negative")

# using naive method
# to convert lists to dictionary
tension_table_print_negative = {}
for i, key in enumerate(test_keys):
   tension_table_print_negative[key] = list_tension_neg[i]

with tab12:
   st.plotly_chart(tension_plot_neg[0], use_container_width=True, key= "tension_g1-")
with tab13:
   st.plotly_chart(tension_plot_neg[1], use_container_width=True, key= "tension_g2-")
with tab14:
   st.plotly_chart(tension_plot_neg[2], use_container_width=True, key= "tension_r-")
with tab15:
   st.plotly_chart(tension_plot_neg[3], use_container_width=True, key= "tension_Mf-")
with tab16:
   st.plotly_chart(tension_plot_neg[4], use_container_width=True, key= "tension_Mts-")
with tab17:
   st.plotly_chart(tension_plot_neg[5], use_container_width=True, key= "tension_Mudl-")
with tab18:
   st.plotly_chart(tension_plot_neg[6], use_container_width=True, key= "tension_Mfolla-")
with tab19:
   st.plotly_chart(tension_plot_neg[7], use_container_width=True, key= "tension_t-")
with tab20:
   st.plotly_chart(tension_plot_neg[8], use_container_width=True, key= "tension_c-")
with tab21:
   st.plotly_chart(tension_plot_neg[9], use_container_width=True, key= "tension_v-")
with tab22:
   st.plotly_chart(tension_plot_neg[10], use_container_width=True, key= "tension_tot-")

df_tension_neg = pd.DataFrame.from_dict(tension_table_print_negative, orient = "index").T #.reset_index()
st.write(df_tension_neg)

## COEFFICIENTI DI SICUREZZA SULLE AZIONI
coeff = pd.read_excel('coefficienti.xlsx', index_col=0) 
st.write(coeff)
#ClasseAnima(d, t, fyk, yn, sigma1, sigma2)

# Editable table using st.data_editor
# Elenco delle verifiche in Markdown
# Descrizione con Markdown e LaTeX


st.markdown(r"""
## Verifiche strutturali

   1. **Verifica di resistenza:**  
   La tensione è calcolata come segue:

   $$
   \sigma(y) = \frac{N}{A} + \frac{M\cdot y}{I}
   $$

   Dove:
   - $$ \sigma(y)$$: Tensione normale nel punto.
   - N: Forza normale (assiale).
   - A: Area della sezione trasversale.
   - M: Momento flettenti 
   - I: Momento di inerzia
   - y: Coordinate del punto rispetto al baricentro della sezione.


   2. **Verifica di stabilità:**
      - Instabilità flesso-torsionale (LTB).
      - Instabilità locale (instabilità delle anime o delle piattabande).

   3. **Verifica a fatica:**
      - Analisi del numero di cicli e tensioni alternate.
      - Verifica della categoria di dettaglio (EN 1993-1-9).

   4. **Verifica degli stati limite di esercizio (SLE):**
      - Limitazione delle deformazioni.
      - Controllo delle vibrazioni.
      - Limitazione delle tensioni in esercizio.

   5. **Verifica delle connessioni:**.
      - verifica delle saldature.
      - verifica pioli.


## Normativa di riferimento
- EN 1993-1-1: Progettazione delle strutture in acciaio - Regole generali.
- EN 1993-1-8: Progettazione delle connessioni in acciaio.
- EN 1993-1-9: Verifica a fatica delle strutture in acciaio.
""")

