import numpy as np
from cmath import pi, sqrt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
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

def combinazione(Sollecitazioni):
   
   path = "coefficienti.xlsx"
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

                  "D_bar": [16], 
                  "int_bar": [100]}

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



## COSTRUZIONE SOLETTA IN CALCESTRUZZO
clsSection = RectangularSection(Bcls, Hcls, [0, 0], material="C25/30")
p = 100 # passo bar
pointG0 = [[-p*i, 35] for i in range(0, int(Bcls*0.5/100))] + [[p*i, 35] for i in range(1, int(Bcls*0.5/100))]
pointG1 = [[-p*i, Hcls-35] for i in range(0, int(Bcls*0.5/100))] + [[p*i, Hcls-35] for i in range(1, int(Bcls*0.5/100))]
b0 = renforcementBar(12, pointG0)
b1 = renforcementBar(12, pointG1)
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
n = 6


#print(Isection)

SectionComposite_I = CompositeSection(Isection, clsSection, [b0, b1], n)
cplot = plotSection_ploty(SectionComposite_I)
#plot section in STREAMLIT
st.plotly_chart(cplot , use_container_width=True)
#cplot.show()

ninf = 15
n0 = 6
nr = 18
nc = 17

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
st.write(df_section_prop)


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


def tension(dictProp, Sollecitazioni):

   hi_plot = list(hi) + [hi[-1], hi[0], hi[0]]

   ## G1+
   N = Sollecitazioni["G1+"]["N"]
   Mf = Sollecitazioni["G1+"]["Mf"]

   g1_sigmaN = N*1000/dictProp["g1"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["g1"]["Pg"][1]
   g1_sigmaMf = (Mf*1000/dictProp["g1"]["Iy"])*hg #contributo per momento flettente
   g1_sigma = g1_sigmaN + g1_sigmaMf
   g1_sigma[0], g1_sigma[1] = 0, 0
   g1_sigma_plot = list(g1_sigma) + [0.0, 0.0, g1_sigma[0]]
   
   ## G2+
   N = Sollecitazioni["G2+"]["N"]
   Mf = Sollecitazioni["G2+"]["Mf"]

   g2_sigmaN = N*1000/dictProp["g2"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["g2"]["Pg"][1]
   g2_sigmaMf = (Mf*1000**2/dictProp["g2"]["Iy"])*hg #contributo per momento flettente
   g2_sigma = g2_sigmaN + g2_sigmaMf
   g2_sigma[0], g2_sigma[1] =  g2_sigma[0]/ninf, g2_sigma[1]/ninf
   g2_sigma_plot = list(g2_sigma) + [0.0, 0.0, g2_sigma[0]]

   ## MQ+
   N = Sollecitazioni["MQ+"]["N"]
   Mf = Sollecitazioni["MQ+"]["Mf"]

   ts_sigmaN = N*1000/dictProp["mobili"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["mobili"]["Pg"][1]
   ts_sigmaMf = (Mf*1000**2/dictProp["mobili"]["Iy"])*hg #contributo per momento flettente
   ts_sigma = ts_sigmaN + ts_sigmaMf
   ts_sigma[0], ts_sigma[1] =  ts_sigma[0]/n0, ts_sigma[1]/n0
   ts_sigma_plot = list(ts_sigma) + [0.0, 0.0, ts_sigma[0]]

   ## Md+
   N = Sollecitazioni["Md+"]["N"]
   Mf = Sollecitazioni["Md+"]["Mf"]

   udl_sigmaN = N*1000/dictProp["mobili"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["mobili"]["Pg"][1]
   udl_sigmaMf = (Mf*1000**2/dictProp["mobili"]["Iy"])*hg #contributo per momento flettente
   udl_sigma = udl_sigmaN + udl_sigmaMf
   udl_sigma[0], udl_sigma[1] =  udl_sigma[0]/n0, udl_sigma[1]/n0
   udl_sigma_plot = list(udl_sigma) + [0.0, 0.0, udl_sigma[0]]

   list_sigma = [g1_sigma_plot,
                 g2_sigma_plot,
                 ts_sigma_plot,
                 udl_sigma_plot]

   sigma_tot_plot = np.sum(list_sigma, axis=0)
   

   
   fig = go.Figure()

   #fig.add_trace(go.Scatter(x=g1_sigma_plot, y=hi_plot, fill='tozeroy', name = "tensione g1")) 
   #fig.add_trace(go.Scatter(x=g2_sigma_plot, y=hi_plot, fill='tozeroy', name = "tensione g2")) 
   fig.add_trace(go.Scatter(x=sigma_tot_plot, y=hi_plot, fill='tozeroy', name = "tensione totale")) 

   fig.update_layout(
      title=dict(
         text="Tensione totale sulla sezione"
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


   return fig, list_sigma


## PLOT TENSION
tension_plot, list_tension = tension(dictProp, Sollecitazioni)
st.plotly_chart(tension_plot , 
                use_container_width=True)


#ClasseAnima(d, t, fyk, yn, sigma1, sigma2)


#RIFARE QUESTA FUNZIONE
#---------------------------------------------#
def ConcioPropCalculate(dictConci, n = [18.34, 17.53, 6.45]): #Calcolo proprietà lorde della sezione senza considerare Beff
   
   dictProp = {}
   for i in dictConci:
      #costant Prop
      tw = dictConci[i]['SectionDefine']['tickness']['Web']
      H = dictConci[i]['SectionDefine']['variabile']["H"][0]
      tc = dictConci[i]["slab"]["hc"]
      phi = dictConci[i]["slab"]["D_bar"]
      p = dictConci[i]["slab"]["int_bar"] # passo bar
      
      bsup_ext = dictConci[i]["IntSup_ext"]
      bsup_int = dictConci[i]["IntSup_int"]
      binf_ext = dictConci[i]["IntInf_ext"]
      binf_int = dictConci[i]["IntInf_int"]
      
      ## SECTION PROP CALCULATE
      dictProp[i] = {'SectionProp':{"fase 0": None, "fase 1": None, "fase 2": None, "fase 3": None, "fase 4": None }}
      
      for bi_sup_e, bi_sup_i, bi_inf_e, bi_inf_i in zip(bsup_ext, bsup_int, binf_ext, binf_int):
         beffSup = bi_sup_e + bi_sup_i
         beffInf = bi_inf_e + bi_inf_i
         
         #calcolo t equivalente
         Asup = dictConci[i]['SectionDefine']['tickness']['Psup_ext']*bi_sup_e + dictConci[i]['SectionDefine']['tickness']['Psup_int']*bi_sup_i
         tpsup = Asup/(beffSup)
         Asup = dictConci[i]['SectionDefine']['tickness']['Pinf_ext']*bi_inf_e + dictConci[i]['SectionDefine']['tickness']['Pinf_int']*bi_inf_i
         tpinf = Asup/(beffInf)
         
         #assemble section
         clsSection = RectangularSection(beffSup, tc, [0, 0])
         n_bar = int(np.floor(beffSup/p)) #numero di bar
         pointG0 = [[p*i - (n_bar-1)*p*0.5, tc/2] for i in range(0, n_bar)] 
         b0 = renforcementBar(phi, pointG0 )
         
         orPlateSup1 = OrizontalPlate(beffSup, tpsup, [0, tc])
         vribs1 = V_rib(283, 300, 25, 6, [0, tc+tpsup]) #cl4Dict={"Binst":75, "Be1":60}
         wPlate1 = WebPlate(H-tpsup-tpinf, tw, [0, tc+tpsup], 0, material=None, cl4Dict=None)
         WebRibs = L__rib(120, 80, 10, [tw/2, 860-120], angle = -90) 
         pInfRibs = L__rib(120, 80, 10, [-10/2, H+tc-tpinf-120]) 
         orPlateInf1 = OrizontalPlate(beffInf, tpinf, [0, (tc+H-tpinf)])
         ## V ribs
         p_ribs1 = 600 # passo ribs
         n_ribs1 = int(np.floor(beffSup/p_ribs1)) #numero di ribs
         intRibs1 = [p_ribs1*i - (n_ribs1-1)*p_ribs1*0.5 for i in range(0, n_ribs1)] 
         ## Web ribs
         p_WebRibs = 600 # passo ribs
         n_WebRibs = 2 #numero di ribs
         int_WebRibs = [0] #[0, -700] 
         ## Plate inf ribs
         p_PinfRibs = 900 # passo ribs
         n_PinfRibs = int(np.floor(beffInf/p_PinfRibs)) #numero di ribs
         int_PinfRibs = [p_PinfRibs*i - (n_PinfRibs-1)*p_PinfRibs*0.5 for i in range(0, n_PinfRibs)]
         SectionnNassirya = OrthotropicSection1(orPlateSup1, vribs1,  intRibs1,  wPlate1,  WebRibs,  int_WebRibs, orPlateInf1, pInfRibs,  int_PinfRibs)
         #plotSection(SectionnNassirya, center=True, propTable=True)

         """
         #Fase 0 - Construction - partial section
         sectionProp = SectionnNassirya
         dictProp[i]['SectionProp']["fase 0"] = sectionProp
         
         #Fase 1 - G1 - Only steel
         sectionProp1 = SectionnNassirya
         dictProp[i]['SectionProp']["fase 1"] = sectionProp1
         
         #Fase 2 - G2 - t inf
         sectionProp2 = CompositeSection(SectionnNassirya, clsSection, [b0], n[0])
         dictProp[i]['SectionProp']["fase 2"] = sectionProp2
         
         #Fase 3 - R - ritiro
         sectionProp3 = CompositeSection(SectionnNassirya, clsSection, [b0], n[1])
         dictProp[i]['SectionProp']["fase 3"] = sectionProp3
         
         #Fase 4 - C - cedimenti
         sectionProp3 = CompositeSection(SectionnNassirya, clsSection, [b0], n[1])
         dictProp[i]['SectionProp']["fase 3"] = sectionProp3
         
         #Fase 5 - Qm - mobili
         sectionProp4 = CompositeSection(SectionnNassirya, clsSection, [b0], n[2])
         dictProp[i]['SectionProp']["fase 4"] = sectionProp4
         
         #Fase 6 - Fessurato
         sectionProp4 = CompositeSection(SectionnNassirya, clsSection, [b0], n[2])
         dictProp[i]['SectionProp']["fase 4"] = sectionProp4
         """
         
   return dictProp


