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

def resistenza_saldatura_EC(materiale, a, gamma_m2=1.25):
    
   #  Calcola la resistenza della saldatura per unità di lunghezza.

   #  Parametri:
   #  - materiale (str): Qualità dell'acciaio (deve essere presente nel dizionario).
   #  - fk (float): Resistenza caratteristica del metallo d'apporto (MPa).
   #  - a (float): Spessore del cordone di saldatura (mm).
   #  - gamma_m2 (float): Fattore di sicurezza per la saldatura (default: 1.25).

   #  Restituisce:
   #  - F_w_Rd (float): Resistenza della saldatura per unità di lunghezza (N/mm).
    
   acciai = {
      "S235": {"fy": 235, "fu": 360, "beta_w": 0.80},
      "S275": {"fy": 275, "fu": 430, "beta_w": 0.85},
      "S275 N/NL": {"fy": 275, "fu": 390, "beta_w": 0.85},
      "S275 M/ML": {"fy": 275, "fu": 370, "beta_w": 0.85},
      "S355": {"fy": 355, "fu": 510, "beta_w": 0.90},
      "S355 N/NL": {"fy": 355, "fu": 490, "beta_w": 0.90},
      "S355 M/ML": {"fy": 355, "fu": 470, "beta_w": 0.90},
      "S420 N/NL": {"fy": 420, "fu": 520, "beta_w": 1.00},
      "S420 M/ML": {"fy": 420, "fu": 520, "beta_w": 1.00},
      "S450": {"fy": 450, "fu": 550, "beta_w": 1.00},
      "S460 N/NL": {"fy": 460, "fu": 540, "beta_w": 1.00},
      "S460 M/ML": {"fy": 460, "fu": 540, "beta_w": 1.00},
      "S460 Q/QL/QL1": {"fy": 460, "fu": 570, "beta_w": 1.00},
   }

   if materiale not in acciai:    
      raise ValueError("Materiale non riconosciuto. Scegli un materiale valido")
    
   beta_w = acciai[materiale]["beta_w"]
   f_yk = acciai[materiale]["fy"]
   f_tk = acciai[materiale]["fu"]
   
   # Formula per la resistenza della saldatura
   F_w_Rd = (f_tk * a) / (np.sqrt(3) * beta_w * gamma_m2)  # sqrt(3) ≈ 1.732
   
   return F_w_Rd

def Resistenza_Piolo(d, ft, fck, Ec, hsc, gamma_v= 1.25):
   """
   Calcola la resistenza ultima di calcolo a taglio (SLU) di un piolo.

   Parametri:
   - d (float): Diametro del gambo del piolo (16 ≤ d ≤ 25 mm).
   - ft (float): Resistenza a rottura dell'acciaio del piolo (≤ 500 MPa).
   - fck (float): Resistenza caratteristica del calcestruzzo (MPa).
   - Ec (float): Modulo elastico del calcestruzzo (MPa).
   - hsc (float): Altezza del piolo dopo la saldatura.
   - gamma_v (float): Fattore parziale di sicurezza del materiale (default: 1.25).

   Restituisce:
   - P_Rd (float): Resistenza ultima di calcolo a taglio (SLU).
   """
   if not (16 <= d <= 25):
      raise ValueError("Il diametro d deve essere compreso tra 16 e 25 mm.")
   if ft > 500:
      raise ValueError("La resistenza a rottura dell'acciaio ft non deve superare 500 MPa.")
   
   # Calcolo del coefficiente α
   if 3 <= hsc / d <= 4:
      alpha = 0.2 * (hsc / d + 1)
   elif hsc / d > 4:
      alpha = 1.0
   else:
      raise ValueError("Il rapporto hsc/d deve essere ≥ 3.")
   
   # Calcolo dei due valori di resistenza
   P_Rd_a = (0.8 * ft * (pi * d**2 / 4)) / (gamma_v*1000)
   P_Rd_c = (0.29 * alpha * d**2 * sqrt(fck * Ec)) / (gamma_v*1000)

   st.markdown(r""" 
   Resistenza a taglio del gambo del bullone: 
   $$
   {P_{Rd,a} = \frac{0.8 \cdot f_t \cdot \left( \pi \cdot d^2 / 4 \right)}{\gamma_v}}
   $$
   """)

   st.markdown(r"""
   Resistenza a schiacciamento del calcestruzzo : 
   $$
   {P_{Rd,c} = \frac{0.29 \cdot \alpha \cdot d^2 \cdot \sqrt{f_{ck} \cdot E_c}}{\gamma_v}}
   $$
   """)

   st.markdown(r"""
   la resistenza dei connettori è determinata come il più piccolo dei valori sopra riportati : 
   $$
   {\min(P_{Rd,a}; P_{Rd,c})}
   $$
   """)

    
   # Restituisce il valore minimo
   return (P_Rd_a, P_Rd_c)

def combinazione(list_sigma, category = "A1_sfav"):
   
   path = "coefficienti.xlsx"

   gamma = pd.read_excel(path, "gamma", index_col=0)
   psi = pd.read_excel(path, "psi", index_col=0)

   gamma_g1 = gamma.loc['g1', category]
   gamma_g2 = gamma.loc['g2', category]
   gamma_r = gamma.loc['ritiro', category]
   gamma_c = gamma.loc['cedimenti', category]
   gamma_t = gamma.loc['temperatura', category]
   gamma_m = gamma.loc['mobili', category]
   gamma_v = gamma.loc['vento', category]

   psi_ts = psi.loc['tandem', "psi0"]

   sigma_slu = (np.array(list_sigma[0])*gamma_g1 #G1
   + np.array(list_sigma[1])*gamma_g2 #G2
   + np.array(list_sigma[2])*gamma_r #Ritiro

   + np.array(list_sigma[4])*gamma_m #Carichi mobili concentrati
   + np.array(list_sigma[5])*gamma_m #Carichi mobili distribuiti
   + np.array(list_sigma[6])*gamma_m #Carico folla
   + np.array(list_sigma[7])*gamma_t #temperatura
   + np.array(list_sigma[8])*gamma_c #cedimenti
   + np.array(list_sigma[9])*gamma_v)

   sigma_rara = (np.array(list_sigma[0])*1 #G1
   + np.array(list_sigma[1])*1 #G2
   + np.array(list_sigma[2])*1 #Ritiro

   + np.array(list_sigma[4])*psi.loc['tandem', "psi0"]#Carichi mobili concentrati
   + np.array(list_sigma[5])*psi.loc['distribuiti', "psi0"] #Carichi mobili distribuiti
   + np.array(list_sigma[6])*0 #Carico folla
   + np.array(list_sigma[7])*psi.loc['temperatura', "psi0"] #temperatura
   + np.array(list_sigma[8])*0 #cedimenti
   + np.array(list_sigma[9])*psi.loc['ventoE', "psi0"])

   sigma_frequente = (np.array(list_sigma[0])*1 #G1
   + np.array(list_sigma[1])*1 #G2
   + np.array(list_sigma[2])*1 #Ritiro

   + np.array(list_sigma[4])*psi.loc['tandem', "psi1"]#Carichi mobili concentrati
   + np.array(list_sigma[5])*psi.loc['distribuiti', "psi1"] #Carichi mobili distribuiti
   + np.array(list_sigma[6])*psi.loc['folla', "psi1"] #Carico folla
   + np.array(list_sigma[7])*psi.loc['temperatura', "psi1"] #temperatura
   + np.array(list_sigma[8])*0 #cedimenti
   + np.array(list_sigma[9])*psi.loc['ventoPS', "psi1"])

   sigma_qp = (np.array(list_sigma[0])*1 #G1
   + np.array(list_sigma[1])*1 #G2
   + np.array(list_sigma[2])*1 #Ritiro

   + np.array(list_sigma[4])*psi.loc['tandem', "psi2"]#Carichi mobili concentrati
   + np.array(list_sigma[5])*psi.loc['distribuiti', "psi2"] #Carichi mobili distribuiti
   + np.array(list_sigma[6])*psi.loc['folla', "psi2"] #Carico folla
   + np.array(list_sigma[7])*psi.loc['temperatura', "psi2"] #temperatura
   + np.array(list_sigma[8])*0 #cedimenti
   + np.array(list_sigma[9])*psi.loc['ventoPS', "psi2"])

   return sigma_slu, sigma_rara, sigma_frequente, sigma_qp


## START --> Sollecitazioni
# Convert dictionary to DataFrame
df = pd.DataFrame(Sollecitazioni).T.reset_index()
df.rename(columns={'index': 'Tipo'}, inplace=True)

# Editable table using st.data_editor
st.title("Edit Sollecitazioni")
st.write("Modify the values in the table below:")

# Display the editable data table
edited_df_soll = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",  # Allow adding/removing rows
)

# Save changes and convert back to dictionary
updated_dict_soll = edited_df_soll.set_index('Tipo').T.to_dict()
st.json(updated_dict_soll, expanded=False)

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
edited_df_sec = st.data_editor(
    df_section,
    use_container_width=True,
    num_rows="dynamic",  # Allow adding/removing rows
)

# convert back to dictionary
updated_dict_section = edited_df_sec.to_dict()
st.json(updated_dict_section , expanded=False)



##INPUT
# Extract updated values from the table
Hcls = float(edited_df_sec.loc["hcls"][0])
Bcls = float(edited_df_sec.loc['Bcls'][0])
hpredall = float(edited_df_sec.loc['h_predalle'][0])
phi_sup = float(edited_df_sec.loc['phi_sup'][0])
int_sup = float(edited_df_sec.loc['int_sup'][0])
phi_inf = float(edited_df_sec.loc['phi_inf'][0])
int_inf = float(edited_df_sec.loc['int_inf'][0])

bf = float(edited_df_sec.loc['bPsup'][0])
tbf = float(edited_df_sec.loc['tPsup'][0])

bf_r = float(edited_df_sec.loc['brPsup'][0])
tbrf = float(edited_df_sec.loc['trPsup'][0])

hw = float(edited_df_sec.loc['ha'][0])
tw = float(edited_df_sec.loc['ta'][0])

rbf_inf = float(edited_df_sec.loc['brPinf'][0])
rtbf_inf = float(edited_df_sec.loc['trPinf'][0])

binf = float(edited_df_sec.loc['bPinf'][0])
tbf_inf = float(edited_df_sec.loc['tPinf'][0])

ninf = float(edited_df_sec.loc['n_inf'][0])
n0 = float(edited_df_sec.loc['n_0'][0])
nr = float(edited_df_sec.loc['n_r'][0])
nc = float(edited_df_sec.loc['n_c'][0])



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
#PARTIRE DA QUI PER CREARE IL MOMENTO STATICO PER IL CALCOLO DELLA FORZA NELLE SALDATURE

def Sx_plate(listPlate, clsSection, dictProp, condition = "positive"): 

   #Fase 1 - Only steel
   SectionComposite_g1 = builtSection(listPlate)
   #Fase 2 - G2 - t inf
   SectionComposite_g2 = CompositeSection(SectionComposite_g1, clsSection, [b0, b1], ninf)
   #Fase 3 - R - ritiro
   SectionComposite_r = CompositeSection(SectionComposite_g1, clsSection, [b0, b1], nr)
   #Fase 4 - C - cedimenti
   SectionComposite_c = CompositeSection(SectionComposite_g1, clsSection, [b0, b1], nc)
   #Fase 5 - Qm - mobili
   SectionComposite_m = CompositeSection(SectionComposite_g1, clsSection, [b0, b1], n0)
   #Fase 6 - Fessurato
   clsSection_F = RectangularSection(0.1, 0.1, [0, 0], material="C25/30")
   SectionComposite_f = CompositeSection(SectionComposite_g1, clsSection_F, [b0, b1], 100000)

   #calcolo del momento statico della soletta rispetto il baricentro della sezione composta
   Sx_g1 = SectionComposite_g1["A"]*( dictProp["g1"]["Pg"][1] - SectionComposite_g1["Pg"][1])
   Sx_g2 = SectionComposite_g2["A"] *( dictProp["g2"]["Pg"][1] - SectionComposite_g2["Pg"][1])
   Sx_r = SectionComposite_r["A"] *( dictProp["r"]["Pg"][1] - SectionComposite_r["Pg"][1])
   Sx_fat = SectionComposite_m["A"] *( dictProp["mobili"]["Pg"][1] - SectionComposite_m["Pg"][1])
   Sx_ts = SectionComposite_m["A"] *( dictProp["mobili"]["Pg"][1] - SectionComposite_m["Pg"][1])
   Sx_udl = SectionComposite_m["A"] *( dictProp["mobili"]["Pg"][1] - SectionComposite_m["Pg"][1])
   Sx_folla = SectionComposite_m["A"] *( dictProp["mobili"]["Pg"][1] - SectionComposite_m["Pg"][1])
   Sx_t = SectionComposite_g2["A"] *( dictProp["g2"]["Pg"][1] - SectionComposite_g2["Pg"][1])
   Sx_c = SectionComposite_c["A"] *( dictProp["c"]["Pg"][1] - SectionComposite_c["Pg"][1])
   Sx_v = SectionComposite_g2["A"] *( dictProp["g2"]["Pg"][1] - SectionComposite_g2["Pg"][1])

   Sx_list = [Sx_g1, Sx_g2, Sx_r, Sx_fat, Sx_ts, Sx_udl, Sx_folla, Sx_t, Sx_c, Sx_v]

   # calcolo del braccio della forza interna
   z_g1 = dictProp["g1"]["Iy"]/(Sx_g1) # Acls *( dictProp["g1"]["Pg"][1]-yg_pl)
   z_g2 = dictProp["g2"]["Iy"]/(Sx_g2)
   z_r = dictProp["r"]["Iy"]/(Sx_r)
   z_fat = dictProp["mobili"]["Iy"]/(Sx_fat)
   z_ts = dictProp["mobili"]["Iy"]/(Sx_ts)
   z_udl = dictProp["mobili"]["Iy"]/(Sx_udl)
   z_folla = dictProp["mobili"]["Iy"]/(Sx_folla)
   z_t = dictProp["g2"]["Iy"]/(Sx_t)
   z_c = dictProp["c"]["Iy"]/(Sx_c)
   z_v = dictProp["g2"]["Iy"]/(Sx_v)

   z_list = [z_g1, z_g2, z_r, z_fat, z_ts, z_udl, z_folla, z_t, z_c, z_v]

   return Sx_list, z_list

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

hi_plot = list(hi) + [hi[-1], hi[0], hi[0]]

Acls = gapCls*Bcls

def Sx_slab(bcls, hcls,  dictProp, condition = "positive"): 

   Acls = bcls*hcls

   #calcolo del momento statico della soletta rispetto il baricentro della sezione composta
   Sx_g1 = 0.0 # Acls *( dictProp["g1"]["Pg"][1]-hcls/2)
   Sx_g2 = Acls *( dictProp["g2"]["Pg"][1]-hcls/2)
   Sx_r = Acls *( dictProp["r"]["Pg"][1]-hcls/2)
   Sx_fat = Acls *( dictProp["mobili"]["Pg"][1]-hcls/2)
   Sx_ts = Acls *( dictProp["mobili"]["Pg"][1]-hcls/2)
   Sx_udl = Acls *( dictProp["mobili"]["Pg"][1]-hcls/2)
   Sx_folla = Acls *( dictProp["mobili"]["Pg"][1]-hcls/2)
   Sx_t = Acls *( dictProp["g2"]["Pg"][1]-hcls/2)
   Sx_c = Acls *( dictProp["c"]["Pg"][1]-hcls/2)
   Sx_v = Acls *( dictProp["g2"]["Pg"][1]-hcls/2)

   Sx_list = [Sx_g1, Sx_g2, Sx_r, Sx_fat, Sx_ts, Sx_udl, Sx_folla, Sx_t, Sx_c, Sx_v]

   # calcolo del braccio della forza interna
   z_g1 = 0.0 # Acls *( dictProp["g1"]["Pg"][1]-hcls/2)
   z_g2 = dictProp["g2"]["Iy"]/(Sx_g2/ninf)
   z_r = dictProp["r"]["Iy"]/(Sx_r/nr)
   z_fat = dictProp["mobili"]["Iy"]/(Sx_fat/n0)
   z_ts = dictProp["mobili"]["Iy"]/(Sx_ts/n0)
   z_udl = dictProp["mobili"]["Iy"]/(Sx_udl/n0)
   z_folla = dictProp["mobili"]["Iy"]/(Sx_folla/n0)
   z_t = dictProp["g2"]["Iy"]/(Sx_t/ninf)
   z_c = dictProp["c"]["Iy"]/(Sx_c/nc)
   z_v = dictProp["g2"]["Iy"]/(Sx_v/ninf)

   z_list = [z_g1, z_g2, z_r, z_fat, z_ts, z_udl, z_folla, z_t, z_c, z_v]

   return Sx_list, z_list


def tension(dictProp, Sollecitazioni, hi_plot, condition = "positive"):

   posList = ["G1+", "G2+", 'R+', 'Mfat+', 'MQ+', 'Md+', 'Mf+','T+', 'C+', 'V+']
   negList = ["G1-", "G2-", 'R-', 'Mfat-', 'MQ-', 'Md-', 'Mf-','T-', 'C-', 'V-']

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

   t_sigmaN = N*1000/dictProp["g2"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["g2"]["Pg"][1]
   t_sigmaMf = (Mf*1000**2/dictProp["g2"]["Iy"])*hg #contributo per momento flettente
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

   v_sigmaN = N*1000/dictProp["g2"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["g2"]["Pg"][1]
   v_sigmaMf = (Mf*1000**2/dictProp["g2"]["Iy"])*hg #contributo per momento flettente
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

def tension_plot(x_list, y_list):
   fig = go.Figure()

   #fig.add_trace(go.Scatter(x=g1_sigma_plot, y=hi_plot, fill='tozeroy', name = "tensione g1")) 
   #fig.add_trace(go.Scatter(x=g2_sigma_plot, y=hi_plot, fill='tozeroy', name = "tensione g2")) 
   fig.add_trace(go.Scatter(x=x_list, y=y_list, fill='tozeroy')) 

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

   return fig



## PLOT TENSION POSITIVE
tension_plot_plus, list_tension = tension(dictProp, updated_dict_soll, hi_plot)

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

tension_plot_neg, list_tension_neg = tension(dictProp, updated_dict_soll, hi_plot, condition="negative")

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
gamma = pd.read_excel('coefficienti.xlsx', "gamma", index_col=0) 
psi = pd.read_excel('coefficienti.xlsx', "psi", index_col=0) 
st.write(gamma)
st.write(psi)
#ClasseAnima(d, t, fyk, yn, sigma1, sigma2)

## TENSIONI COMBINATE
tension_slu, tension_rara, tension_frequente, tension_qp = combinazione(list_tension)
tension_slu_neg, tension_rara, tension_frequente, tension_qp = combinazione(list_tension_neg)

tab23, tab24, tab25, tab26 = st.tabs(["slu", "rara", "frequente", "quasi permanente"])

slu_plot = tension_plot(tension_slu, hi_plot)
rara_plot = tension_plot(tension_rara, hi_plot)
freq_plot = tension_plot(tension_frequente, hi_plot)
qp_plot = tension_plot(tension_qp, hi_plot)

with tab23:
   st.plotly_chart(slu_plot, use_container_width=True, key= "tension_slu")
with tab24:
   st.plotly_chart(rara_plot, use_container_width=True, key= "tension_rara")
with tab25:
   st.plotly_chart(freq_plot, use_container_width=True, key= "tension_frequente")
with tab26:
   st.plotly_chart(qp_plot, use_container_width=True, key= "tension_qp")


# Editable table using st.data_editor
# Elenco delle verifiche in Markdown
# Descrizione con Markdown e LaTeX

st.markdown("""   
            #### Verifiche
            """)
st.markdown("""   
            ##### 1) Calcolo della classe 
            """)

st.markdown("""   
            ##### 2) Verifiche tensionali sugli elementi in acciaio (S.L.U.)
            """)
# Creiamo un dizionario con i dati delle verifiche

dc1 = max(abs(tension_slu[2]), abs(tension_slu[3]))/338
dc2 = max(abs(tension_slu[3]), abs(tension_slu[4]))/338
dc3 = max(abs(tension_slu[4]), abs(tension_slu[5]))/338
dc4 = max(abs(tension_slu[5]), abs(tension_slu[6]))/338
dc5 = max(abs(tension_slu[6]), abs(tension_slu[7]))/338

data = {
    "Verifica": [
        "piattabanda superiore",
        "raddoppio superiore",
        "anima",
        "raddoppio inferiore",
        "piattabanda inferiore",
    ],
    "sigma [MPa]": [max(abs(tension_slu[2]), abs(tension_slu[3])),
              max(abs(tension_slu[3]), abs(tension_slu[4])),
              max(abs(tension_slu[4]), abs(tension_slu[5])),
              max(abs(tension_slu[5]), abs(tension_slu[6])),
              max(abs(tension_slu[6]), abs(tension_slu[7])),
    ],
    "fyd [MPa]": [338,
              338,
              338,
              338,
              338,
    ],

    "D/C": [dc1,
            dc2,
            dc3,
            dc4,
            dc5,
    ],

    "Esito": [
        "✅" if dc1<= 1 else "❌",
        "✅" if dc2<= 1 else "❌",
        "✅" if dc3<= 1 else "❌",
        "✅" if dc4 <= 1 else "❌",
        "✅" if dc5 <= 1.0 else "❌",
    ]
}

# Creiamo un DataFrame con i dati
df_ver1_slu = pd.DataFrame(data)
# Mostriamo la tabella
st.table(df_ver1_slu)

st.markdown(r"""
   La tensione è stata calcolata come segue:
            
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
   
   Per quanto rigurada la resistenza è stata considerata quella del materiale ad associato al singolo componente in acciaio
""")


st.markdown("""   
            ##### 3) Verifica a taglio - instabilità dell'anima (S.L.U.)
            """)


def checkTaglio_Instabilita(d, tw, fy, a = None):
   epsilon = np.sqrt(235/fy)
   eta = 1.20

   if a == None:
      k_tau = 5.34
   elif a/d < 1:
      k_tau = 4 + 5.34/(a/d)**2
   elif a/d >= 1:
      k_tau = 5.34 + 4/(a/d)**2

   if a == None:
      lambda_w = (d)/(86.4*epsilon*tw)
   else:
      lambda_w = (d/tw)/(37.4*epsilon*np.sqrt(k_tau)) # snellezza adimensioanle dell'anima
   
   ## calcolo tab = chi*tau_rd 
   if lambda_w >= 1.08 and a != None: #Non rigid end post
      tau_ba = (0.83/lambda_w)*(fy/np.sqrt(3))

   elif lambda_w <= 1.08 and lambda_w >= 0.83/eta: #Non rigid end post end rigid end post
      tau_ba = (0.83/lambda_w)*(fy/np.sqrt(3))

   elif lambda_w < 0.83/eta: #Non rigid end post end rigid end post
      tau_ba = (eta)*(fy/np.sqrt(3))
   
   
   Vba_rd = (d*tw*tau_ba)/(1.15*1000)

   st.markdown(fr"""
   Fattore di imbozzamanto per taglio: 
   $$k_{{\tau}} = {k_tau:.2f} \, \text{{}}$$
   """)

   st.markdown(fr"""
   Snellezza adimensionale dell'anima: 
   $$\overline{{\lambda}}_{{w}} = {lambda_w:.2f} \, \text{{}}$$
   """)

   st.markdown(fr"""
   Resistenza post-critica a taglio: 
   $${{\tau}}_{{ba}} = {tau_ba:.2f} \, \text{{MPa}}$$
   """)

   st.markdown(fr"""
   Resistenza a taglio: 
   $$V_{{ba,Rd}} = {Vba_rd:.2f} \, \text{{KN}}$$
   """)


   return Vba_rd

def Sollecitazione_list(Sollecitazioni, condition = "positive", cds = "T"):

   posList = ["G1+", "G2+", 'R+', 'Mfat+', 'MQ+', 'Md+', 'Mf+','T+', 'C+', 'V+']
   negList = ["G1-", "G2-", 'R-', 'Mfat-', 'MQ-', 'Md-', 'Mf-','T-', 'C-', 'V-']

   ## G1+
   if condition == "positive":
      cds_g1 = Sollecitazioni[posList[0]][cds]
   else:
      cds_g1 = Sollecitazioni[negList[0]][cds]

   #st.write(g1_sigma_plot)
   
   ## G2+
   if condition == "positive":
      cds_g2 = Sollecitazioni[posList[1]][cds]
   else:
      cds_g2 = Sollecitazioni[negList[1]][cds]

   ## R+ (CONTROLLARE)
   if condition == "positive":
      cds_r = Sollecitazioni[posList[2]][cds]
   else:
      cds_r = Sollecitazioni[negList[2]][cds]

   ## Mfat+ Fatica (CONTROLLARE)
   if condition == "positive":
      cds_fat = Sollecitazioni[posList[3]][cds]
   else:
      cds_fat = Sollecitazioni[negList[3]][cds]

   ## MQ+ Mobili concentrati
   if condition == "positive":
      cds_ts = Sollecitazioni[posList[4]][cds]
   else:
      cds_ts = Sollecitazioni[negList[4]][cds]

   ## Md+ Mobili distribuiti
   if condition == "positive":
      cds_udl = Sollecitazioni[posList[5]][cds]
   else:
      cds_udl = Sollecitazioni[negList[5]][cds]

   ## Mf+ Mobili folla
   if condition == "positive":
      cds_folla = Sollecitazioni[posList[6]][cds]
   else:
      cds_folla = Sollecitazioni[negList[6]][cds]

   ## T+ termica(CONTROLLARE)
   if condition == "positive":
      cds_t = Sollecitazioni[posList[7]][cds]
   else:
      cds_t = Sollecitazioni[negList[7]][cds]

   ## C+ (CONTROLLARE)
   if condition == "positive":
      cds_c = Sollecitazioni[posList[8]][cds]
   else:
      cds_c = Sollecitazioni[negList[8]][cds]

   ## V+ vento(CONTROLLARE)
   if condition == "positive":
      cds_v = Sollecitazioni[posList[9]][cds]
   else:
      cds_v = Sollecitazioni[negList[9]][cds]

   cds_list = [cds_g1, cds_g2, cds_r, cds_fat, cds_ts, cds_udl, cds_folla, cds_t, cds_c, cds_v]

   return cds_list

#st.write(updated_dict_soll)
Ved_pos = Sollecitazione_list(updated_dict_soll, condition = "positive", cds= "T")
Ved_neg = Sollecitazione_list(updated_dict_soll, condition = "negative", cds= "T")

Ved_slu_pos = combinazione(Ved_pos, category = "A1_sfav")
Ved_slu_neg = combinazione(Ved_neg, category = "A1_sfav")

taglio_anima = checkTaglio_Instabilita(hw, tw, 355)


dc1 = Ved_slu_pos[0]/taglio_anima
dc2 = Ved_slu_neg[0]/taglio_anima

data = {
    "Verifica": [
        "Taglio positivo",
        "Taglio negativo",
    ],
    "Ved [KN]": [Ved_slu_pos[0],
            Ved_slu_neg[0],

    ],
    "Vrd [KN]": [taglio_anima,
            taglio_anima,
    ],

    "D/C": [dc1,
            dc2,
    ],

    "Esito": [
        "✅" if dc1<= 1 else "❌",
        "✅" if dc2<= 1 else "❌",
    ]
}

# Creiamo un DataFrame con i dati
df_ver2_slu = pd.DataFrame(data)
# Mostriamo la tabella
st.table(df_ver2_slu)

st.markdown("""   
            ##### 4) Verifica delle saldature di composizione
            """)

st.markdown(r"""
Il calcolo della forza agente sulle saldature è stato eseguito secondo la formula di Jourawski 
   $$
   V_{\tau} = \tau \cdot b = \frac{T}{b} \cdot \frac{S}{I} = \frac{T}{z}
   $$
   La resistenza della saldatura è calcolata con il metodo semplificato :         
   $$
   F_{w,Rd} = \frac{{f_k \cdot a}}{{\sqrt{{3}} \cdot \beta_w \cdot \gamma_{M2}}}
   $$        
""")

#momento statico per le saldature
# saldature piattabanda superiore con raddoppio
#st.write(listDict[0:1])
Stau_s1 = Sx_plate(listDict[0:1], clsSection, dictProp, condition = "positive")
V_s1_pos = np.array(Ved_pos)/(np.array(Stau_s1[1]))
V_s1_neg = np.array(Ved_neg)/(np.array(Stau_s1[1]))

Vs1_comb_pos = combinazione(list(V_s1_pos), category = "A1_sfav")
Vs1_comb_neg = combinazione(list(V_s1_neg), category = "A1_sfav")

#st.write(Stau_s1[1])
# saldature raddoppio piattabanda superiore con anima
Stau_s2 = Sx_plate(listDict[0:2], clsSection, dictProp, condition = "positive")
V_s2_pos = np.array(Ved_pos)/(np.array(Stau_s2[1]))
V_s2_neg = np.array(Ved_neg)/(np.array(Stau_s2[1]))

Vs2_comb_pos = combinazione(list(V_s2_pos), category = "A1_sfav")
Vs2_comb_neg = combinazione(list(V_s2_neg), category = "A1_sfav")
#st.write(Stau_s2[1])
# saldature raddoppio piattabanda inferiore con anima
Stau_s3 = Sx_plate(listDict[0:3], clsSection, dictProp, condition = "positive")
V_s3_pos = np.array(Ved_pos)/(np.array(Stau_s3[1]))
V_s3_neg = np.array(Ved_neg)/(np.array(Stau_s3[1]))

Vs3_comb_pos = combinazione(list(V_s3_pos), category = "A1_sfav")
Vs3_comb_neg = combinazione(list(V_s3_neg), category = "A1_sfav")
#st.write(Stau_s3[1])
# saldature piattabanda inferiore con anima
Stau_s4 = Sx_plate(listDict[0:4], clsSection, dictProp, condition = "positive")
V_s4_pos = np.array(Ved_pos)/(np.array(Stau_s4[1]))
V_s4_neg = np.array(Ved_neg)/(np.array(Stau_s4[1]))

Vs4_comb_pos = combinazione(list(V_s4_pos), category = "A1_sfav")
Vs4_comb_neg = combinazione(list(V_s4_neg), category = "A1_sfav")
#st.write(Stau_s4[1])

a = 6 # gola 6 mm
res_cordoni = resistenza_saldatura_EC("S235", a, gamma_m2=1.25)*2/1000

ds1_sald_pos = Vs1_comb_pos[0]/res_cordoni
ds1_sald_neg = Vs1_comb_neg[0]/res_cordoni
ds2_sald_pos = Vs2_comb_pos[0]/res_cordoni
ds2_sald_neg = Vs2_comb_neg[0]/res_cordoni
ds3_sald_pos = Vs3_comb_pos[0]/res_cordoni
ds3_sald_neg = Vs3_comb_neg[0]/res_cordoni
ds4_sald_pos = Vs4_comb_pos[0]/res_cordoni
ds4_sald_neg = Vs4_comb_neg[0]/res_cordoni


## VERIFICA ALLO SLU
data = {
    "Verifica": [
        "Taglio saldatura pittabanda sup. - raddoppio sup. (M positivo)",
        "Taglio saldatura pittabanda sup. - raddoppio sup. (M negativo)",
        "Taglio saldatura raddoppio sup. - anima (M positivo)",
        "Taglio saldatura raddoppio sup. - anima (M negativo)",
        "Taglio saldatura anima - raddoppio inf. (M positivo)",
        "Taglio saldatura anima - raddoppio inf. (M negativo)",
        "Taglio saldatura raddoppio inf. - piattabanda inf. (M positivo)",
        "Taglio saldatura raddoppio inf. - piattabanda inf. (M positivo)",
    ],
    "Ved [KN]": [Vs1_comb_pos[0],
            Vs1_comb_neg[0],
            Vs2_comb_pos[0],
            Vs2_comb_neg[0],
            Vs3_comb_pos[0],
            Vs3_comb_neg[0],
            Vs4_comb_pos[0],
            Vs4_comb_neg[0],
    ],
    "Vrd [KN]": [res_cordoni,
            res_cordoni,
            res_cordoni,
            res_cordoni,
            res_cordoni,
            res_cordoni,
            res_cordoni,
            res_cordoni,
    ],

    "D/C": [ds1_sald_pos,
            ds1_sald_neg,
            ds2_sald_pos,
            ds2_sald_neg,
            ds3_sald_pos,
            ds3_sald_neg,
            ds4_sald_pos,
            ds4_sald_neg,
    ],

    "Esito": ["✅" if ds1_sald_pos <= 1 else "❌",
        "✅" if ds1_sald_neg <= 1 else "❌",
        "✅" if ds2_sald_pos <= 1 else "❌",
        "✅" if ds2_sald_neg <= 1 else "❌",
        "✅" if ds3_sald_pos <= 1 else "❌",
        "✅" if ds3_sald_neg <= 1 else "❌",
        "✅" if ds4_sald_pos <= 1 else "❌",
        "✅" if ds4_sald_neg <= 1 else "❌",
    ]
}

# Creiamo un DataFrame con i dati
df_sald = pd.DataFrame(data)
# Mostriamo la tabella
st.table(df_sald)


st.markdown("""   
            ##### 5) Verifica dei pioli
            """)

#Med_pos = Sollecitazione_list(updated_dict_soll, condition = "positive", cds= "Mf")
#Med_neg = Sollecitazione_list(updated_dict_soll, condition = "negative", cds= "Mf")

st.markdown(r"""
Il calcolo elastico della forza di scorrimento è stato eseguito secondo la formula di Jourawski 
   $$
   V = T \cdot \frac{S}{I} = \frac{T}{z}
   $$
""")

Sx, zx = Sx_slab(Bcls, gapCls,  dictProp, condition = "positive")

V_pioli_pos = np.array(Ved_pos)/(np.array(zx))
V_pioli_neg = np.array(Ved_neg)/(np.array(zx))
V_pioli_pos[0]= 0.0
V_pioli_neg[0]= 0.0

Vpioli_comb_pos = combinazione(list(V_pioli_pos), category = "A1_sfav")
Vpioli_comb_neg = combinazione(list(V_pioli_neg), category = "A1_sfav")

# Esempio di utilizzo
d = 16  # mm
ft = 450  # MPa
fck = 30  # MPa
Ec = 30000  # MPa
hsc = 100  # mm

nfp = 1 #numero file di pioli
s_pioli = 200 #passo pioli in un metro
nPioli_tot = (1000/s_pioli)*nfp

resPiolo = np.min(Resistenza_Piolo(d, ft, fck, Ec, hsc)).real

st.markdown(fr"""
La resistenza del singolo piolo è pari a:  {resPiolo:.2f} kN
""")


resTotalePioli = resPiolo*nPioli_tot
resPioli_sle = 0.6*resTotalePioli

dc_pioli_pos_slu = np.abs(Vpioli_comb_pos[0]/resTotalePioli)
dc_pioli_neg_slu = np.abs(Vpioli_comb_neg[0]/resTotalePioli)
dc_pioli_pos_sle = np.abs(Vpioli_comb_pos[1]/(resPioli_sle))
dc_pioli_neg_sle = np.abs(Vpioli_comb_neg[1]/(resPioli_sle))

## VERIFICA ALLO SLU
data = {
    "Verifica": [
        "Taglio positivo (SLU)",
        "Taglio negativo (SLU)",
        "Taglio positivo (SLE)",
        "Taglio negativo (SLE)",
    ],
    "Ved [KN]": [Vpioli_comb_pos[0],
            Vpioli_comb_neg[0],
            Vpioli_comb_pos[1],
            Vpioli_comb_neg[1],
    ],
    "Vrd [KN]": [resTotalePioli,
            resTotalePioli,
            resPioli_sle,
            resPioli_sle
    ],

    "D/C": [dc_pioli_pos_slu ,
            dc_pioli_neg_slu ,
            dc_pioli_pos_sle ,
            dc_pioli_neg_sle ,
    ],

    "Esito": [
        "✅" if dc_pioli_pos_slu <= 1 else "❌",
        "✅" if dc_pioli_neg_slu <= 1 else "❌",
        "✅" if dc_pioli_pos_sle <= 1 else "❌",
        "✅" if dc_pioli_neg_sle <= 1 else "❌",
    ]
}

# Creiamo un DataFrame con i dati
df_pioli = pd.DataFrame(data)
# Mostriamo la tabella
st.table(df_pioli)

st.markdown("""   
            ##### 6) Verifiche dettagli a fatica
            """)

delta_sigma = (np.array(list_tension[3]) - np.array(list_tension_neg[3]))
delta_tau_cls = (V_pioli_pos[3] - V_pioli_neg[3])
delta_tau1 = (V_s1_pos[3] - V_s1_neg[3])
delta_tau2 = (V_s2_pos[3] - V_s2_neg[3])
delta_tau3 = (V_s3_pos[3] - V_s3_neg[3])
delta_tau4 = (V_s4_pos[3] - V_s4_neg[3])

## VERIFICA ALLO SLU
data = {
    "Punto": [
        "Estradosso soletta",
        "Intradosso soletta",
        "Estradosso piattabanda superiore",
        "Estradosso raddoppio piattabanda superiore",
        "Estradosso anima",
        "Istradosso anima",
        "Intradosso raddoppio piattabanda inferiore",
        "Intradosso piattabanda inferiore",
    ],
    "Delta [MPa]": delta_sigma[0:8],
    "scorrimento [KN]": [0.0, 
                         delta_tau_cls, 
                         delta_tau1, 
                         delta_tau2, 
                         delta_tau2, 
                         delta_tau3, 
                         delta_tau3, 
                         delta_tau4],
}

# Creiamo un DataFrame con i dati
df_pioli = pd.DataFrame(data)
# Mostriamo la tabella
st.table(df_pioli)