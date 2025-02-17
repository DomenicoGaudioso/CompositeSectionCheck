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



def Sx_slab(bcls, hcls,  dictProp, condition = "positive", n0 = 1, ninf = 1, nr = 1, nc = 1): 

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


def tension(dictProp, Sollecitazioni, hi, condition = "positive", n0 = 1, ninf = 1, nr = 1, nc = 1):
   #st.write("CIAO")

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
   g1_sigmaMf = (Mf*1000**2/dictProp["g1"]["Iy"])*hg #contributo per momento flettente
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
   
   Acls_r = (dictProp["r"]["A"] - dictProp["fe"]["A"])
   sigma_r = -N*1000/(Acls_r*nr)

   r_sigmaN = N*1000/dictProp["r"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["r"]["Pg"][1]
   r_sigmaMf = (Mf*1000**2/dictProp["r"]["Iy"])*hg #contributo per momento flettente
   r_sigma = r_sigmaN + r_sigmaMf
   r_sigma[0], r_sigma[1] =  r_sigma[0]/nr + sigma_r, r_sigma[1]/nr + sigma_r
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

   Acls_t = (dictProp["g2"]["A"] - dictProp["fe"]["A"])
   sigma_t = -N*1000/(Acls_t*ninf)

   t_sigmaN = N*1000/dictProp["g2"]["A"] #contributo per sola forza normale
   hg = np.array(hi)+dictProp["g2"]["Pg"][1]
   t_sigmaMf = (Mf*1000**2/dictProp["g2"]["Iy"])*hg #contributo per momento flettente
   t_sigma = t_sigmaN + t_sigmaMf
   t_sigma[0], t_sigma[1] =  t_sigma[0]/ninf + sigma_t, t_sigma[1]/ninf + sigma_t
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
   c_sigma[0], c_sigma[1] =  c_sigma[0]/nc , c_sigma[1]/nc
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
   hi_plot = hi + [hi[-1], 0, 0]

   #st.write(hi_plot)
   #st.write(list_sigma[1] )
   
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

def Sx_plate(listPlate, clsSection, dictProp, condition = "positive", bar = [1, 1], n = [1, 1, 1, 1]): 

   #Fase 1 - Only steel
   SectionComposite_g1 = builtSection(listPlate)
   #Fase 2 - G2 - t inf
   SectionComposite_g2 = CompositeSection(SectionComposite_g1, clsSection, bar, n[1])
   #Fase 3 - R - ritiro
   SectionComposite_r = CompositeSection(SectionComposite_g1, clsSection, bar, n[2])
   #Fase 4 - C - cedimenti
   SectionComposite_c = CompositeSection(SectionComposite_g1, clsSection, bar, n[3])
   #Fase 5 - Qm - mobili
   SectionComposite_m = CompositeSection(SectionComposite_g1, clsSection, bar, n[0])
   #Fase 6 - Fessurato
   clsSection_F = RectangularSection(0.1, 0.1, [0, 0], material="C25/30")
   SectionComposite_f = CompositeSection(SectionComposite_g1, clsSection_F, bar, 100000)

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


def checkTension(sigma, fyd):
   # Creiamo un dizionario con i dati delle verifiche

   dc1 = max(abs(sigma[2]), abs(sigma[3]))/338
   dc2 = max(abs(sigma[3]), abs(sigma[4]))/338
   dc3 = max(abs(sigma[4]), abs(sigma[5]))/338
   dc4 = max(abs(sigma[5]), abs(sigma[6]))/338
   dc5 = max(abs(sigma[6]), abs(sigma[7]))/338

   data = {
      "Verifica": [
         "piattabanda superiore",
         "raddoppio superiore",
         "anima",
         "raddoppio inferiore",
         "piattabanda inferiore",
      ],
      "sigma [MPa]": [max(abs(sigma[2]), abs(sigma[3])),
               max(abs(sigma[3]), abs(sigma[4])),
               max(abs(sigma[4]), abs(sigma[5])),
               max(abs(sigma[5]), abs(sigma[6])),
               max(abs(sigma[6]), abs(sigma[7])),
      ],
      "fyd [MPa]": [fyd,
               fyd,
               fyd,
               fyd,
               fyd,
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

   return df_ver1_slu

def checkTaglio_Instabilita(d, tw, fy, a = None):
   epsilon = np.sqrt(235/fy)
   eta = 1.20

   if a == None:
      k_tau = 5.34
   elif a/d < 1 and type(a) == "float":
      k_tau = 4 + 5.34/(a/d)**2
   elif a/d >= 1 and type(a) == "float":
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


def webBreathing(l, h, t, sigma, shear, a = None, typeBridge = "road bridges"):

   if typeBridge == "road bridges":
      limit = min(30 + 4.0*l, 300)

   elif typeBridge == "railways bridges":
      limit = min(55 + 3.3*l, 250)
   
   rapporto = h/t
   if rapporto <= limit and l>20:
      st.text("la verifica del respiro dell'anima non è necessaria")
   else:
      st.text("è necessaria la verifica del respiro dell'anima ")

   sigma_e = 190000*(t/h)**2


   sigma1 = sigma[0]
   sigma2 = sigma[1]
   psi = round(sigma1/sigma2,3)
   #print(sigma1, sigma2, "psi", psi)
   # CALCOLO COEFFICIENTE DI IMBOZZAMENTO
   ksigma = 4.0 if  psi == 1.00 else 8.2/(1.05+psi) if 1> psi >0 else 7.81 if psi == 0 else 7.81-6.29*psi + 9.78*psi**2 if 0> psi >-1 else 23.9 if psi == -1 else 5.98*(1-psi)**2 if -1> psi >-3 else print("WARNING: risulta fuori dalle condizioni impostate") 

   if a == None:
      k_tau = 5.34
   elif a/h < 1 :
      k_tau = 4 + 5.34/(a/h)**2
   elif a/h >= 1 :
      k_tau = 5.34 + 4/(a/h)**2
   

   tau_ed = shear/(t*h)*2 # ho messo il due per approssimare che in mezzeria la tensione è maggiore
   sigma_max= max(abs(sigma1), abs(sigma2))
   sigma_ec3 = np.sqrt((sigma_max/(ksigma*sigma_e))**2 + (1.1*tau_ed/(k_tau*sigma_e))**2)

   data1 = {
      "sigma1": [sigma1],
      "sigma2": [sigma2],
      "psi": [psi],
      "ksigma": [ksigma],
      "ktau": [k_tau],
      "Ved [KN]": [shear],
      "sigma_max [MPa]": [sigma_max],
      "tau_max [MPa]": [tau_ed],
      "sigma_ec [MPa]": [sigma_ec3],
      "Esito": ["✅" if sigma_ec3 <= 1.1 else "❌"]
   }

   # Creiamo un DataFrame con i dati
   df1= pd.DataFrame(data1)
   st.table(df1)

   return data1