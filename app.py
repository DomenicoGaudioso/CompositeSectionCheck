import numpy as np
from cmath import pi, sqrt
import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from SGMPy_material import *
from SGMPy_section_v01 import *
from SGMPy_check import *
import plotly.graph_objs as go
from streamlit_option_menu import option_menu


st.set_page_config(page_icon="🚄", page_title=":rainbow[𝒻𝑒𝓂 DTO]  ", layout="wide")


with st.sidebar:
   st.title(' ⛏️ :rainbow[ISCT]') #💻🌈🖱️
   txtName2 = ":red[**I**]:gray[-]  :orange[**S**]:gray[ection] :green[**C**]:gray[omposite] :green[**T**]:gray[ool]"
               
   st.markdown(txtName2)
   imageName = ['Screenshot 2025-01-28 182905.png', "Screenshot 2025-01-28 182928.png"]
   #isertImage(imageName[0], width = 210)


   # Utilizza st.markdown per inserire i link
   st.markdown("## Contacts")
   st.write("Name: Domenico")
   st.write("Surname: Gaudioso")
   st.write("📧 domenicogaudioso@outlook.it")
   st.markdown("📱 [LinkedIn]({'https://www.linkedin.com/in/il_tuo_profilo_linkedin'})", unsafe_allow_html=True)
   st.markdown("💻 [GitHub]({'https://github.com/DomenicoGaudioso'})", unsafe_allow_html=True)

   st.markdown("## About")
   # Link di Streamlit
   st.markdown(f"[Streamlit]({'https://www.streamlit.io/'})", unsafe_allow_html=True)


txtName2 = " **⛏️ :rainbow[ISCT]**  -- :red[**I**]:gray[-]  :orange[**S**]:gray[ection] :green[**C**]:gray[omposite] :green[**T**]:gray[ool]"
            
st.markdown(txtName2)
selected3 = option_menu(None, ["Documentazione", "Sollecitazioni", "Sezione", "Tensioni", "Verifiche"],
                         
    icons=['book-half', 'cloud-arrow-up-fill', "magic", "triangle", "clipboard2-pulse-fill"], 
    menu_icon="cast", default_index=0, orientation="horizontal"
)

input_data = {}  


if selected3 == "Documentazione":
   st.title("Documentazione in WIP")



Sollecitazioni = {'G1+':{'N': -0.0, 'T': 65, 'Mf': 89, 'Mt': 0.0}, 'G1-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0}, # peso proprio
                  'G2+':{'N': 0.0, 'T': 74, 'Mf': 77, 'Mt': 0.0}, 'G2-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0}, # permanenti portati
                  'R+':{ 'N': 1314, 'T': 0.0, 'Mf': 175, 'Mt': 0.0}, 'R-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0},  # ritiro
                  'Mfat+':{ 'N': 0.0, 'T': 0, 'Mf': 0.0, 'Mt': 0.0}, 'Mfat-':{ 'N': 0.0, 'T': 0, 'Mf': 0.0, 'Mt': 0.0}, # fatica
                  'MQ+':{ 'N': 0.0, 'T': 286, 'Mf': 336, 'Mt': 0.0}, 'MQ-':{ 'N': 0.0, 'T': -248, 'Mf': -91.0, 'Mt': 0.0}, # mobili concentrati
                  'Md+':{'N': 0.0, 'T': 34, 'Mf': 48, 'Mt': 0.0}, 'Md-':{ 'N': 0.0, 'T': -37, 'Mf': -16, 'Mt': 0.0}, # mobili distribuiti
                  'Mf+':{ 'N': 0.0, 'T': 0.0, 'Mf': 0, 'Mt': 0.0}, 'Mf-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0}, # folla
                  'T+':{ 'N': 0.0, 'T':2, 'Mf': 53, 'Mt': 0.0}, 'T-':{ 'N': 0.0, 'T': -2, 'Mf': -64, 'Mt': 0.0},   # termica
                  'C+':{ 'N': 0.0, 'T': 0, 'Mf': 0, 'Mt': 0.0}, 'C-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0},   # cedimenti
                  'V+':{ 'N': 0.0, 'T': 0, 'Mf': 0, 'Mt': 0.0}, 'V-':{ 'N': 0.0, 'T': 0.0, 'Mf': 0.0, 'Mt': 0.0},   # vento
      }

# Convert dictionary to DataFrame
df = pd.DataFrame(Sollecitazioni).T.reset_index()
df.rename(columns={'index': 'Tipo'}, inplace=True)
   
if selected3 == "Sollecitazioni":

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

   st.session_state["dict_soll"] = updated_dict_soll
   
try:
   input_section = st.session_state["input_section"]
except:
   input_section = { 'l': [12*1000], #lunghezza del concio
                  
                     "bPsup": [250], #larghezza piattabanda superiore
                     'tPsup': [20], #spessore piattabanda superiore
                     "brPsup": [0], # larghezza raddoppio piattabanda superiore
                     'trPsup': [0], #spessore raddoppio piattabanda superiore
                     "ha": [360], #altezza anima
                     "ta": [12], #spessore anima
                     "brPinf": [0], #larghezza raddoppio piattabanda inferiore
                     'trPinf': [0], #spessore raddoppio piattabanda inferiore
                     "bPinf": [300], #larghezza piattabanda inferiore
                     'tPinf': [20], #spessore piattabanda inferiore

                     "hcls": [204],
                     "h_predalle": [6],
                     "Bcls": [1500],

                     "phi_sup": [20], 
                     "int_sup": [200],
                     "phi_inf": [1], 
                     "int_inf": [500],

                     "n_inf": [15], 
                     "n_0": [6],
                     "n_r": [15], 
                     "n_c": [16],

                     "c_sup": [40], 
                     "c_inf": [40],

                     "mat_cls": ["C35/40"],
                     "mat_steel": ["S355"],
                     }


if selected3 == "Sezione":


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
   updated_df_sec = edited_df_sec.applymap(lambda x: np.float64(x) if str(x).replace('.', '', 1).isdigit() else x)
   updated_dict_section = updated_df_sec.to_dict()
   st.json(updated_dict_section , expanded=False)
   st.session_state["input_section"] = updated_dict_section

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

   c_sup = float(edited_df_sec.loc['c_sup'][0])
   c_inf = float(edited_df_sec.loc['c_inf'][0])

   mat_cls = edited_df_sec.loc['mat_cls'][0]
   mat_steel = edited_df_sec.loc['mat_steel'][0]






   ## COSTRUZIONE SOLETTA IN CALCESTRUZZO
   clsSection = RectangularSection(Bcls, Hcls, [0, 0], material=mat_cls)
   pointG0 = [[-int_sup*i, c_sup] for i in range(0, int(Bcls*0.5/int_sup))] + [[int_sup*(i+1), c_sup] for i in range(0, int(Bcls*0.5/int_sup)-1)]

   pointG1 = [[-int_inf*i, Hcls-c_inf-hpredall] for i in range(0, int(Bcls*0.5/int_inf))] + [[int_inf*i, Hcls-c_inf-hpredall] for i in range(1, int(Bcls*0.5/int_inf))]
   b0 = renforcementBar(phi_sup, pointG0)
   b1 = renforcementBar(phi_inf, pointG1)
   st.session_state["clsSection"] = clsSection
   st.session_state["cls_bar"] = [b0, b1]

   # = rectangularCA(clsSection, [b0, b1])
   #cplot = plotSection_ploty(c)
   #cplot.show()


   ## COSTRUZIONE SEZIONE IN ACCIAIO 
   gapCls = hpredall+Hcls
   PlateSup = OrizontalPlate(bf, tbf, [0, gapCls], material=mat_steel)
   rPlateSup = OrizontalPlate(bf_r, tbrf, [0, gapCls+tbf], material=mat_steel)
   #vribs1 = V_rib(283, 300, 25, 6, [0, 70+16]) #cl4Dict={"Binst":75, "Be1":60}
   wPlate1 = WebPlate(hw, tw, [0, gapCls+tbf+tbrf], 0, material=mat_steel, cl4Dict=None)
   rPlateInf = OrizontalPlate(rbf_inf, rtbf_inf, [0, gapCls+tbf+tbrf+hw], material=mat_steel)
   PlateInf = OrizontalPlate(binf, tbf_inf, [0, (gapCls+tbf+tbrf+hw+rtbf_inf)], material=mat_steel)

   st.session_state["gapCls"] = gapCls
   st.session_state["PlateSup"] = PlateSup
   st.session_state["rPlateSup"] = rPlateSup
   st.session_state["wPlate1"] = wPlate1
   st.session_state["rPlateInf"] = rPlateInf
   st.session_state["PlateInf"] = PlateInf

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
   st.session_state["listDict"] = listDict
   #PARTIRE DA QUI PER CREARE IL MOMENTO STATICO PER IL CALCOLO DELLA FORZA NELLE SALDATURE

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

   st.session_state["dictProp"] = dictProp

   # per print table
   table["g1"] = { i :dictProp["g1"][i] for i in listParams}
   table["g2"] = { i :dictProp["g2"][i] for i in listParams}
   table["r"] = { i :dictProp["r"][i] for i in listParams}
   table["c"] = { i :dictProp["c"][i] for i in listParams}
   table["mobili"] = { i :dictProp["mobili"][i] for i in listParams}
   table["fe"] = { i :dictProp["fe"][i] for i in listParams}

   df_section_prop = pd.DataFrame.from_dict(table, orient = "index").T #.reset_index()
   st.write(df_section_prop)

   hi = [0, 
      st.session_state["gapCls"], 
      st.session_state["gapCls"], 
      st.session_state["gapCls"]+tbf, 
      st.session_state["gapCls"]+tbf+tbrf, 
      st.session_state["gapCls"]+tbf+tbrf+hw, 
      st.session_state["gapCls"]+tbf+tbrf+hw+rtbf_inf, 
      st.session_state["gapCls"]+tbf+tbrf+hw+rtbf_inf+tbf_inf]
   
   hi_plot = list(hi) + [hi[-1], hi[0], hi[0]]
   st.session_state["hi_plot"] = hi_plot


Acls = st.session_state["gapCls"]*st.session_state["input_section"]["0"]["Bcls"]

if selected3 == "Tensioni":
   ## PLOT TENSION POSITIVE
   
   tension_plot_plus, list_tension = tension(st.session_state["dictProp"], 
                                             st.session_state["dict_soll"], 
                                             st.session_state["hi_plot"][0:9],
                                             condition = "positive",
                                             n0 = st.session_state["input_section"]["0"]["n_0"],
                                             ninf = st.session_state["input_section"]["0"]["n_inf"],
                                             nr = st.session_state["input_section"]["0"]["n_r"],
                                             nc = st.session_state["input_section"]["0"]["n_c"]
                                             )

   #st.write(st.session_state["hi_plot"][0:9])
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

   tension_plot_neg, list_tension_neg = tension(st.session_state["dictProp"], 
                                             st.session_state["dict_soll"], 
                                             st.session_state["hi_plot"][0:9],
                                             condition = "negative",
                                             n0 = st.session_state["input_section"]["0"]["n_0"],
                                             ninf = st.session_state["input_section"]["0"]["n_inf"],
                                             nr = st.session_state["input_section"]["0"]["n_r"],
                                             nc = st.session_state["input_section"]["0"]["n_c"]
                                             )
   
   st.session_state["list_tension_pos"] = list_tension
   st.session_state["list_tension_neg"] = list_tension_neg

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
   tension_slu_neg, tension_rara_neg, tension_frequente_neg, tension_qp_neg = combinazione(list_tension_neg)

   st.session_state["tension_slu"] = [tension_slu, tension_slu_neg]
   st.session_state["tension_rara"] = [tension_rara, tension_rara_neg]
   st.session_state["tension_frequente"] = [tension_frequente, tension_frequente_neg]
   st.session_state["tension_qp"] = [tension_qp, tension_qp_neg]


   hi_plot_comb = st.session_state["hi_plot"][0:9]+[st.session_state["hi_plot"][0:9][-1], 0, 0]
   slu_plot = tension_plot(st.session_state["tension_slu"][0], hi_plot_comb)
   rara_plot = tension_plot(st.session_state["tension_rara"][0], hi_plot_comb)
   freq_plot = tension_plot(st.session_state["tension_frequente"][0], hi_plot_comb)
   qp_plot = tension_plot(st.session_state["tension_qp"][0], hi_plot_comb)

   slu_plot_neg = tension_plot(st.session_state["tension_slu"][1], hi_plot_comb)
   rara_plot_neg = tension_plot(st.session_state["tension_rara"][1], hi_plot_comb)
   freq_plot_neg = tension_plot(st.session_state["tension_frequente"][1], hi_plot_comb)
   qp_plot_neg = tension_plot(st.session_state["tension_qp"][1], hi_plot_comb)

   #st.write(hi_plot_comb)
   #st.write(tension_slu)

   st.title("Combinazione positive")
   tab23, tab24, tab25, tab26 = st.tabs(["slu", "rara", "frequente", "quasi permanente"])

   with tab23:
      st.plotly_chart(slu_plot, use_container_width=True, key= "tension_slu")
   with tab24:
      st.plotly_chart(rara_plot, use_container_width=True, key= "tension_rara")
   with tab25:
      st.plotly_chart(freq_plot, use_container_width=True, key= "tension_frequente")
   with tab26:
      st.plotly_chart(qp_plot, use_container_width=True, key= "tension_qp")

   st.title("Combinazione negative")
   tab27, tab28, tab29, tab30 = st.tabs(["slu", "rara", "frequente", "quasi permanente"])

   with tab27:
      st.plotly_chart(slu_plot_neg, use_container_width=True, key= "tension_slu_neg")
   with tab28:
      st.plotly_chart(rara_plot_neg, use_container_width=True, key= "tension_rara_neg")
   with tab29:
      st.plotly_chart(freq_plot_neg, use_container_width=True, key= "tension_frequente_neg")
   with tab30:
      st.plotly_chart(qp_plot_neg, use_container_width=True, key= "tension_qp_neg")

   


# Editable table using st.data_editor
# Elenco delle verifiche in Markdown
# Descrizione con Markdown e LaTeX

if selected3 == "Verifiche":
   st.markdown("""   
               #### Verifiche
               """)
   st.markdown("""   
               ##### 1) Calcolo della classe 
               """)

   st.markdown("""   
               ##### 2) Verifiche tensionali sugli elementi in acciaio (S.L.U.)
               """)
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
               ###### Combinazione positiva
               """)
   checkTension(st.session_state["tension_slu"][0], 335/1.05)
   st.markdown("""   
               ###### Combinazione negativa
               """)
   checkTension(st.session_state["tension_slu"][1], 335/1.05)

   st.markdown("""   
            ##### 3) Verifica a taglio - instabilità dell'anima (S.L.U.)
            """)

   #st.write(updated_dict_soll)
   Ved_pos = Sollecitazione_list(st.session_state["dict_soll"], condition = "positive", cds= "T")
   Ved_neg = Sollecitazione_list(st.session_state["dict_soll"], condition = "negative", cds= "T")

   Ved_slu_pos = combinazione(Ved_pos, category = "A1_sfav")
   Ved_slu_neg = combinazione(Ved_neg, category = "A1_sfav")
   taglio_anima = checkTaglio_Instabilita(st.session_state["input_section"]["0"]['ha'], st.session_state["input_section"]["0"]['ta'], 355)


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
   n_list = [st.session_state["input_section"]["0"]["n_0"],
   st.session_state["input_section"]["0"]["n_inf"],
   st.session_state["input_section"]["0"]["n_r"],
   st.session_state["input_section"]["0"]["n_c"]]
   
   #st.write(listDict[0:1])
   Stau_s1 = Sx_plate(st.session_state["listDict"][0:1], 
                      st.session_state["clsSection"], 
                      st.session_state["dictProp"], 
                      condition = "positive", 
                      bar = st.session_state["cls_bar"],
                      n = n_list )
   
   V_s1_pos = np.array(Ved_pos)/(np.array(Stau_s1[1]))
   V_s1_neg = np.array(Ved_neg)/(np.array(Stau_s1[1]))

   Vs1_comb_pos = combinazione(list(V_s1_pos), category = "A1_sfav")
   Vs1_comb_neg = combinazione(list(V_s1_neg), category = "A1_sfav")

   #st.write(Stau_s1[1])
   # saldature raddoppio piattabanda superiore con anima
   Stau_s2 = Sx_plate(st.session_state["listDict"][0:2], st.session_state["clsSection"], st.session_state["dictProp"], condition = "positive", bar = st.session_state["cls_bar"], n = n_list)
   V_s2_pos = np.array(Ved_pos)/(np.array(Stau_s2[1]))
   V_s2_neg = np.array(Ved_neg)/(np.array(Stau_s2[1]))

   Vs2_comb_pos = combinazione(list(V_s2_pos), category = "A1_sfav")
   Vs2_comb_neg = combinazione(list(V_s2_neg), category = "A1_sfav")
   #st.write(Stau_s2[1])
   # saldature raddoppio piattabanda inferiore con anima
   Stau_s3 = Sx_plate(st.session_state["listDict"][0:3], st.session_state["clsSection"], st.session_state["dictProp"], condition = "positive", bar = st.session_state["cls_bar"], n = n_list)
   V_s3_pos = np.array(Ved_pos)/(np.array(Stau_s3[1]))
   V_s3_neg = np.array(Ved_neg)/(np.array(Stau_s3[1]))

   Vs3_comb_pos = combinazione(list(V_s3_pos), category = "A1_sfav")
   Vs3_comb_neg = combinazione(list(V_s3_neg), category = "A1_sfav")
   #st.write(Stau_s3[1])
   # saldature piattabanda inferiore con anima
   Stau_s4 = Sx_plate(st.session_state["listDict"][0:4], st.session_state["clsSection"], st.session_state["dictProp"], condition = "positive", bar = st.session_state["cls_bar"], n = n_list)
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

   Sx, zx = Sx_slab(st.session_state["input_section"]["0"]["Bcls"], st.session_state["gapCls"],  st.session_state["dictProp"], condition = "positive")

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
   st.image("Screenshot 2025-01-28 182905.png", caption="Dettagli a Fatica per travi saldate")
   st.image("Screenshot 2025-01-28 182928.png", caption="Dettagli a Fatica per travi bullonate")

      # Dizionario con i coefficienti di sicurezza
   gamma_mf_values = {
      ("Poco sensibile", "Moderate"): 1.00,
      ("Poco sensibile", "Significative"): 1.15,
      ("Sensibile", "Moderate"): 1.15,
      ("Sensibile", "Significative"): 1.35
   }

   # Selezione dei parametri
   sensibilita = st.selectbox("Seleziona la sensibilità alla fatica:", ["Poco sensibile", "Sensibile"])
   conseguenze = st.selectbox("Seleziona le conseguenze della rottura:", ["Moderate", "Significative"])

   # Mostra il coefficiente corrispondente
   gamma_mf = gamma_mf_values[(sensibilita, conseguenze)]
   st.write(f"### Coefficiente di sicurezza γₘ𝒻 = {gamma_mf:.2f}")

   delta_sigma = (np.array(st.session_state["list_tension_pos"][3]) - np.array(st.session_state["list_tension_neg"][3]))*gamma_mf
   delta_tau_cls = (V_pioli_pos[3] - V_pioli_neg[3])*gamma_mf
   delta_tau1 = (V_s1_pos[3] - V_s1_neg[3])*gamma_mf
   delta_tau2 = (V_s2_pos[3] - V_s2_neg[3])*gamma_mf
   delta_tau3 = (V_s3_pos[3] - V_s3_neg[3])*gamma_mf
   delta_tau4 = (V_s4_pos[3] - V_s4_neg[3])*gamma_mf

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
      "scorrimento [KN]": [
         0.0,
         delta_tau_cls,
         delta_tau1,
         delta_tau2,
         delta_tau2,
         delta_tau3,
         delta_tau3,
         delta_tau4,
      ],
      "Dettaglio": [
         [71, 81],
         [81],
         [81],
         [81],
         [81],
         [81],
         [81],
         [81],
      ],
   }

   # Creazione del dataframe
   df = pd.DataFrame(data)

   # Opzioni disponibili per i dettagli
   dettagli_disponibili = [71, 80, 90, 100, 112, 125, 160]

   # Lista per salvare i dettagli aggiornati
   dettagli_aggiornati = []

   for index, row in df.iterrows():
      # Widget multiselect per ogni riga
      dettagli_selezionati = st.multiselect(
         f"Seleziona i dettagli per {row['Punto']}",
         options=dettagli_disponibili,
         default=[],
         key=f"multiselect_{index}",
      )
      dettagli_aggiornati.append(dettagli_selezionati)

   # Aggiornamento del dataframe con i dettagli selezionati
   df["Dettaglio"] = dettagli_aggiornati

   # Visualizzazione del dataframe aggiornato
   st.write("Dataframe aggiornato:")
   st.dataframe(df)


   ###