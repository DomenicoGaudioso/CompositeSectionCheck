import numpy as np
import streamlit as st
#import matplotlib.pyplot as plt
from SGMPy_material import *
from SGMPy_section_v01 import *
from SGMPy_beff import *


def ConcioPropCalculateFEA(dictConci, n = [18.34, 17.53, 22.66, 6.45]): #Calcolo proprietà lorde della sezione senza considerare Beff
   
   dictProp = {}
   for i in dictConci:
      #costant Prop
      tw = dictConci[i]['SectionDefine']['tickness']['Web']
      H = dictConci[i]['SectionDefine']['variabile']["H"][0]
      tc = dictConci[i]["slab"]["hc"]
      phi = dictConci[i]["slab"]["D_bar"]
      p = dictConci[i]["slab"]["int_bar"] # passo bar
      
      ## stiffnes property
      irrSup = dictConci[i]['SectionDefine']['stiffnes']["sup"]
      irrWeb = dictConci[i]['SectionDefine']['stiffnes']["web"]
      irrInf = dictConci[i]['SectionDefine']['stiffnes']["inf"]
      
      #fase 0
      bsup_ext_f0 = dictConci[i]["IntSup_ext"]
      bsup_int_f0 = dictConci[i]["IntSup_int"]
      binf_ext_f0 = dictConci[i]["IntInf_ext"]
      binf_int_f0 = dictConci[i]["IntInf_int"]
      
      try:
         fase0 = dictConci[i]["fase0"][0]
         if fase0 == "IntSup_ext":
            bsup_ext_f0 = [dictConci[i]["fase0"][1]]
         elif fase0 == "IntSup_int":
            bsup_int_f0 = [dictConci[i]["fase0"][1]] 
         elif fase0 == "IntInf_ext":
            binf_ext_f0 = [dictConci[i]["fase0"][1]] 
         elif fase0 == "IntInf_int":
            binf_ext_f0 = [dictConci[i]["fase0"][1]] 
      except:
         pass
               
               
      #fasi successive alla fase 0
      bsup_ext = dictConci[i]["IntSup_ext"]
      bsup_int = dictConci[i]["IntSup_int"]
      binf_ext = dictConci[i]["IntInf_ext"]
      binf_int = dictConci[i]["IntInf_int"]
               

      
      ## SECTION PROP CALCULATE
      dictProp[i] = {"fase 0": None, "fase 1": None, "fase 2": None, "fase 3": None, "fase 4": None }
      
      for bi_sup_e, bi_sup_i, bi_inf_e, bi_inf_i in zip(bsup_ext_f0, bsup_int_f0, binf_ext_f0, binf_int_f0):
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
         if irrSup["type"] == "V":
            vribs1 = V_rib(irrSup["params"][0], irrSup["params"][1], irrSup["params"][2], irrSup["params"][3], [0, tc+tpsup]) #cl4Dict={"Binst":75, "Be1":60}
         st.write(tw)
         wPlate1 = WebPlate(H-tpsup-tpinf, tw, [0, tc+tpsup], 0, material=None, cl4Dict=None)
         
         if irrWeb != "None" and irrWeb["type"] == "L":
            WebRibs = L__rib(irrWeb["params"][0], irrWeb["params"][1], irrWeb["params"][2], irrWeb["params"][3], angle = irrWeb["params"][4] ) #
            p_WebRibs = irrWeb["dist"] # passo ribs
            Hw = H - tpsup - tpinf
            n_WebRibs = int(np.floor(Hw/p_WebRibs)) #numero di ribs
            int_WebRibs = [-p_WebRibs*(i) for i in range(1, n_WebRibs)] 
         else:
            WebRibs = None
            int_WebRibs = None
            
         
         orPlateInf1 = OrizontalPlate(beffInf, tpinf, [0, (tc+H-tpinf)])
         
         if irrInf["type"] == "L":
            pInfRibs = L__rib(irrInf["params"][0], irrInf["params"][1], irrInf["params"][2], [0, H+tc-tpinf-irrInf["params"][0]]) 
            
         ## V ribs
         p_ribs1 = irrSup["dist"] # passo ribs
         n_ribs1 = int(np.floor(beffSup/p_ribs1)) #numero di ribs
         intRibs1 = [p_ribs1*i - (n_ribs1-1)*p_ribs1*0.5 for i in range(0, n_ribs1)] 
         ## Plate inf ribs
         p_PinfRibs = irrInf["dist"] # passo ribs
         n_PinfRibs = int(np.floor(beffInf/p_PinfRibs)) #numero di ribs
         int_PinfRibs = [p_PinfRibs*i - (n_PinfRibs-1)*p_PinfRibs*0.5 for i in range(0, n_PinfRibs)] 
         if tw == 0:
            SectionnNassirya = OrthotropicSection1(orPlateSup1, vribs1,  intRibs1,  None,  WebRibs,  int_WebRibs, orPlateInf1, pInfRibs,  int_PinfRibs)
         else:
            SectionnNassirya = OrthotropicSection1(orPlateSup1, vribs1,  intRibs1,  wPlate1,  WebRibs,  int_WebRibs, orPlateInf1, pInfRibs,  int_PinfRibs)
         #plotSection(SectionnNassirya, center=True, propTable=True)

         #Fase 0 - Construction - partial section
         sectionProp = SectionnNassirya
         dictProp[i]["fase 0"] = sectionProp
      
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
         if irrSup["type"] == "V":
            vribs1 = V_rib(irrSup["params"][0], irrSup["params"][1], irrSup["params"][2], irrSup["params"][3], [0, tc+tpsup]) #cl4Dict={"Binst":75, "Be1":60}
         
         
         wPlate1 = WebPlate(H-tpsup-tpinf, tw, [0, tc+tpsup], 0, material=None, cl4Dict=None)
         
         if irrWeb != "None" and irrWeb["type"] == "L":
            WebRibs = L__rib(irrWeb["params"][0], irrWeb["params"][1], irrWeb["params"][2], irrWeb["params"][3], angle = irrWeb["params"][4]) 
            p_WebRibs = irrWeb["dist"] # passo ribs
            Hw = H - tpsup - tpinf
            n_WebRibs = int(np.floor(Hw/p_WebRibs)) #numero di ribs
            int_WebRibs = [-p_WebRibs*(i) for i in range(1, n_WebRibs)] 
         else:
            WebRibs = None
            int_WebRibs = None
         
         orPlateInf1 = OrizontalPlate(beffInf, tpinf, [0, (tc+H-tpinf)])
         
         if irrInf["type"] == "L":
            pInfRibs = L__rib(irrInf["params"][0], irrInf["params"][1], irrInf["params"][2], [0, H+tc-tpinf-irrInf["params"][0]]) 
            
         ## V ribs
         p_ribs1 = irrSup["dist"] # passo ribs
         n_ribs1 = int(np.floor(beffSup/p_ribs1)) #numero di ribs
         intRibs1 = [p_ribs1*i - (n_ribs1-1)*p_ribs1*0.5 for i in range(0, n_ribs1)] 
         ## Web ribs

         ## Plate inf ribs
         p_PinfRibs = irrInf["dist"] # passo ribs
         n_PinfRibs = int(np.floor(beffInf/p_PinfRibs)) #numero di ribs
         int_PinfRibs = [p_PinfRibs*i - (n_PinfRibs-1)*p_PinfRibs*0.5 for i in range(0, n_PinfRibs)] 
         if tw == 0:
            SectionnNassirya = OrthotropicSection1(orPlateSup1, vribs1,  intRibs1,  None,  WebRibs,  int_WebRibs, orPlateInf1, pInfRibs,  int_PinfRibs)
         else:
            SectionnNassirya = OrthotropicSection1(orPlateSup1, vribs1,  intRibs1,  wPlate1,  WebRibs,  int_WebRibs, orPlateInf1, pInfRibs,  int_PinfRibs)
         #plotSection(SectionnNassirya, center=True, propTable=True)
         
         #Fase 1 - G1 - Only steel
         sectionProp1 = SectionnNassirya
         dictProp[i]["fase 1"] = sectionProp1
         
         #Fase 2 - G2 - t inf
         sectionProp2 = CompositeSection(SectionnNassirya, clsSection, [b0], n[0])
         dictProp[i]["fase 2"] = sectionProp2
         
         #Fase 3 - R - ritiro
         sectionProp3 = CompositeSection(SectionnNassirya, clsSection, [b0], n[1])
         dictProp[i]["fase 3"] = sectionProp3
         
         #Fase 4 - eps - Cedimenti imposti
         sectionProp4 = CompositeSection(SectionnNassirya, clsSection, [b0], n[2])
         dictProp[i]["fase 4"] = sectionProp4
         
         #Fase 5 - Qm - mobili
         sectionProp4 = CompositeSection(SectionnNassirya, clsSection, [b0], n[3])
         dictProp[i]["fase 5"] = sectionProp4
         
   return dictProp


def ConcioPropCalculate(dictConci, n = [18.34, 17.53, 6.45]):
   dictProp = {}
   for i in dictConci:
      #costant Prop
      tw = dictConci[i]['SectionDefine']['tickness']['Web']
      H = dictConci[i]['SectionDefine']['variabile']["H"][0]
      tc = dictConci[i]["slab"]["hc"]
      phi = dictConci[i]["slab"]["D_bar"]
      p = dictConci[i]["slab"]["int_bar"] # passo bar
      
      ## stiffnes property
      irrSup = dictConci[i]['SectionDefine']['stiffnes']["sup"]
      irrWeb = dictConci[i]['SectionDefine']['stiffnes']["web"]
      irrInf = dictConci[i]['SectionDefine']['stiffnes']["inf"]
      
      
      
      bsup_ext = dictConci[i]['SectionDefine']['variabile']["IntSup_ext"]
      bsup_int = dictConci[i]['SectionDefine']['variabile']["IntSup_int"]
      binf_ext = dictConci[i]['SectionDefine']['variabile']["IntInf_ext"]
      binf_int = dictConci[i]['SectionDefine']['variabile']["IntInf_int"]
      
      ## SECTION PROP CALCULATE
      dictProp[i] = {"fase 0": [], "fase 1": [], "fase 2": [], "fase 3": [], "fase 4": [] }
      
      for bi_sup_e, bi_sup_i, bi_inf_e, bi_inf_i in zip(bsup_ext, bsup_int, binf_ext, binf_int):
         beffSup = bi_sup_e[i] + bi_sup_i[i]
         beffInf = bi_inf_e[i] + bi_inf_i[i]
         
         #calcolo t equivalente
         Asup = dictConci[i]['SectionDefine']['tickness']['Psup_ext']*bsup_ext[i] + dictConci[i]['SectionDefine']['tickness']['Psup_int']*bsup_int[1]
         tpsup = Asup/(beffSup)
         Asup = dictConci[i]['SectionDefine']['tickness']['Pinf_ext']*bsup_ext[i] + dictConci[i]['SectionDefine']['tickness']['Pinf_int']*bsup_int[1]
         tpinf = Asup/(beffInf)
         
         #assemble section
         clsSection = RectangularSection(beffSup, tc, [0, 0])
         n_bar = int(np.floor(beffSup/p)) #numero di bar
         pointG0 = [[p*i - (n_bar-1)*p*0.5, tc/2] for i in range(0, n_bar)] 
         b0 = renforcementBar(phi, pointG0 )
         
         orPlateSup1 = OrizontalPlate(beffSup, tpsup, [0, tc])
         
         if irrSup["type"] == "V":
            vribs1 = V_rib(irrSup["params"][0], irrSup["params"][1], irrSup["params"][2], irrSup["params"][3], [0, tc+tpsup]) #cl4Dict={"Binst":75, "Be1":60}
         
         
         wPlate1 = WebPlate(H-tpsup-tpinf, tw, [0, tc+tpsup], 0, material=None, cl4Dict=None)
         
         if irrWeb["type"] == "L":
            WebRibs = L__rib(irrWeb["params"][0], irrWeb["params"][1], irrWeb["params"][2], [tw, tc+tpsup], angle = irrWeb["params"][4]) 
            
         
         orPlateInf1 = OrizontalPlate(beffInf, tpinf, [0, (tc+H-tpinf)])
         
         if irrInf["type"] == "L":
            pInfRibs = L__rib(irrInf["params"][0], irrInf["params"][1], irrInf["params"][2], [0, H+tc-tpinf]) 
            
         ## V ribs
         p_ribs1 = irrSup["dist"] # passo ribs
         n_ribs1 = int(np.floor(beffSup/p_ribs1)) #numero di ribs
         intRibs1 = [p_ribs1*i - (n_ribs1-1)*p_ribs1*0.5 for i in range(0, n_ribs1)] 
         ## Web ribs
         p_WebRibs = irrWeb["dist"] # passo ribs
         Hw = H - tpsup - tpinf
         n_WebRibs = int(np.floor(Hw/p_WebRibs)) #numero di ribs
         int_WebRibs = [-p_WebRibs*(i) for i in range(1, n_WebRibs)] 
         ## Plate inf ribs
         p_PinfRibs = irrInf["dist"] # passo ribs
         n_PinfRibs = int(np.floor(beffInf/p_PinfRibs)) #numero di ribs
         int_PinfRibs = [p_PinfRibs*i - (n_PinfRibs-1)*p_PinfRibs*0.5 for i in range(0, n_PinfRibs)] 
         SectionnNassirya = OrthotropicSection1(orPlateSup1, vribs1,  intRibs1,  wPlate1,  WebRibs,  int_WebRibs, orPlateInf1, pInfRibs,  int_PinfRibs)
         #plotSection(SectionnNassirya, center=True, propTable=True)

         #Fase 0 - Construction - partial section
         dictProp[i]["fase 0"].append(SectionnNassirya)
         #Fase 1 - G1 - Only steel
         dictProp[i]["fase 1"].append(SectionnNassirya)
         #Fase 2 - G2 - t inf
         dictProp[i]["fase 2"].append(CompositeSection(SectionnNassirya, clsSection, [b0], n[0]))
         #Fase 3 - R - ritiro
         dictProp[i]["fase 3"].append(CompositeSection(SectionnNassirya, clsSection, [b0], n[1]))
         #Fase 4 - cedimenti imposti
         dictProp[i]["fase 4"].append(CompositeSection(SectionnNassirya, clsSection, [b0], n[2]))
         #Fase 5 - Qm - mobili
         dictProp[i]["fase 5"].append(CompositeSection(SectionnNassirya, clsSection, [b0], n[3]))
         
   return dictProp


def concioTableProp(dictConcio):
   dictTable = {}
   fase = ["fase 0", "fase 1", "fase 2", "fase 3", "fase 4", "fase 5"]
   dictkey = list(dictConcio.keys())
   for ic in dictkey:
      nSec = len(dictConcio[ic]["fase 0"])
      try:
         for isec in range(0, nSec):
            name = "C{}_S{}".format(ic,isec)
            dictTable[name] = {}
            for ifase in fase:
               A = round(dictConcio[ic][ifase][isec]["A"],2)
               Az = round(dictConcio[ic][ifase][isec]["A"]*dictConcio[ic][ifase][isec]["az"],2)
               Iy = round(dictConcio[ic][ifase][isec]["Iy"],2)
               yg = round(dictConcio[ic][ifase][isec]["Pg"][1],2)
               dictTable[name][ifase] = {"A": A, "Az": A, "Iy": Iy, "yg": yg}
      except:
            for ifase in fase:
               A = round(dictConcio[ic][ifase]["A"],2)
               Ay = round(dictConcio[ic][ifase]["Ay"],2)
               Az = round(dictConcio[ic][ifase]["Az"],2)
               Iy = round(dictConcio[ic][ifase]["Iy"],2)
               Iz = round(dictConcio[ic][ifase]["Iz"],2)
               yg = round(dictConcio[ic][ifase]["Pg"][1],2)
               dictTable[name][ifase] = {"A": A, "Ay": Ay, "Az": Az, "Iy": Iy, "Iz": Iz, "yg": yg}
         
         
               
   return dictTable

def reportConcio(dictTable, dictConcio = None):
   st.markdown("**Calcolo delle proprietà delle sezioni di ogni concio strutturale**  ")
   
   multi = '''**Colonne**  
   Fase 0: Sezione in acciaio - sezione non completa,    
   Fase 1: Sezione in acciaio - sezione completa,  
   Fase 2: Sezione in acciaio con soletta in calcestruzzo collaborante (G2 - t∞),  
   Fase 3: Sezione in acciaio con soletta in calcestruzzo collaborante (Ritiro - ts),  
   Fase 4: Sezione in acciaio con soletta in calcestruzzo collaborante (Cedimenti - ts),   
   Fase 5: Sezione in acciaio con soletta in calcestruzzo collaborante (Mobili - t0),    
   
   **Righe**  
   A: area della sezione in mm^2, nel caso di sezione mista si riporta l'area omogenizzata all'acciaio,  
   Iy: momento d'inerzia della sezione in mm^4, nel caso di sezione mista si riporta l'inerzia omogenizzata all'acciaio,  
   yg: distanza del baricentro rispetto al lembo superiore, nel caso di sezione mista si riporta il valore rispetto all'estradosso della soletta.  
     
       
   Nota: se la struttura in acciao è subito costruita in modo completo allo la fase 0 e la fase 1 coincidono.  
   '''
   st.markdown(multi)
   
   if dictConcio is not None:
      keyConcio = list(dictConcio.keys())
   
   key = list(dictTable.keys())
   for num, i in enumerate(key):
      nome = i
      df = pd.DataFrame(dictTable[i])
      st.markdown("**{}**  ".format(nome))
      
      if dictConcio is not None:
         st.write(keyConcio[num])
         st.write("PLOT FASE 0 - PARTIAL STEEL")
         fig = plotSection_matLib(dictConcio[keyConcio[num]]["fase 0"], center=False, propTable=False)
         #st.pyplot(fig, use_container_width = True)
         st.plotly_chart(fig)
         st.write("PLOT FASE 1 - COMPLETE STEEL")
         fig = plotSection_matLib(dictConcio[keyConcio[num]]["fase 1"], center=False, propTable=False)
         #st.pyplot(fig, use_container_width = True)
         st.plotly_chart(fig)
         st.write("PLOT COMPOSITE SECTION")
         fig = plotSection_matLib(dictConcio[keyConcio[num]]["fase 2"], center=False, propTable=False)
         #st.pyplot(fig, use_container_width = True)
         st.plotly_chart(fig)
         
      st.dataframe(df, width=1200)
      
   return