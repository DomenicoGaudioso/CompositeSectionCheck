import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
from SGMPy_material import *
from SGMPy_section_v01 import *
import streamlit as st


## LAMBDA FUNCTION

## FUNCTION 
def Progress(dictInput, dictCorrConci):
    # Calcolo delle progressive 
    start = []
    end = []
    xs = 0
    for i in dictCorrConci['Conci officina']:
        xe = xs + dictInput[str(i)]["L"]
        start.append(xs)
        end.append(xe)
        try:
            dictInput[str(i)]["prog"].append([xs, xe])
        except:
            dictInput[str(i)]["prog"] = [[xs, xe]]
            
        xs = xe
    
    return dictInput

def lenghtEquivalent(dictInput, dictCorrConci):
    dictLen = {}
    x = 0
    for i, j in enumerate(dictCorrConci["campate"]):
        if dictCorrConci["type"][i] == "sbalzo":
            le = 2*j
            x_start = x 
            x_end = x+j
            
            dictLen[x_start] = {"x_start": x_start, "type": "cantilever", "Le": le}
            dictLen[x_end] = {"x_start": x_end, "type": "cantilever", "Le": le}
            
        elif dictCorrConci["type"][i] == "campata":
            le = 0.70*j
            x_start = x+j/4 
            x_end = x+j-j/4
            
            x_start = x 
            x_end = x+j/4
            
            if dictCorrConci["type"][i-1] == "sbalzo":
                le_app=2*dictCorrConci["campate"][i-1]
                dictLen[x_end] = {"x_start": x_start, "type": "cantilever", "Le": le_app}
            else: 
                le_app=0.25*(j + dictCorrConci["campate"][i-1])
                dictLen[x_end] = {"x_start": x_start, "type": "hogging bending"}
                
            dictLen[x_end] = {"x_start": x_end, "type": "sagging bending", "Le": le}
            
            x_start = x+j/4+j/2 
            x_end = x+j
            le_app=0.25*(j + dictCorrConci["campate"][i+1])
            dictLen[x_start] = {"x_start": x_start, "type": "sagging bending", "Le": le}
            dictLen[x_end] = {"x_start": x_end, "type": "hogging bending", "Le": le_app}
            
        elif dictCorrConci["type"][i] == "spalla_end":
            #appoggio M-
            le = 0.85*j
            
            x_start = x 
            x_end = x+j/4
            le_app=0.25*(j + dictCorrConci["campate"][i-1])
            dictLen[x_start] = {"x_start": x_start, "type": "hogging bending", "Le": le_app}
            dictLen[x_end] = {"x_start": x_end, "type": "sagging bending", "Le": le}
            #appoggio M nullo
            x_start = x+j/4 
            x_end = x+j-j/4
            
            x_start = x+j-j/4 
            x_end = x+j
            dictLen[x_start] = {"x_start": x_start, "type": "sagging bending", "Le": le}
            dictLen[x_end] = {"x_start": x_end, "type": "end support", "Le": le}
            
        x = x + j
    #plt.show()
    
    return dictLen

def le_plot(data):
    # Estraiamo le coordinate per il plot
    x_values = [data[key]['x_start'] for key in data]
    y_values = [data[key]['Le'] for key in data]

    # Creiamo il plot
    plt.plot(x_values, y_values, marker='o', linestyle='-')

    # Aggiungiamo le etichette agli assi
    plt.xlabel('x_start')
    plt.ylabel('Le')

    # Aggiungiamo un titolo al grafico
    plt.title('Plot di Le rispetto a x_start')

    # Visualizziamo il grafico
    plt.grid(True)
    plt.show()
    return

def beff_plot(data, fig = False, ys = 0.0, versor = 1):
    # Estraiamo le coordinate per il plot
    x_values = [data[key]['x_start'] for key in data]
    y_values = [data[key]['beff']*versor+ys for key in data]
    labels = [int(data[key]['beff']) for key in data]
    # Creiamo il plot
    if fig == False:
        fig = go.Figure()
        
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers+text', text=labels, textposition='top right', textfont_size=8))
    
    # Aggiunta delle linee tratteggiate
    for i, j in zip(x_values,y_values):
        fig.add_shape(type="line",
                    x0=i, y0=ys, 
                    x1=i, y1=j+ys, 
                    line=dict(color="gray", width=1, dash="dash"))
        
        fig.add_annotation(x=i, y=0,
                    text=str(i),
                    arrowcolor="blue",
                    arrowsize=1,
                    arrowwidth=1,
                    arrowhead=1,
                    font=dict(size=8))
    
    # Aggiungiamo le etichette agli assi
    # Aggiungiamo le etichette agli assi
    fig.update_xaxes(title_text='Length [mm]')
    fig.update_yaxes(title_text='beff [mm]')

    # Aggiungiamo un titolo al grafico
    #fig.update_layout(title='beff', showlegend=False)
    
    # Visualizziamo il grafico
    # Impostazione dell'uguaglianza degli assi
    fig.update_layout(autosize=False, width=1300, height=500)

    return fig

def ao_factor(Asl, bo, t):
    ao = np.sqrt(1+Asl/(bo*t))
    return ao

def k_factor(ao, bo, Le):
    k = ao*bo/Le
    return k

def Beta_factor(k, typeZone):

    if typeZone == "sagging bending":
        if k <= 0.02: 
            b = 1.0
        elif k <= 0.70 and k > 0.02:
            b = 1/(1+6.4*k**2)
        elif k > 0.70:
            b = 1/(5.9*k)
    elif typeZone == "hogging bending":
        if k <= 0.02: 
            b = 1.0
        elif k <= 0.70 and k > 0.02:
            b = 1/(1+6.0*(k-1/(2500*k)) + 1.6*k**2)
        elif k > 0.70:
            b = 1/(8.6*k)
    elif typeZone == "end support":
        if k <= 0.02: 
            b1 = 1.0
        elif k <= 0.70 and k > 0.02:
            b1 = 1/(1+6.4*k**2)
        elif k > 0.70:
            b1 = 1/(5.9*k)
        b = min((0.55+0.025/k)*b1, b1)
    elif typeZone == "cantilever":
        if k <= 0.02: 
            b2 = 1.0
        elif k <= 0.70 and k > 0.02:
            b2 = 1/(1+6.0*(k-1/(2500*k)) + 1.6*k**2)
        elif k > 0.70:
            b2 = 1/(8.6*k)
        b = b2
         
    return b

def find_field(dictionary, x_start, x_end):
    keys = sorted(dictionary.keys())  # Ordina le chiavi in modo crescente
    fields = []  # Lista per memorizzare i campi attraversati

    for i in range(len(keys) - 1):  # Itera fino al penultimo elemento
        current_key = keys[i]
        next_key = keys[i + 1]

        # Caso in cui l'intervallo inizia e termina all'interno del campo corrente
        if x_start >= current_key and x_end <= next_key:
            field_start = x_start
            field_end = x_end
            fields.append({'x': field_start, 'Le': dictionary[current_key]["Le"], 'type': dictionary[current_key]['type']})
            fields.append({'x': field_end, 'Le': dictionary[current_key]["Le"], 'type': dictionary[current_key]['type']})
        # Caso in cui l'intervallo inizia prima del campo corrente e termina all'interno del campo corrente
        elif x_start < current_key and x_end <= next_key and x_end >= current_key:
            field_start = current_key
            field_end = x_end
            fields.append({'x': field_start, 'Le': dictionary[current_key]["Le"], 'type': dictionary[current_key]['type']})
            fields.append({'x': field_end, 'Le': dictionary[current_key]["Le"], 'type': dictionary[current_key]['type']})
        # Caso in cui l'intervallo inizia all'interno del campo corrente e termina dopo il campo corrente
        elif x_start >= current_key and x_end > next_key and x_end <= current_key:
            field_start = x_start
            field_end = next_key
            fields.append({'x': field_start, 'Le': dictionary[current_key]["Le"], 'type': dictionary[current_key]['type']})
            fields.append({'x': field_end, 'Le': dictionary[current_key]["Le"], 'type': dictionary[current_key]['type']})
        # Caso in cui l'intervallo attraversa completamente il campo corrente
        elif x_start < current_key and x_end > next_key:
            field_start = current_key
            field_end = next_key
            fields.append({'x': field_start, 'Le': dictionary[current_key]["Le"], 'type': dictionary[current_key]['type']})
            fields.append({'x': field_end, 'Le': dictionary[current_key]["Le"], 'type': dictionary[current_key]['type']})
    
    if fields:
        return fields
    else:
        return "Campo non trovato"


def BeffGeneral(dataLe, as_steff, as_steff_eff, int_steff, bo, t, panInfo = 1): #C4.2.3.1.3.4.3 Larghezza collaborante
    #Calcolo della Beff per un cassone
    # 1: calcolo ao per lo sbalzo e per l'interasse centrale
    # 2: calcolo k per lo sbalzo e per l'interasse centrale
    # 3: cerco le Le che ricadono nelle progressive del concio
    # 4: calcolo beff per il concio. La beff può essere anche con 3 tratti diversi
    Asl = as_steff*(bo/int_steff)
    Asl_eff = as_steff_eff*(bo/int_steff)
    Ac_eff = bo*t + Asl_eff
    
    dictBeff = {}
    
    #plt.figure()
    for i in dataLe: # itero per concio
        dictBeff[i] = {'x_start':i}
        # 1 plate superiore --> compresse le parti a momento positivo ##SLU
        # 2 plate inf --> compresse le parti a momento negativo ##SLU
        # 3 non lo considera ##SLE
                
        ao = ao_factor(Asl, bo, t)
        k = k_factor(ao, bo, dataLe[i]["Le"])
        beta = Beta_factor(k, dataLe[i]["type"])
        
        if panInfo == 1:
            if dataLe[i]["type"] == "sagging bending" or dataLe[i]["type"] == "end support":
                asl = max(Ac_eff*beta**k, Ac_eff*beta)
            else:
                asl = max(Ac_eff*beta**k, Ac_eff*beta)
                
        elif panInfo == 2:
            if dataLe[i]["type"] == "hogging bending" or dataLe[i]["type"] == "cantilever":
                asl = max(Ac_eff*beta**k, Ac_eff*beta)
            else:
                asl = max(Ac_eff*beta**k, Ac_eff*beta)

        else:
            asl = Asl
            
        ao = ao_factor(asl, bo, t)
        k = k_factor(ao, bo, dataLe[i]["Le"])
        beta = Beta_factor(k, dataLe[i]["type"])
            

        beff = beta*bo
            
        dictBeff[i]["Asl"] = Asl,
        dictBeff[i]["ao"] = ao
        dictBeff[i]["k"] = k
        dictBeff[i]["beta"] = beta
        dictBeff[i]["beff"] = beff
        
    return dictBeff
    
def beffSum(dict1, dict2):
    dictBeff = {}
    for key in dict1.keys():
        dictBeff[key] = {
            'x_start': dict1[key]['x_start'],
            'ao': None,
            'k': None,
            'beta': None,
            'Asl': dict1[key]['Asl'] + dict2[key]['Asl'],
            'beff': dict1[key]['beff'] + dict2[key]['beff']
        }

    return dictBeff

def plotConci(dictConci, fig):
   
    for i in dictConci:
        x1 = dictConci[i]['prog'][0][0]
        y1 = -dictConci[i]['SectionDefine']['variabile']["H"][0]
        x2 = dictConci[i]['prog'][0][1]
        y2 = -dictConci[i]['SectionDefine']['variabile']["H"][1]
        
        # Calcolo delle coordinate dei vertici del rettangolo
        x = [x1, x2, x2, x1, x1]
        y = [y1, y2, y2, y1, y1]
        
        # Aggiunta del rettangolo
        #fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
        
        fig.add_vrect(x0=x1, x1=x2, 
              annotation_text=i, annotation_position="bottom", line_width=1, line_color ="red")
    
    return fig

def find_intersection_point(polyline, x_coordinate):
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]

        # Controlla se il segmento è verticale
        if x1 == x2:
            # Controlla se la x_coordinate si trova sulla linea verticale
            if x1 == x_coordinate:
                return x1, min(y1, y2) if y1 != y2 else y1
        else:
            # Calcola la pendenza e l'intercetta del segmento
            m = (y2 - y1) / (x2 - x1)
            q = y1 - m * x1

            # Controlla se la x_coordinate è compresa tra x1 e x2
            if x1 <= x_coordinate <= x2 or x2 <= x_coordinate <= x1:
                # Calcola l'ordinata corrispondente
                y_intersect = m * x_coordinate + q
                return x_coordinate, y_intersect

    return None  # Nessuna intersezione trovata

#'IntSup_ext': [5.6], 'IntSup_int': [4.0], 'IntInf_ext': [5.6], 'IntInf_int'
# Funzione per trovare tutti i punti della beff compresi tra i punti iniziale e finale del concio
def assign_BeffConci(dictConci, dict_beff, typeBeff="b0sup", index = 0): 
    
    x_beff = [dict_beff[key]['x_start'] for key in dict_beff]  # Ordina i valori x
    y_beff = [dict_beff[key]['beff'] for key in x_beff]  # Trova i valori y corrispondenti agli x ordinati
    polyline = [(i, j) for i, j in zip(x_beff, y_beff)]
    
    for key, value in dictConci.items():
        # Ottieni i punti x iniziali e finali del concio dal campo 'prog'
        x_concio_start = value['prog'][0][0]
        x_concio_end = value['prog'][0][1]

        # Inizializza una lista vuota per i punti di 'IntSup_ext'
        intsup_ext_values = []

        # Trova il valore di 'beff' all'intersezione iniziale del concio
        #if x_concio_start in dict_beff:
        intsup_ext_values.append(find_intersection_point(polyline, x_concio_start))

        # Trova i punti di 'IntSup_ext' corrispondenti ai punti x del concio
        for xi_beff, yi_beff in zip(x_beff, y_beff):
            if x_concio_start < xi_beff < x_concio_end:
                intsup_ext_values.append((xi_beff, yi_beff))  # Aggiungi il punto della beff

        # Trova il valore di 'beff' all'intersezione finale del concio
        #if x_concio_end in dict_beff:
        intsup_ext_values.append(find_intersection_point(polyline, x_concio_end))

        # Ordina la lista di punti per progressione sull'asse x
        #intsup_ext_values.sort()

        # Aggiungi la lista di punti 'IntSup_ext' alla tua dictionary originale
        dictConci[key]['SectionDefine']['variabile'][typeBeff][index] = intsup_ext_values

    return dictConci