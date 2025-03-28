import numpy as np
from cmath import pi, sqrt
import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from SGMPy_material import *
import plotly.graph_objs as go

# Funzione per verificare se una lista è annidata
def is_lista_annidata(lista):
    for elemento in lista:
        if isinstance(elemento, list):
            return True
    return False

# Funzione per ruotare un punto intorno a un punto di riferimento di un certo angolo in radianti
def rotate_point(point, angle, origin):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return qx, qy


def plotSection_matLib(dictSection, center = False, propTable = False):
    # Verifica se almeno un elemento è una lista
    typeList = is_lista_annidata(dictSection["sec_point"])
    
    fig = plt.figure()
    
        #print("La lista contiene almeno una lista.")
    if typeList == True:
        for i in dictSection["sec_point"]:
            listX, listY = [], []
            for j in i:
                listX.append(j[0])
                listY.append(j[1])
            listX.append(i[0][0])
            listY.append(i[0][1])
            plt.plot(listX, listY, '-')
    else:
        #print("La lista non contiene nessuna lista.")
        listX, listY = [], []
        for i in dictSection["sec_point"]:
            listX.append(i[0])
            listY.append(i[1])
        listX.append(dictSection["sec_point"][0][0])
        listY.append(dictSection["sec_point"][0][1])
    
        plt.plot(listX, listY, '-')
    
    if center == True:
        # Disegna il punto
        plt.scatter(*dictSection["Pg"], color='red', marker='+', s=500, label='Baricentro (+)')
        # Annotazione delle coordinate del punto
        #plt.text(dictSection["Pg"][0]- 50, dictSection["Pg"][1], f'({round(dictSection["Pg"][0],2)}, {round(dictSection["Pg"][1],2)})', color='black', fontsize=12, ha='right')

    if propTable == True:
        
        data_dict = {'A': round(dictSection['A'],2), 
                     'Iy': round(dictSection['Iy'],2), 
                     'Iz': round(dictSection['Iz'],2), 
                     #'It': round(dictSection['It'],2), 
                     'xg': round(dictSection['Pg'][0],2), 
                     'yg': round(dictSection['Pg'][1],2)}
    
    # Aggiungi tabella con i valori del dizionario
        #df = pd.DataFrame(data_dict.items(), columns=['Key', 'Value'])
        # Calcola la larghezza delle colonne in base alla lunghezza dei valori
        #col_widths = [max(len(str(value)) for value in df[col])*0.02 for col in df.columns]
        #plt.table(cellText=df.values, colLabels=df.columns, cellLoc='left', loc='lower left', fontsize=20, colWidths=col_widths)

        
    plt.ylabel('some numbers')
    plt.axis('equal')      # Imposta gli assi con la stessa scala
    plt.show()

    return fig


def plotSection_ploty(dictSection, center=False, propTable=False):
    fig = go.Figure()

    typeList = is_lista_annidata(dictSection["sec_point"])
    #print(dictSection["sec_point"][-1])
    if typeList:
        for i in dictSection["sec_point"]:
            
            listX, listY = zip(*i)
            listX = list(listX) + [i[0][0]]
            listY = list(listY) + [i[0][1]]
            fig.add_trace(go.Scatter(x=listX, y=listY, mode='lines'))
    else:
        listX, listY = zip(*dictSection["sec_point"])
        listX = list(listX) + [dictSection["sec_point"][0][0]]
        listY = list(listY) + [dictSection["sec_point"][0][1]]
        fig.add_trace(go.Scatter(x=listX, y=listY, mode='lines'))

    if center:
        fig.add_trace(go.Scatter(x=[dictSection["Pg"][0]], y=[dictSection["Pg"][1]], mode='markers', marker=dict(color='red', symbol='cross', size=10), name='Baricentro (+)'))
        fig.add_annotation(x=dictSection["Pg"][0] - 50, y=dictSection["Pg"][1], text=f'({round(dictSection["Pg"][0], 2)}, {round(dictSection["Pg"][1], 2)})', showarrow=False)

    if propTable:
        data_dict = {'A': round(dictSection['A'], 2),
                     'Iy': round(dictSection['Iy'], 2),
                     'Iz': round(dictSection['Iz'], 2),
                     'xg': round(dictSection['Pg'][0], 2),
                     'yg': round(dictSection['Pg'][1], 2)}

        df = pd.DataFrame(data_dict.items(), columns=['Key', 'Value'])
        col_widths = [max(len(str(value)) for value in df[col]) * 0.02 for col in df.columns]

        table_trace = go.Table(header=dict(values=df.columns, font=dict(size=20), align='left'),
                               cells=dict(values=[df[col] for col in df.columns], font=dict(size=20), align='left'),
                               columnwidth=col_widths)
        fig.add_trace(table_trace)

    fig.update_layout(yaxis_title='some numbers', showlegend=False, yaxis=dict(scaleanchor="x", scaleratio=1), autosize=True)

    #fig.show()

    return fig


def parametric_circle(t,xc,yc,R):
    x = xc + R*np.cos(t)
    y = yc + R*np.sin(t)
    return x,y

def point_along_line(x1, y1, x2, y2, Binst, Be1):
    # Calcoliamo la lunghezza del segmento tra i due punti
    segment_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    # Calcoliamo la proporzione di Be1 rispetto alla lunghezza del segmento
    proportion_be1 = Be1 / segment_length
    
    # Calcoliamo le coordinate del primo punto spostato
    x1_point = x1 + (x2 - x1) * proportion_be1
    y1_point = y1 + (y2 - y1) * proportion_be1

    # Calcoliamo la lunghezza orizzontale e verticale del segmento tra i due punti spostati
    delta_x = Binst / (1 + ((y2 - y1) / (x2 - x1))**2)**0.5
    delta_y = ((y2 - y1) / (x2 - x1)) * delta_x

    # Calcoliamo le coordinate del secondo punto desiderato
    x2_point = x1_point + delta_x
    y2_point = y1_point + delta_y

    return [[x1_point, y1_point], [x2_point, y2_point]]

def propCalc(coordinates):
    n = len(coordinates)
    # Calcola il momento d'inerzia rispetto all'asse x
    area = 0
    x_baricentro = 0
    y_baricentro = 0
    momento_inerzia_x = 0
    momento_inerzia_y = 0
    for i in range(n):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[(i + 1) % n]
        
        prodotto = (x1 * y2 - x2 * y1)
        
        # Calcola l'area
        area += prodotto
        #baricentro
        x_baricentro += (x1 + x2) * prodotto
        y_baricentro += (y1 + y2) * prodotto
        # Calcola il momento d'inerzia rispetto agli assi x e y
        momento_inerzia_x += (y1**2 + y1 * y2 + y2**2) * prodotto
        momento_inerzia_y += (x1**2 + x1 * x2 + x2**2) * prodotto
        
    area /= 2
    try:
        x_baricentro /= (6 * area)
        y_baricentro /= (6 * area)
    except: #serve per quando i punti hanno coordinata 0
        x_baricentro = 0.0 
        y_baricentro = 0.0

    
    # Calcola il momento d'inerzia rispetto all'asse zero
    momento_inerzia_x /= 12 
    momento_inerzia_y /= 12
    
    # Calcola il momento d'inerzia rispetto al baricentro
    # Calcola il contributo del momento d'inerzia rispetto al baricentro
    momento_inerzia_x -= area * y_baricentro**2
    momento_inerzia_y -= area * x_baricentro**2
    
    propDict = {'A': area,
                "Ay": None,
                "Az": None,
                'Iy': momento_inerzia_x, 
                'Iz': momento_inerzia_y, 
                'It': None, 
                'Pg': [x_baricentro, y_baricentro], 
                'ay': None, 
                'az': None, 
                'sec_point': coordinates}
    
    return propDict

def CircleSection(d, Pg, material = None):

    Area = d**2/4*pi

    alphaY = 9.0/10.0
    alphaZ = 9.0/10.0

    AreaY = Area * alphaY
    AreaZ = Area * alphaZ

    
    Iyy = d**4/64 * pi
    Izz = Iyy

    J = d**4/32 * pi

    wireframe = [ (parametric_circle(t, Pg[0], Pg[1], d/2)) for t in np.linspace(0.0, 2*np.pi, num=20)] 
    
    #print(wireframe)
    return {'A': Area, "Ay": None, "Az": None,'Iy': Iyy, 'Iz': Izz, 'It': J, 'Pg': Pg, 'ay': alphaY, 'az': alphaZ, 'sec_point': wireframe}


def HollowCircleSection(d, t, material = None):

    Area = (d**2 - ( d -2* t)**2)/4 * pi

    alphaY = 9.0/10.0
    alphaZ = 9.0/10.0

    AreaY = Area * alphaY
    AreaZ = Area * alphaZ

    
    Iyy = (d**4 - ( d-2* t)**4)/64 * pi
    Izz = Iyy

    J = (d**4 - (d - 2* t)**4)/32 * pi

    yg = 0.0

    wireframe = list([ [0.] +  list(parametric_circle(t, 0.0, 0.0, d/2)) for t in np.linspace(0.0, 2*np.pi, num=20)])

    return {'A': Area, 'Iy': Iyy, 'Iz': Izz, 'It': J, 'yg': yg, 'ay': alphaY, 'az': alphaZ, 'sec_point': wireframe}


def RectangularSection( b, h, Pg, material = None):

    p1 = (-b/2+Pg[0], -h-Pg[1])
    p2 = (b/2+Pg[0], -h-Pg[1])
    p3 = (b/2+Pg[0], -Pg[1])
    p4 = (-b/2+Pg[0], -Pg[1])

    wireframe = [p1, p2, p3, p4]
    
    Area = b * h
    xg = 0.0 + Pg[0]
    yg = -h/2 + Pg[1]

    alphaY = 5.0/6.0
    alphaZ = 5.0/6.0

    AreaY = Area * alphaY
    AreaZ = Area * alphaZ

    Iyy = (b* h**3)/12

    Izz = (b**3 *h)/12

    if h < b:
        k = 1 / (3+4.1*((h/b)**3/2))
        J = k*b*h**3
    else:
        k = 1 / (3+4.1*((b/h)**(3/2)))
        J = k*h*b**3

    return {'A': Area, 'Iy': Iyy, 'Iz': Izz, 'It': J, 'Pg': [xg, yg], 'ay': alphaY, 'az': alphaZ, 'sec_point': wireframe}

def HollowRectangularSection(width, height, t_sides, t_bottom, t_upper, material = None):
    
    p1 = (0., -width/2, -height/2)
    p2 = (0., width/2, -height/2)
    p3 = (0., width/2, height/2)
    p4 = (0., -width/2, height/2)

    wireframe = [p1, p2, p3, p4]

    Area = (width * height) - (width - 2 * t_sides )* (height - t_bottom - t_upper)

    # alphaY is a function because for some section it is not a constant factor
    alphaY = 9.0/10.0
    alphaZ = 9.0/10.0

    AreaY = Area * alphaY
    AreaZ = Area * alphaZ
    
    yg = None
    Iyy = None
    Izz = None

    
    if t <= (height/2 and width/2):
        h0 =2*((height-t) + (width-t))
        Ah = (height-t)*(width-t)
        k=2*Ah*t/h0
        J = (t**3*h0/3) + 2*k*Ah
        Cw= J/(t + k/t)   #torsional modulus, not used in RHS
    else:
        J = None; Cw = None


def renforcementBar(phi, Pg):
    
    listBar = []
    coordinates = []
    
    A = []
    xi = []
    yi = []
    Iy = []
    Iz = []
    for ig in Pg:
        singleBar = CircleSection(phi, [ig[0], -ig[1]])
        A.append(singleBar["A"])
        xi.append(singleBar["Pg"][0])
        yi.append(singleBar["Pg"][1])
        Iy.append(singleBar["Iy"])
        Iz.append(singleBar["Iz"])
        coordinates.append(singleBar['sec_point'])
        #plotSection(singleBar['sec_point'])
        
    Atot = sum(A)
    xg = sum(a*i for a, i in zip(A,xi))/Atot
    yg = sum(a*i for a, i in zip(A,yi))/Atot
    Iy = sum(j + a*(yg-i)**2 for a, i, j in zip(A, yi, Iy))
    Iz = sum(j + a*(xg-i)**2 for a, i, j in zip(A, xi, Iz))

    #plotSection(np.array(coordinates))
    
    propDict = {'A': Atot, 
                "Ay": None,
                "Az": None,
                'Iy': Iy , 
                'Iz': Iz,
                'It': None, 
                'Pg': [xg, yg], 
                'ay': None, 'az': None, 
                'sec_point': coordinates}
    
    return propDict

def rectangularCA(dictRCA, dictListBar):
    
    Acls = dictRCA["A"] - sum([idict["A"] for idict in dictListBar])
    xg = (dictRCA["A"]*dictRCA["Pg"][0] - sum([idict["A"]*idict["Pg"][0] for idict in dictListBar]))/Acls
    yg = (dictRCA["A"]*dictRCA["Pg"][1] - sum([idict["A"]*idict["Pg"][1] for idict in dictListBar]))/Acls
    Iy = (dictRCA["Iy"]+Acls*(yg-dictRCA["Pg"][1])**2 - sum([idict["Iy"]+idict["A"]*(yg-idict["Pg"][1])**2 for idict in dictListBar]))
    Iz = (dictRCA["Iz"]+Acls*(xg-dictRCA["Pg"][0])**2 - sum([idict["Iz"]+idict["A"]*(xg-idict["Pg"][0])**2 for idict in dictListBar]))
    ## MANCA ORA IL CONTRIBUTO DELLE BARRE DA AGGIUNGERE
    
    coordinates = [dictRCA['sec_point']] 
    for idict in dictListBar:
        for iBar in idict['sec_point']:
            coordinates.append(iBar)#+ [idict['sec_point'] for idict in dictListBar]
    
    propDict = {'A': Acls, 
                "Ay": None,
                "Az": None,
                'Iy': Iy , 
                'Iz': Iz,
                'It': None, 
                'Pg': [xg, yg], 
                'ay': None, 'az': None, 
                'sec_point': coordinates}
    
    return propDict

def V_rib(H, B, r, t, trasl, material=None, cl4Dict=None):
    # Calcoliamo l'altezza del centro del cerchio
    h = H - r
    #fig, ax = plt.subplots(1)
    
    #ax.set_aspect(1)

    # Troviamo l'angolo tangente all'estremità del cerchio
    for angle in np.linspace(0, np.pi/2, 1000):
        tangent_vector = np.array([-r * np.sin(angle), -r * np.cos(angle)])
        x1 = r * np.cos(angle) 
        x2 = -h - r * np.sin(angle)
        direction_A = np.array([x1-B/2, x2])
        direction_A /= np.linalg.norm(direction_A)
        tangent_vector /= np.linalg.norm(tangent_vector)
        
        dot_product_A = np.dot(tangent_vector, direction_A)
        if np.isclose(dot_product_A, 1, rtol=1e-06, atol=1e-08):
            tangent_angle = angle
            break
        #print(dot_product_A)
        
    #plt.show()

    angle = ((np.pi-2*tangent_angle))*(180/np.pi)
    # Generiamo i punti del cerchio esterno
    theta = np.linspace(np.pi-tangent_angle, tangent_angle, 1000)
    x1 = r * np.cos(theta) + trasl[0]
    x2 = -h - r * np.sin(theta)- trasl[1]

    # Creiamo il poligono esterno
    x1, x2 = np.insert(x1, 0, -B/2+trasl[0]), np.insert(x2, 0, -trasl[1])
    x1, x2 = np.append(x1, B/2+trasl[0]), np.append(x2, -trasl[1])

    # Parametri per il cerchio interno
    r2 = r - t

    # Generiamo i punti del cerchio interno
    x3 = r2 * np.cos(theta) + trasl[0]
    x4 = -h - r2 * np.sin(theta) - trasl[1]

    hw = np.abs(x4[0]) - np.abs(trasl[1])
    
    # Creiamo il poligono interno
    x3, x4 = np.insert(x3, 0, -B/2+t/np.cos(tangent_angle)+trasl[0]), np.insert(x4, 0, -trasl[1])
    x3, x4 = np.append(x3, B/2-t/np.cos(tangent_angle)+trasl[0]), np.append(x4, -trasl[1])
    x3, x4 = x3[::-1], x4[::-1]
    #x3, x4 = np.append(x3, -B/2), np.append(x4, 0)
    
     
    #print("hw", hw)

    # Uniamo i punti per ottenere il poligono finale
    X, Y = np.concatenate((x1, x3)), np.concatenate((x2, x4))

    # Combina le liste x1 ed x2 insieme a coordinates
    coordinates = [(xi, yi) for xi, yi in zip(X, Y)]

    # Calcola l'inerzia rispetto agli assi x e y
    # Calcola l'inerzia rispetto agli assi x e y utilizzando il teorema di Steiner
    elasticProp = propCalc(coordinates)
    #print("A", elasticProp["A"])
    #print("baricentro", elasticProp["Pg"][0], elasticProp["Pg"][1])
    #print("Momento d'inerzia rispetto all'asse x",elasticProp["Iy"])
    #print("Momento d'inerzia rispetto all'asse y", elasticProp["Iz"])
    

    if cl4Dict != None: ## CALCOLO PROPRIETA' pER CLASSE 4 DELLA SEZIONE
        ## LATO SX
        x1, y1 = -B/2+trasl[0], -trasl[1]
        x2, y2 = -r * np.cos(tangent_angle)+trasl[0], -h - r * np.sin(tangent_angle)-trasl[1]

        pointBinst = point_along_line(x1, y1, x2, y2, cl4Dict["Binst"], cl4Dict["Be1"])
        #ax.plot([x1, x2], [y1, y2])
        x1_net = [pointBinst[0][0], pointBinst[1][0], pointBinst[1][0] + t*np.cos(tangent_angle), pointBinst[0][0] + t*np.cos(tangent_angle)]
        y1_net = [pointBinst[0][1], pointBinst[1][1], pointBinst[1][1] + t*np.sin(tangent_angle), pointBinst[0][1] + t*np.sin(tangent_angle)]
        
        ## LATO DX
        x1, y1 = B/2+trasl[0], -trasl[1]
        x2, y2 = r * np.cos(tangent_angle)+trasl[0], -h - r * np.sin(tangent_angle)-trasl[1]

        pointBinst = point_along_line(x1, y1, x2, y2, -cl4Dict["Binst"], cl4Dict["Be1"])
        #ax.plot([x1, x2], [y1, y2])
        x2_net = [pointBinst[0][0], pointBinst[1][0], pointBinst[1][0] - t*np.cos(tangent_angle), pointBinst[0][0] - t*np.cos(tangent_angle)]
        y2_net = [pointBinst[0][1], pointBinst[1][1], pointBinst[1][1] + t*np.sin(tangent_angle), pointBinst[0][1] + t*np.sin(tangent_angle)]
        
        
        # Combina le liste x1 ed x2 insieme a coordinates
        coordinatesInst1 = [(xi, yi) for xi, yi in zip(x1_net, y1_net)]
        coordinatesInst2 = [(xi, yi) for xi, yi in zip(x2_net, y2_net)]
        elasticPropInst = propCalc(coordinatesInst1)
        #print("A", elasticPropInst["A"])
        #print("baricentro", elasticPropInst["Pg"][0], elasticPropInst["Pg"][1])
        #print("Momento d'inerzia rispetto all'asse x",elasticPropInst["Iy"])
        #print("Momento d'inerzia rispetto all'asse y", elasticPropInst["Iz"])
        
        Atot = elasticProp["A"] - elasticPropInst["A"]*2
        yg = (elasticProp["A"]*elasticProp["Pg"][1] - elasticPropInst["A"]*2*elasticPropInst["Pg"][1])/Atot
        Iy = elasticProp["Iy"] + elasticProp["A"]*(yg-elasticProp["Pg"][1])**2 - elasticPropInst["Iy"]*2-elasticPropInst["A"]*2*(yg-elasticPropInst["Pg"][1])**2
        Iz = elasticProp["Iz"] - elasticPropInst["Iz"]*2-elasticPropInst["A"]*2*(0-elasticPropInst["Pg"][0])**2
        propDict = {'A': Atot, 
                    "Ay": 0,
                    "Az": None,
                    'Iy': Iy , 
                    'Iz': Iz,
                    'It': None, 
                    'Pg': [0, yg], 
                    'ay': None, 'az': None, 
                    'sec_point': [coordinates, coordinatesInst1, coordinatesInst2]}
    else:
        propDict = elasticProp
    
    #print("A", propDict["A"])

    return propDict   

def L__rib(a, b, e, trasl, angle = 0, mirror = False,  material=None, cl4Dict=None):

    # Combina le liste x1 ed x2 insieme a coordinates
    if mirror is False:
        # poligono
        X = [trasl[0], e + trasl[0], e + trasl[0], b + trasl[0], b + trasl[0], trasl[0]]
        Y = [-trasl[1]-a, -trasl[1]-a, -trasl[1]-e, -trasl[1]-e, -trasl[1], -trasl[1]]
        coordinates = [(xi, yi) for xi, yi in zip(X, Y)]
    else:
        # poligono
        X = [trasl[0], -e + trasl[0], -e + trasl[0], -b + trasl[0], -b + trasl[0], trasl[0]]
        Y = [-trasl[1]-a, -trasl[1]-a, -trasl[1]-e, -trasl[1]-e, -trasl[1], -trasl[1]]
        coordinates = [(xi, yi) for xi, yi in zip(X, Y)]
    
    if angle != 0:
        # Angolo di rotazione in radianti
        angle_radians = np.deg2rad(angle) 
        # Ruotare ciascun punto del poligono
        coordinates = [rotate_point(point, angle_radians, coordinates[0]) for point in coordinates]

                
    
    # Calcola l'inerzia rispetto agli assi x e y
    # Calcola l'inerzia rispetto agli assi x e y utilizzando il teorema di Steiner
    elasticProp = propCalc(coordinates)
    
    return elasticProp 

def T__rib(a, b, ea, eb, trasl, material="S235", cl4Dict=None):
    
    # poligono
    X = [trasl[0]+ea/2, ea/2 + trasl[0], b/2 + trasl[0], b/2 + trasl[0], -b/2 + trasl[0], -b/2 + trasl[0], -ea/2 + trasl[0], -ea/2 + trasl[0]]
    Y = [-trasl[1]-a, -trasl[1]-eb, -trasl[1]-eb, -trasl[1], -trasl[1], -trasl[1]-eb, -trasl[1]-eb, -trasl[1]-a]
    # Combina le liste x1 ed x2 insieme a coordinates
    coordinates = [(xi, yi) for xi, yi in zip(X, Y)]
    
    # Calcola l'inerzia rispetto agli assi x e y
    # Calcola l'inerzia rispetto agli assi x e y utilizzando il teorema di Steiner
    elasticProp = propCalc(coordinates)
    
    return elasticProp 

def OrizontalPlate(B, t, trasl, material="S235", cl4Dict=None):

    # poligono
    X = [-B/2 + trasl[0], -B/2+ trasl[0], B/2+ trasl[0], B/2+ trasl[0]]
    Y = [-trasl[1], -trasl[1]-t, -trasl[1]-t, -trasl[1]]

    # Combina le liste x1 ed x2 insieme a coordinates
    coordinates = [(xi, yi) for xi, yi in zip(X, Y)]

    # Calcola l'inerzia rispetto agli assi x e y
    # Calcola l'inerzia rispetto agli assi x e y utilizzando il teorema di Steiner
    elasticProp = propCalc(coordinates)
    elasticProp["mat"] = steel_ntc18(material, t, gamma_s = 1.15)
    #print("A", elasticProp["A"])
    #print("baricentro", elasticProp["Pg"][0], elasticProp["Pg"][1])
    #print("Momento d'inerzia rispetto all'asse x",elasticProp["Iy"])
    #print("Momento d'inerzia rispetto all'asse y", elasticProp["Iz"])
    

    """     
    if cl4Dict != None: ## CALCOLO PROPRIETA' pER CLASSE 4 DELLA SEZIONE
        ##WIP
    else:
        propDict = elasticProp """

    return elasticProp 

def WebPlate(H, t, trasl, alpha, material="S235", cl4Dict=None): ##PIATTO D'ANIMA

    # poligono
    #dx = H*np.tan(np.radians(alpha))
    X = [-t/2+trasl[0], -t/2+trasl[0], t/2+trasl[0], t/2+trasl[0]]
    Y = [- trasl[1], - trasl[1]-H, -trasl[1]-H, -trasl[1]]
    coordinates = [(xi, yi) for xi, yi in zip(X, Y)]
    
    if alpha != 0:
        # Angolo di rotazione in radianti
        angle_radians = np.deg2rad(alpha) 
        # Ruotare ciascun punto del poligono
        coordinates = [rotate_point(point, angle_radians, coordinates[0]) for point in coordinates]
        

    # Calcola l'inerzia rispetto agli assi x e y
    # Calcola l'inerzia rispetto agli assi x e y utilizzando il teorema di Steiner
    elasticProp = propCalc(coordinates)
    elasticProp["mat"] = steel_ntc18(material, t, gamma_s = 1.15)
    #print("A", elasticProp["A"])
    #print("baricentro", elasticProp["Pg"][0], elasticProp["Pg"][1])
    #print("Momento d'inerzia rispetto all'asse x",elasticProp["Iy"])
    #print("Momento d'inerzia rispetto all'asse y", elasticProp["Iz"])
    

    if cl4Dict != None: ## CALCOLO PROPRIETA' per CLASSE 4 DELLA SEZIONE
        ##WIP
        #print("ciao")
        #st.write(cl4Dict)
        if cl4Dict["bc"]<0:
            h1 = - trasl[1]-cl4Dict["be1"] 
            h2 = - trasl[1] -cl4Dict["be1"] - cl4Dict["delta"]
        elif cl4Dict["bc"]>0:
            h1 = - trasl[1] -H + cl4Dict["be1"] + cl4Dict["delta"]
            h2 = - trasl[1] -H + cl4Dict["be1"]

        X = [-t/2+trasl[0], -t/2+trasl[0], t/2+trasl[0], t/2+trasl[0]]
        Y = [h1, h2, h2, h1]
        coordinates = [(xi, yi) for xi, yi in zip(X, Y)]
        if alpha != 0:
            # Angolo di rotazione in radianti
            angle_radians = np.deg2rad(alpha) 
            # Ruotare ciascun punto del poligono
            coordinates = [rotate_point(point, angle_radians, coordinates[0]) for point in coordinates]
        
        elasticPropInst = propCalc(coordinates)
        #elasticProp["mat"] = steel_ntc18(material, t, gamma_s = 1.15)
        Atot = elasticProp["A"] - elasticPropInst["A"]
        yg = (elasticProp["A"]*elasticProp["Pg"][1] - elasticPropInst["A"]*elasticPropInst["Pg"][1])/Atot
        Iy = elasticProp["Iy"] + elasticProp["A"]*(yg-elasticProp["Pg"][1])**2 - elasticPropInst["Iy"]-elasticPropInst["A"]*(yg-elasticPropInst["Pg"][1])**2
        Iz = elasticProp["Iz"] - elasticPropInst["Iz"]-elasticPropInst["A"]*(0-elasticPropInst["Pg"][0])**2
        
        elasticProp["A"] = Atot
        elasticProp["Pg"][1] = yg
        elasticProp["Iy"] = Iy
        elasticProp["Iz"] = Iz
        #st.write(elasticProp)
        
        elasticProp["sec_point"]= (elasticProp["sec_point"], coordinates)
        #st.write(elasticProp)

    return elasticProp 

def Classe1Anima(d, t, fyk, yn):
    
    eps = np.sqrt(235/fyk)
    a = d/t
    alpha = yn
    
    classe_1 = {}
    # solo compressione
    # flessione
    classe_1["flessione"] = [a, 72*eps, True] if a <= 72*eps else False
    # compressione 
    classe_1["compressione"] = [a, 33*eps, True] if a <= 33*eps else False 
    #flessione e compressione
    alpha_tensionSup = yn  
    if alpha_tensionSup > 0.5:
        classe_1["flessione e compressione"] = [a, 396*eps/(13*alpha_tensionSup-1), True] if a <= 396*eps/(13*alpha_tensionSup-1) else False 
    if alpha_tensionSup <= 0.5:
        classe_1["flessione e compressione"] = [a, 36*eps/(alpha_tensionSup), True] if a <= 36*eps/(alpha_tensionSup) else False 

    return classe_1

def Classe2Anima(d, t, fyk, yn):
    
    eps = np.sqrt(235/fyk)
    a = d/t
    alpha = yn
    
    ## CALCOLO CLASSE SEZIONALE - saldata
    classe_2 = {}

    # flessione
    classe_2["flessione"] = [a, 83*eps,True] if 72*eps < a <= 83*eps else False
    # compressione 
    classe_2["compressione"] = [a, 38*eps,True] if 33*eps < a <= 38*eps else False
    #flessione e compressione
    alpha_tensionSup = yn 
    if alpha_tensionSup > 0.5:
        classe_2["flessione e compressione"] = [a, 456*eps/(13*alpha_tensionSup-1),True] if 396*eps/(13*alpha_tensionSup-1) < a <= 456*eps/(13*alpha_tensionSup-1) else False
    if alpha_tensionSup < 0.5:
        classe_2["flessione e compressione"] =  [a, 41.5*eps/(alpha_tensionSup),True] if 36*eps/(alpha_tensionSup) < a <= 41.5*eps/(alpha_tensionSup) else False
    
    return classe_2

def Classe3Anima(d, t, fyk, yn, sigma1, sigma2):
    
    eps = np.sqrt(235/fyk)
    a = d/t
    alpha = yn
    
    ## CALCOLO CLASSE SEZIONALE - saldata
    classe_3 = {}

    # flessione
    classe_3["flessione"] = [a, 124*eps,True] if 83*eps < a <= 124*eps else False
    # compressione 
    classe_3["compressione"] = [a, 42*eps,True] if 38*eps < a <= 42*eps else False
    #flessione e compressione
    alpha_tensionSup = alpha 
    psi = sigma1/sigma2
    #print(sigma1, sigma2, "psi", psi)
    
    if psi > -1:
        classe_3["flessione e compressione"] = [a, 42*eps/(0.67+0.33*psi),True] if 456*eps/(13*alpha_tensionSup-1) < a <= 42*eps/(0.67+0.33*psi) else False
    if psi <= -1:
        classe_3["flessione e compressione"] =  [a, (62*eps/(1 - psi))*np.sqrt(-psi),True] if 41.5*eps/(alpha_tensionSup) < a <= (62*eps/(1 - psi))*np.sqrt(-psi) else False
    
    return classe_3

def Classe4Anima(d, t, fyk, yn, sigma1, sigma2):
    eps = np.sqrt(235/fyk)
    a = d/t
    alpha = abs(d*sigma1/(abs(sigma1)+abs(sigma2)))
    #alpha = abs(yn)
    #st.write(alpha)
    ## CALCOLO CLASSE SEZIONALE - saldata
    
    # classe anima
    # CALCOLO PSI
    psi = round(sigma2/sigma1,3)
    #print(sigma1, sigma2, "psi", psi)
    # CALCOLO COEFFICIENTE DI IMBOZZAMENTO
    ksigma = 4.0 if  psi == 1.00 else 8.2/(1.05+psi) if 1> psi >0 else 7.81 if psi == 0 else 7.81-6.29*psi + 9.78*psi**2 if 0> psi >-1 else 23.9 if psi == -1 else 5.98*(1-psi)**2 if -1> psi >-3 else print("WARNING: risulta fuori dalle condizioni impostate") 
    # CALCOLO SNELLEZZA DEL PANNELLO
    lamP = (a)/(28.4*eps*np.sqrt(ksigma))
    # CALCOLO FATTORE DI RIDUZIONE PER PANNELLI IRRIGIDITI DA ENTRAMBI I LATI
    rid = 1 if lamP <= 0.673 else (lamP-0.055*(3+psi))/lamP**2 if lamP > 0.673 else print("WARNING: risulta fuori dalle condizioni impostate")
    # CALCOLO LARGHEZZA EFFICACE
    if psi == 1:
        beff = rid*d
        be1 = 0.5*beff
        be2 = 0.5*beff
        delta = d - beff
        bc = None
        bt = None
    
    elif 1> psi >= 0:
        beff = rid*d
        be1 = 2*beff/(5-psi)
        be2 = beff - be1
        delta = d - beff
        bc = None
        bt = None
        
    elif psi < 0:
        beff = rid*d/(1-psi)
        be1 = 0.4*beff
        be2 = 0.6*beff
        bc = d*sigma1/(abs(sigma1)+abs(sigma2))
        bt = d-abs(bc)
        delta = abs(bc) - beff
    
    #flessione e compressione
    #alpha_tensionSup = alpha 
    
    classe_4 = {"be1": be1, "be2": be2, "beff": beff, "bc": bc, "bt": bt, "rho": rid, "delta": delta}

    return classe_4

def ClasseAnima(d, t, fyk, yn, sigma1, sigma2): #PANNELLO IRRIGIDITO DA ENTRAMBI I LATI
    classe = {'flessione': None, 'compressione': None, 'flessione e compressione': None}
    
    c1 = Classe1Anima(d, t, fyk, yn)
    c2 = Classe2Anima(d, t, fyk, yn)
    c3 = Classe3Anima(d, t, fyk, yn, sigma1, sigma2)
    c4 = Classe4Anima(d, t, fyk, yn, sigma1, sigma2)
    
    # FLESSIONE
    if c1["flessione"] != False and c2["flessione"] == False and c3["flessione"] == False:
        classe["flessione"] = 1
    elif c2["flessione"] != False and c1["flessione"] == False and c3["flessione"] == False:
        classe["flessione"] = 2
    elif c3["flessione"] != False and c1["flessione"] == False and c2["flessione"] == False:
        classe["flessione"] = 3
    else:
        classe["flessione"] = 4
        
    # COMPRESSIONE
    if c1["compressione"] != False and c2["compressione"] == False and c3["compressione"] == False:
        classe["compressione"] = 1
    elif c2["compressione"] != False and c1["compressione"] == False and c3["compressione"] == False:
        classe["compressione"] = 2
    elif c3["compressione"] != False and c1["compressione"] == False and c2["compressione"] == False:
        classe["compressione"] = 3
    else:
        classe["compressione"] = 4
        
    # FLESSIONE E COMPRESSIONE    
    if c1["flessione e compressione"] != False and c2["flessione e compressione"] == False and c3["flessione e compressione"] == False:
        classe["flessione e compressione"] = 1 
    elif c2["flessione e compressione"] != False and c1["flessione e compressione"] == False and c3["flessione e compressione"] == False:
        classe["flessione e compressione"] = 2
    elif c3["flessione e compressione"] != False and c1["flessione e compressione"] == False and c2["flessione e compressione"] == False:
        classe["flessione e compressione"] = 3
    else:
        classe["flessione e compressione"] = 4
    
    classeDict = {"result": classe,
                  "detail": {1: c1, 2: c2, 3: c3, 4: c4}}
    
    return classeDict

def Classe1Piattabanda(d, t, fyk):
    
    eps = np.sqrt(235/fyk)
    a = d/t
    
    classe_1 = {}
    # solo compressione
    # compressione 
    classe_1["compressione"] = [a, 9*eps, True] if a <= 9*eps else False 
    #flessione e compressione

    return classe_1

def Classe2Piattabanda(d, t, fyk):
    
    eps = np.sqrt(235/fyk)
    a = d/t
    
    ## CALCOLO CLASSE SEZIONALE - saldata
    classe_2 = {}

    # compressione 
    classe_2["compressione"] = [a, 10*eps,True] if 9*eps < a <= 10*eps else False

    return classe_2

def Classe3Piattabanda(d, t, fyk):
    eps = np.sqrt(235/fyk)
    a = d/t
    
    ## CALCOLO CLASSE SEZIONALE - saldata
    classe_3 = {}

    # compressione 
    classe_3["compressione"] = [a, 14*eps,True] if 10*eps < a <= 14*eps else False
    #flessione e compressione

    return classe_3

def Classe4Piattabanda(d, t, fyk):
    eps = np.sqrt(235/fyk)
    a = d/t
    ## CALCOLO CLASSE SEZIONALE - saldata
    
    # classe anima
    # CALCOLO PSI
    psi = 1
    #print(sigma1, sigma2, "psi", psi)
    # CALCOLO COEFFICIENTE DI IMBOZZAMENTO
    ksigma = 0.43 if  psi == 1.00 else 0.57 if psi == 0 else 0.85 if psi == -1 else 0.57-0.21*psi + 0.07*psi**2 if 1>= psi >=-3 else print("WARNING: risulta fuori dalle condizioni impostate") 
    # CALCOLO SNELLEZZA DEL PANNELLO
    lamP = (a)/(28.4*eps*np.sqrt(ksigma))
    # CALCOLO FATTORE DI RIDUZIONE PER PANNELLI IRRIGIDITI DA ENTRAMBI I LATI
    rid = 1 if lamP <= 0.748 else (lamP-0.188)/lamP**2 if lamP > 0.748 else print("WARNING: risulta fuori dalle condizioni impostate")
    # CALCOLO LARGHEZZA EFFICACE
    if 1>= psi >= 0:
        beff = rid*d
        
    elif psi < 0:
        bc = d/(1-psi)
        beff = rid*bc
    
    classe_4 = [beff]

    return classe_4

def ClassePiattabanda(d, t, fyk): #PANNELLO IRRIGIDITO DA ENTRAMBI I LATI
    classe = {'flessione': None, 'compressione': None, 'flessione e compressione': None}
    
    c1 = Classe1Piattabanda(d, t, fyk)
    c2 = Classe2Piattabanda(d, t, fyk)
    c3 = Classe3Piattabanda(d, t, fyk)
    c4 = Classe4Piattabanda(d, t, fyk)
    
        
    # COMPRESSIONE
    if c1["compressione"] != False and c2["compressione"] == False and c3["compressione"] == False:
        classe["compressione"] = 1
    elif c2["compressione"] != False and c1["compressione"] == False and c3["compressione"] == False:
        classe["compressione"] = 2
    elif c3["compressione"] != False and c1["compressione"] == False and c2["compressione"] == False:
        classe["compressione"] = 3
    else:
        classe["compressione"] = 4
    
    classeDict = {"result": classe,
                  "detail": {1: c1, 2: c2, 3: c3, 4: c4}}
    
    return classeDict
        
## SEZIONE PICCOLA  
#orPlateSup = OrizontalPlate(2500, 16, [0, 0])  
#vribs = V_rib(283, 300, 25, 6, [0, 16])
#wPlate = WebPlate(1300-16-12, 15, [0, 16], -np.radians(16), material=None, cl4Dict=None)
#orPlateInf = OrizontalPlate(1200, 12, [-(1300-16-12)*np.tan(np.radians(16)), (1300-12)])


def OrthotropicSection1(plateSup, PSribs, intRibs, web, webRibs, WintRibs, plateInf, PIribs, PIintRibs):
    #Section of Nassyria Bridge (Ponte su fiume Adda)
    
    listPoint = [plateSup['sec_point']]
    
    for i in intRibs:
        if any(isinstance(elemento, list) for elemento in PSribs['sec_point']):
            listPoint.append(PSribs['sec_point'][0]+np.array([i, 0])) # per le classi 4 da sottrarre
        else:
            listPoint.append(PSribs['sec_point']+np.array([i, 0])) # per le classi 4 da sottrarre
    
    if WintRibs != None:
        for i in WintRibs:
            if any(isinstance(elemento, list) for elemento in webRibs['sec_point']):
                listPoint.append(webRibs['sec_point'][0]+np.array([0, i])) # per le classi 4 da sottrarre
            else:
                listPoint.append(webRibs['sec_point']+np.array([0, i])) # per le classi 4 da sottrarre
        nribs_web = len(WintRibs)
    else:
        nribs_web = 0
        webRibs = {'A': 0, 'Iy': 0 , 'Iz': 0,'Pg': [0, 0]} 
    
    for i in PIintRibs:
        if any(isinstance(elemento, list) for elemento in PIribs['sec_point']):
            listPoint.append(PIribs['sec_point'][0]+np.array([i, 0])) # per le classi 4 da sottrarre
        else:
            listPoint.append(PIribs['sec_point']+np.array([i, 0])) # per le classi 4 da sottrarre
    listPoint.append(plateInf['sec_point'])
    
    if web == None:
        web = {'A': 0, 'Iy': 0 , 'Iz': 0,'Pg': [0, 0]} 
    else:
        listPoint.append(web['sec_point'])
        
    
    # CALCOLO PROPRIETA' DELLA SEZIONE
    nribs = len(intRibs)
    nribs_pi = len(PIintRibs)
    Aw = web["A"] + PSribs["A"]*nribs #area taglio verticale
    Atot = plateSup["A"] + PSribs["A"]*nribs + web["A"] + webRibs["A"]*nribs_web + plateInf["A"] + PIribs["A"]*nribs_pi
    #print(plateSup["A"],PSribs["A"]*nribs,web["A"], plateInf["A"])
    yg = (plateSup["A"]*plateSup["Pg"][1] + PSribs["A"]*nribs*PSribs["Pg"][1] + 
          web["A"]*web["Pg"][1] + webRibs["A"]*nribs_web*webRibs["Pg"][1] + 
          plateInf["A"]*plateInf["Pg"][1]+ PIribs["A"]*nribs_pi*PIribs["Pg"][1])/Atot
    
    xg = (plateSup["A"]*plateSup["Pg"][0] + sum(PSribs["A"]*(PSribs["Pg"][0]+np.array(intRibs)))+ 
          web["A"]*web["Pg"][0] + sum(PSribs["A"]*(PSribs["Pg"][0]+np.array(intRibs))) +
          plateInf["A"]*plateInf["Pg"][0])/Atot
    
    Ay = plateSup["A"] + plateInf["A"]
    
    Iy = 0
    Iy += plateSup["Iy"] + plateSup["A"]*(yg-plateSup["Pg"][1])**2 
    Iy += PSribs["Iy"]*nribs + PSribs["A"]*nribs*(yg-PSribs["Pg"][1])**2 
    Iy += web["Iy"] + web["A"]*(yg-web["Pg"][1])**2 
    Iy += webRibs["Iy"]*nribs_web + webRibs["A"]*nribs_web*(yg-webRibs["Pg"][1])**2 
    Iy += plateInf["Iy"] + plateInf["A"]*(yg-plateInf["Pg"][1])**2 
    Iy += PIribs["Iy"]*nribs_pi + PIribs["A"]*nribs_pi*(yg-PIribs["Pg"][1])**2 
    
    Iz = 0
    Iz += plateSup["Iz"] + plateSup["A"]*(xg-plateSup["Pg"][0])**2 
    Iz += PSribs["Iz"]*nribs + sum(PSribs["A"]*(xg+np.array(intRibs))**2 )
    Iz += web["Iz"] + web["A"]*(xg-web["Pg"][0])**2 
    Iz += plateInf["Iz"] + plateInf["A"]*(xg-plateInf["Pg"][0])**2 
    
    propDict = {'A': Atot, 
                "Ay": Ay,
                "Az": Aw,
                'Iy': Iy , 
                'Iz': Iz,
                'It': None, 
                'Pg': [xg, yg], 
                'ay': None, 
                'az': Aw/Atot, 
                'sec_point': listPoint}
    ## PLOT
    #plotSection(listPoint)
    
    return propDict

def builtSection(listDict):
    #Section of Nassyria Bridge (Ponte su fiume Adda)
    
    listPoint = []
    
    for i in listDict:
        if any(isinstance(elemento, list) for elemento in i['sec_point']):
            for j in i['sec_point']:
                listPoint.append(j) # per le classi 4 da sottrarre
        else:
            listPoint.append(i['sec_point'])
        
    
    # CALCOLO PROPRIETA' DELLA SEZIONE
    #Az = np.sum([i["Az"] for i in listDict]) #area taglio verticale
    #Ay = np.sum([i["Ay"] for i in listDict]) #area taglio verticale
    Atot = np.sum([i["A"] for i in listDict])
    #print(plateSup["A"],PSribs["A"]*nribs,web["A"], plateInf["A"])
    yg = np.sum([i["A"]*i["Pg"][1] for i in listDict])/Atot
    xg = np.sum([i["A"]*i["Pg"][0] for i in listDict])/Atot
    
    Iy = np.sum([i["Iy"]+i["A"]*(yg-i["Pg"][1])**2 for i in listDict])
    Iz = np.sum([i["Iz"]+i["A"]*(xg-i["Pg"][0])**2 for i in listDict])
    
    propDict = {'A': Atot, 
                "Ay": 0,
                "Az": 0,
                'Iy': Iy , 
                'Iz': Iz,
                'It': None, 
                'Pg': [xg, yg], 
                'ay': 0, 
                'az': 0/Atot, 
                'sec_point': listPoint,
                "dictPlate": listDict}
    ## PLOT
    #plotSection(listPoint)
    
    return propDict

def CompositeSection(SteelSection, ClsSection, dictListBar, n=6):
    
    ## CALCESTRUZZO AL NETTO DELLE BARRE DI ARMATURA
    Acls = ClsSection["A"] #- sum([idict["A"] for idict in dictListBar])
    xg_cls = (ClsSection["A"]*ClsSection["Pg"][0] - sum([idict["A"]*idict["Pg"][0] for idict in dictListBar]))/Acls
    yg_cls = (ClsSection["A"]*ClsSection["Pg"][1] - sum([idict["A"]*idict["Pg"][1] for idict in dictListBar]))/Acls
    Iy_cls = (ClsSection["Iy"]+Acls*(yg_cls-ClsSection["Pg"][1])**2 - sum([idict["Iy"]+idict["A"]*(yg_cls-idict["Pg"][1])**2 for idict in dictListBar]))
    Iz_cls = (ClsSection["Iz"]+Acls*(xg_cls-ClsSection["Pg"][0])**2 - sum([idict["Iz"]+idict["A"]*(xg_cls-idict["Pg"][0])**2 for idict in dictListBar]))
    
    A_bar = sum([idict["A"] for idict in dictListBar])
    ## PROPRIETA DELLA SEZIONE COMPOSTA
    Atot = SteelSection["A"] + (Acls-A_bar)/n + sum([idict["A"] for idict in dictListBar])
    yg = (SteelSection["A"]*SteelSection["Pg"][1] + (Acls-A_bar)*yg_cls/n + sum([idict["A"]*idict["Pg"][1] for idict in dictListBar]))/Atot
    xg = (SteelSection["A"]*SteelSection["Pg"][0] + (Acls-A_bar)*xg_cls/n + sum([idict["A"]*idict["Pg"][0] for idict in dictListBar]))/Atot
    
    
    Iy = (Iy_cls+Acls*(yg-yg_cls)**2)/n + sum([idict["Iy"]+idict["A"]*(yg-idict["Pg"][1])**2 for idict in dictListBar])
    Iz = (Iz_cls+Acls*(xg-xg_cls)**2)/n + sum([idict["Iz"]+idict["A"]*(xg-idict["Pg"][0])**2 for idict in dictListBar])
    
    Iy += SteelSection["Iy"] + SteelSection["A"]*(yg-SteelSection["Pg"][1])**2 
    #Iy += ClsSection["Iy"]/n + ClsSection["A"]*(yg-ClsSection["Pg"][1])**2/n 
    
    Iz += SteelSection["Iz"] + SteelSection["A"]*(xg-SteelSection["Pg"][0])**2 
    #Iz += ClsSection["Iz"]/n + ClsSection["A"]*(xg-ClsSection["Pg"][0])**2/n 
    
    coordinates = []
    # for i in ClsSection['sec_point']:
    #     coordinates.append(i)
        
    for idict in dictListBar:
        for iBar in idict['sec_point']:
            coordinates.append(iBar)#+ [idict['sec_point'] for idict in dictListBar]
    
    coordinates = coordinates + [ClsSection['sec_point']] + SteelSection['sec_point']#+ [idict['sec_point'] for idict in dictListBar]
    
    ay = (ClsSection["A"]/n + SteelSection["Ay"])/Atot
    Ay = ClsSection["A"]/n + SteelSection["Ay"]
    az = SteelSection["Az"]/Atot
            
    propDict = {'A': Atot, 
                "Ay": Ay, 
                "Az": SteelSection["Az"], 
                'Iy': Iy , 
                'Iz': Iz,
                'It': None, 
                'Pg': [xg, yg], 
                'ay': None, 'az': az, 
                'sec_point': coordinates}
    
    
    return propDict
