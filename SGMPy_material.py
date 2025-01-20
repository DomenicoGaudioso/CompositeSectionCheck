import numpy as np
from cmath import pi, sqrt
#import matplotlib.pyplot as plt



def steel_ntc18(nome, t, gamma_s = 1.15):
    """  
    Args:
        nome (_str_): _nome del tipo di acciaio, esempio: S235 oppure S275 N/NL_
        t (_float_): _spessore in mm_
        
    è stata sviluppata solo la EN 10025-2
    da sviluppare ancora gli acciai di:
    EN10025-3
    EN10025-4
    EN10025-5
    
    Returns:
        _type_: _description_
    """

    gamma_s = 78.5 #KN/m3
    Es = 210000 # modulo elastico (Youn's) [MPa]
    v = 0.3 #poisson
    Gs = Es/(2*(1+v)) # modulo di taglio
    alpha = 12*10^(-6) #coefficiente di espansione lineare [°C-1]
    
    
    
    # EN 10025 - 2
    if nome == "S235":
        if t <= 40:
            fyk, ftk = 235, 360
        elif 40 <= t <= 80 :
            fyk, ftk = 215, 360
        elif t > 80:
            fyk, ftk = None, None

    if nome == "S275":
        if t <= 40:
            fyk, ftk = 275, 430
        elif 40 <= t <= 80 :
            fyk, ftk = 255, 410
        elif t > 80:
            fyk, ftk = None, None    

    if nome == "S355":
        if t <= 40:
            fyk, ftk = 355, 510
        elif 40 <= t <= 80 :
            fyk, ftk = 335, 470
        elif t > 80:
            fyk, ftk = None, None                 

    if nome == "S450":
        if t <= 40:
            fyk, ftk = 440, 550
        elif 40 <= t <= 80 :
            fyk, ftk = 410, 550
        elif t > 80:
            fyk, ftk = None, None             
            
            
    fyd = fyk/gamma_s #resistenza caratteristica di progetto
    fyd_shear = fyd/(np.sqrt(3)) #resistenza a taglio di progetto
    
    #deformazione
    epsilon_sy = fyk/Es #deformazione allo snervamento
    epsilon_su = 4/1000 #deformazione ultima
    epsilon = np.sqrt(235/fyk)
    
    outDict = {"type": "steel",
            "nome": nome,
            "fyk" : fyk,
            "fyd" : fyd,
            "fyd_shear": fyd_shear,
            "gamma_s": gamma_s,
            "Es" : Es,
            "v" : v,
            "Gs": Gs,
            "epsilon": epsilon,
            }
    
    return outDict

def steelBar_ntc18(fyk):
    #input (fyk): resistenza caratteristica cubica in MPa
    gamma_s = 78.5
    fyd = fyk/1.15 #resistenza caratteristica di progetto
    Es = 210000
    v = 0.3 #poisson
    
    outDict = {"type": "steel bar",
               "gamma_s": gamma_s,
               "fyk" : fyk,
               "fyd" : fyd,
               "Es" : Es,
               "v" : v}

    return outDict

def cls_ntc18(Rck):
    #input (Rck): resistenza caratteristica cubica in MPa
    #non considera la riduzione di resistenza di progetto nel caso di spessore < 50 mm

    #peso
    gamma_c = 25 #KN/m3

    #11.2.10.1 Resistenza a compressione
    fck = 0.83*Rck #resistenza cilindrica caratteristica [11.2.1]
    fcm = fck + 8 #resistenza cilindrica media [11.2.2]
    fcd = 0.85*fck/1.5 #resistenza di progetto [4.1.3]

    #11.2.10.1 Resistenza a trazione
    fctm = 0.30*fck**(2/3) if Rck <= 60 else 2.12*np.log( 1 + fcm/10)  #resistenza media a trazione semplice
    fcfm = 1.2*fctm #resistenza media a flessione
    fctd = fcfm/1.5 #resistenza a trazione di progetto

    #Modulo elastico
    Ecm = 22000*(fcm/10)**(0.3)

    #poisson
    v = 0.2 #non fessurato
    vf = 0.0 #fessurato

    #Modulo di taglio
    Gcm = Ecm/(2*(1+v))
    Gcf = Ecm/(2*(1+vf))

    #dilatazione termica
    alpha = 10*10**-6 #°C^-1

    #deformazioni
    eps_c2 = 0.20/100 if Rck <= 60 else 0.20/100 + (0.0085/100) *(fck -50)**0.53
    eps_c3 = 0.175/100 if Rck <= 60 else 0.175/100 + (0.055/100) *((fck -50)/40)
    eps_cu = 0.35/100 if Rck <= 60 else 0.26/100 + (3.5/100) *((90 -fck)/100)**4
    eps_c4 = 0.07/100 if Rck <= 60 else 0.2*eps_cu
    
    outDict = {"type": "concreate",
               "gamma_c": gamma_c,
               "Rck" : Rck,
                "fck" : fck,
                "fcm" : fcm,
                "fcd" : fcd,
                "fctm" : fctm,
                "fcfm" : fcfm,
                "fctd" : fctd,
                "Ecm" : Ecm,
                "v": v,
                "vf": vf,
                "Gcm": Gcm,
                "Gcf": Gcf,
                "alpha": alpha,
                "eps_c2": eps_c2,
                "eps_c3": eps_c3,
                "eps_c4": eps_c4,
                "eps_cu": eps_cu
                   }
    
    return outDict