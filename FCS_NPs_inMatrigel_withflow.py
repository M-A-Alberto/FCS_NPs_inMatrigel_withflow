# -*- coding: utf-8 -*-
"""
Created on Wed May  1 06:58:13 2024

@author: amart
"""



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import tukey_hsd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score 


os.chdir(r"XXXXXXXXX") #Working directory

sns.set_theme()
sns.set_style("ticks")
sns.set_context("paper",font_scale=1.25)
sns.set_palette("tab10")


#Define models
def diffusion(tau,g0,td):
    
    """
    Free diffusion model
    """
    
    y = g0/((1+(tau/td))*(1+(w**2/z0**2)*(tau/td))**(1/2))
    
    return y

def anomalous_D(tau,g0,td,a,):
    
    """
    Anomalous diffusion model
    """
    
    y = g0/((1+(tau/td)**a)*(1+(w**2/z0**2)*(tau/td)**a)**(1/2))
    
    return y
    

def flow(tau,g0,v):
    """
    Pure convection flow model    

    Parameters
    ----------
    tau : numpy.ndarray
        Lag-time
    g0 : float
       Correlation amplitude.
    v : float
      Flow velocity [m/s].

    Returns
    -------
    y : numpy.ndarray
        ACF.

    """
        
    
    y = g0*np.exp(-(v*tau/w)**2)

    return y

def combined_anomalous_model(tau,g0,td,v,a):
    """
    Combined anomalous diffusion convection flow model    

    Parameters
    ----------
    tau : numpy.ndarray
        Lag-time
    g0 : float
       Correlation amplitude.
    td : float 
       Diffusion time [s]
    v : float
      Flow velocity [m/s].
    a : float
      Anomalous exponent.
      
    Returns
    -------
    y : numpy.ndarray
        ACF.

    """
    
    y = g0/((1+(tau/td)**a)*(1+(w**2/z0**2)*((tau/td)**a))**(1/2))*np.exp(-(v*tau/w)**2/((1+(tau/td)**a)*(1+(w**2/z0**2)*((tau/td)**a))**(1/2)))
    
    
    
    return y

# Define fuctions to fit different models
def fit_anomalous_diffusion_model(t,G_exp,results,k,point,Q,ax,C,exp):
    
    try:
        pars,cov = curve_fit(anomalous_D,t,G_exp,bounds = ((0,0,0),(np.inf,np.inf,np.inf)))
        
        (g0,tau_d,a) = pars
        
        d_w = 2/a
        
        D = w**2/(4*tau_d*1e-3)
        
        G_fit = anomalous_D(t,*pars)
        
        chi_2 = r2_score(G_exp,G_fit)
        

        print(f"Flow {Q} Point {point}: Anomalous model r2: {chi_2:0.4f}")
        
        
        ax.plot(t,G_fit,"r-",label = "Anomalous Model")
        
        if chi_2 > 0.994:
            results.loc[k,"Matrigel Concentration (mg/ml)"] = C
            results.loc[k,"Experiment"] = exp
            results.loc[k,"Q ($\mu$l/min)"] = Q
            results.loc[k,"D$_{anomalous}$ ($\mu$m$^2$/s$^{\alpha}$)"] = D
            results.loc[k,r"Diffusion exponent ($\alpha$)"] = a
            results.loc[k,"Fractal dimension (d$_w$)"] = d_w
            results.loc[k,"G$_0$ anomalous diffusion model"] = g0
            results.loc[k,"Point"] = point
        else:
            
            print("Anomalous Diffusion model discarded")
        
    except:
        
        print(f"Flow {Q} Point {point} anomalous model not converged")
      
    
    
    return results
    

def fit_flow_model(t,G_exp,results,k,point,Q,ax,C,exp):
    
    try:
        pars,cov = curve_fit(flow,t,G_exp,p0 = [0.1,10],bounds=((0,0),(np.inf,np.inf)))
        
        (g0,v) = pars
        
        
        
        G_fit = flow(t,*pars)
        
        chi_2 = r2_score(G_exp,G_fit)
        
        ax.plot(t,G_fit,"g-",label = "Flow Model")
        
        results.loc[k,"Matrigel Concentration (mg/ml)"] = C
        results.loc[k,"Experiment"] = exp
        results.loc[k,"Q ($\mu$l/min)"] = Q
        results.loc[k,"Flow velocity ($\mu$m/s)"] = np.abs(v)*1000
        results.loc[k,"G$_0$ flow model"] = g0
        results.loc[k,"Point"] = point
        
      
        
        
    
    except:
        
        print(f"Flow {Q} Point {point} flow model not converged")
       
    return results


def fit_diffusion_model(t,G_exp,results,k,point,Q,ax,C,exp):
    
    try:
        pars,cov = curve_fit(diffusion,t,G_exp,bounds=((0,0),(np.inf,np.inf)))
        
        (g0,td) = pars
        
        D = w**2/(4*td*1e-3)
        
        G_fit = diffusion(t,*pars)
        
        chi_2 = r2_score(G_exp,G_fit)
        
        ax.plot(t,G_fit,"r--",label = "Pure Diffusion Model")
        
        results.loc[k,"Matrigel Concentration (mg/ml)"] = C
        results.loc[k,"Experiment"] = exp
        results.loc[k,"Q ($\mu$l/min)"] = Q
        results.loc[k,"D$_{pure}$ ($\mu$m$^2$/s)"] = D
        results.loc[k,"G$_0$ diffusion model"] = g0
        results.loc[k,"Point"] = point
        
        
    except:
        
        print(f"Flow {Q} Point {point} diffusion model not converged")
       
    return results


def fit_combined_anomalous_model(t,G_exp,results,k,point,Q,C,exp):
    
    try:
        pars_combined,cov = curve_fit(combined_anomalous_model,t,G_exp,bounds = ((0,0,-np.inf,0),(np.inf,np.inf,np.inf,np.inf)))
        
        (g0,td,v,a) = pars_combined
        
        d_w = 2/a
        
        D = w**2/(4*td*1e-3)
        
        G_fit = combined_anomalous_model(t,*pars_combined)
        
        chi_2 = r2_score(G_exp,G_fit)
        

        print(f"Flow {Q} Point {point}: Combined Anomalous model r2: {chi_2:0.4f}")
        
        
        #ax.plot(t,G_fit,"-",color = "orange",label = "Combined Anomalous Model")
        
        if chi_2 > 0.994:
            results.loc[k,"Matrigel Concentration (mg/ml)"] = C
            results.loc[k,"Experiment"] = exp
            results.loc[k,"Q ($\mu$l/min)"] = Q
            results.loc[k,"D$_{anomalous}$ ($\mu$m$^2$/s$^{\alpha}$)"] = D
            results.loc[k,"Flow velocity ($\mu$m/s)"] = np.abs(v)*1000
            results.loc[k,r"Diffusion exponent ($\alpha$)"] = a
            results.loc[k,"Fractal dimension (d$_w$)"] = d_w
            results.loc[k,"G$_0$ anomalous diffusion model"] = g0
            results.loc[k,"Point"] = point
            
        else:
            
            print("Combined Anomalous model discarded")
            
    except:
        
        print(f"Flow {Q} Point {point} combined anomalous model not converged")
      
    
    
    return results

#Plotting function
def plot_graphs(results_anomalous,results_combined_anomalous,results_diffusion,results_flow):
    
    palette = "tab10"
    """
    sns.barplot(data = results_anomalous,x="Q ($\mu$l/min)",y="D$_{anomalous}$ ($\mu$m$^2$/s$^{\alpha}$)",hue = "Matrigel Concentration (mg/ml)",palette = palette,errorbar="se",capsize=0.1)
    plt.title("From anomalous model")
    plt.show()
    plt.close()
    
    
    
    sns.barplot(data = results_anomalous,x="Q ($\mu$l/min)",y=r"Diffusion exponent ($\alpha$)",hue = "Matrigel Concentration (mg/ml)",palette = palette,errorbar="se",capsize=0.1)
    plt.title("From anomalous model")
    plt.show()
    plt.close()
    
    """
    
    sns.barplot(data = results_combined_anomalous,x="Q ($\mu$l/min)",y="D$_{anomalous}$ ($\mu$m$^2$/s$^{\alpha}$)",hue = "Matrigel Concentration (mg/ml)",palette = palette,errorbar="se",capsize=0.1,saturation=1)
    #plt.title("From combined anomalous model")
    plt.show()
    plt.close()
    
    
    sns.barplot(data = results_combined_anomalous,x="Q ($\mu$l/min)",y=r"Diffusion exponent ($\alpha$)",hue = "Matrigel Concentration (mg/ml)",palette = "hls",errorbar="se",capsize=0.1,saturation=1)
    #plt.title("From combined anomalous model")
    plt.show()
    plt.close()
    
    
    sns.barplot(data = results_combined_anomalous,x="Q ($\mu$l/min)",y="Flow velocity ($\mu$m/s)",hue = "Matrigel Concentration (mg/ml)",palette = "Paired",errorbar="se",capsize=0.1,saturation=1)
    #plt.title("From combined anomalous model")
    plt.show()
    plt.close()
    
    sns.barplot(data = results_combined_anomalous,x="Q ($\mu$l/min)",y="G$_0$ anomalous diffusion model",hue = "Matrigel Concentration (mg/ml)",palette = "Set2",errorbar="se",capsize=0.1,saturation=1)
    #plt.title("From combined anomalous model")
    plt.show()
    plt.close()
    
    """
    sns.violinplot(data = results_combined_anomalous,x="Q ($\mu$l/min)",y="Flow velocity ($\mu$m/s)",hue = "Matrigel Concentration (mg/ml)",palette = palette)
    plt.title("From combined anomalous model")
    plt.show()
    plt.close()
    
    
    
    sns.barplot(data = results_diffusion,x="Q ($\mu$l/min)",y="D$_{pure}$ ($\mu$m$^2$/s)",hue = "Matrigel Concentration (mg/ml)",palette = palette,errorbar="se",capsize=0.1)
    plt.title("From pure diffusion model")
    plt.show()
    plt.close()
    
    
    
    
    
    sns.barplot(data=results_flow, x = "Q ($\mu$l/min)", y = "Flow velocity ($\mu$m/s)",hue = "Matrigel Concentration (mg/ml)",palette = palette,errorbar="se",capsize=0.1)
    
    plt.title("From pure flow model")
    plt.show()
    plt.close()
    """
    


kb = 1.38e-23


T = 273.15+37

#Calibrations
z0s = {"100424":8.68965,"140324":3.96377,"160424":5.3711475,"170424":2.18211,"240424":7.99921e10,"250424":2.63658,"180924":2.67922,"190924":3.39069e12}


ws = {"100424":0.24816,"140324":0.239,"160424":0.24862,"170424":0.23252,"240424":0.2536,"250424":0.23893,"180924":0.20826,"190924":0.21549}


results_diffusion = pd.DataFrame()
results_anomalous = pd.DataFrame()
results_flow = pd.DataFrame()
results_combined_anomalous = pd.DataFrame()

exclude=[]
#exclude = ["10 0","10 1","10 12"]
k=0

#Analyze data
for C in os.listdir():
    
    if C != "Figures":
        
        
        for exp in os.listdir(f"{C}"):
            
            w = ws[exp]
            
            z0 = z0s[exp]
            
            
            
            if exp in ["100424","140324","160424","170424"]:
                Q = 0
                
                for file in os.listdir(f"{C}/{exp}"):
                    data = pd.read_csv(f"{C}/{exp}/{file}",skiprows=1,sep="\t",skipinitialspace=True)
                    
                    point=0
                    
                    for col in data.columns:
                        if "Correlation Channel" in col:
                            
                            
                                G_exp = data.loc[:,col].dropna()
                                
                                t = data.loc[:len(G_exp)-1,"Time [ms]"]
                                
                                #plt.title(f"{C} {exp} {Q} {point}")
                                
                                #plt.plot(t,G_exp,"b-",label="Data")
                                
                                #ax = plt.gca()
                                if f"{Q} {point}" in exclude:
                                    print(f"{Q} {point} excluded")
                                else:
                                        
                                    #results_diffusion = fit_diffusion_model(t, G_exp, results_diffusion, k, point, Q, ax,C,exp)
                                    #results_anomalous = fit_anomalous_diffusion_model(t,G_exp,results_anomalous,k,point,Q,ax,C,exp)
                                    #results_flow = fit_flow_model(t,G_exp,results_flow,k,point,Q,ax,C,exp)
                                    results_combined_anomalous = fit_combined_anomalous_model(t,G_exp,results_combined_anomalous,k,point,Q,C,exp)
                                    
                                k+=1
                                point+=1
                                if point==4:
                                    point=0
                                """
                                plt.xscale("log")
                                plt.xlabel("Time (ms)")
                                plt.ylabel(r"G ($\tau$)")
                                plt.legend()
                                #plt.show()
                                plt.close()
                                """
                                
            else:
                for Q in [10,20,50]:
                    
                    for file in os.listdir(f"{C}/{exp}"):
                        
                        if f"{Q} ul_min" in file:
                            
                            data = pd.read_csv(f"{C}/{exp}/{file}",skiprows=1,sep="\t",skipinitialspace=True)
                            
                            point=0
                            
                            for col in data.columns:
                                if "Correlation Channel" in col:
                                    
                                    
                                        G_exp = data.loc[:,col].dropna()
                                        
                                        t = data.loc[:len(G_exp)-1,"Time [ms]"]
                                        
                                        #plt.title(f"{C} {exp} {Q} {point}")
                                        
                                        #plt.plot(t,G_exp,"b-",label="Data")
                                        
                                        #ax = plt.gca()
                                        if f"{Q} {point}" in exclude:
                                            print(f"{Q} {point} excluded")
                                        else:
                                                
                                            #results_diffusion = fit_diffusion_model(t, G_exp, results_diffusion, k, point, Q, ax,C,exp)
                                            #results_anomalous = fit_anomalous_diffusion_model(t,G_exp,results_anomalous,k,point,Q,ax,C,exp)
                                            #results_flow = fit_flow_model(t,G_exp,results_flow,k,point,Q,ax,C,exp)
                                            results_combined_anomalous = fit_combined_anomalous_model(t,G_exp,results_combined_anomalous,k,point,Q,C,exp)
                                            
                                        k+=1
                                        point+=1
                                        if point==4:
                                            point=0
                                        """
                                        plt.xscale("log")
                                        plt.xlabel("Time (ms)")
                                        plt.ylabel(r"G ($\tau$)")
                                        plt.legend()
                                        #plt.show()
                                        plt.close()
                                        """
                            
                    
#plot_graphs(results_anomalous,results_combined_anomalous,results_diffusion,results_flow)            


#Plotting and statistics
means = results_combined_anomalous.groupby(by=["Matrigel Concentration (mg/ml)","Experiment","Q ($\mu$l/min)"],as_index=False,observed=False).mean()



sns.barplot(data = means,x="Q ($\mu$l/min)",y="D$_{anomalous}$ ($\mu$m$^2$/s$^{\alpha}$)",hue = "Matrigel Concentration (mg/ml)",palette = "tab10",errorbar="se",capsize=0.1,saturation=1)
#plt.title("From combined anomalous model")
plt.legend(loc="lower center",title = "Matrigel Concentration (mg/ml)")
plt.ylabel(r"Diffusion Coefficient ($\mu$m$^2$/s$^{\alpha}$)")
plt.savefig(r"Figures/Diffusion Coefficients.tif",dpi=100,bbox_inches="tight")
plt.show()
plt.close()


sns.barplot(data = means,x="Q ($\mu$l/min)",y="Flow velocity ($\mu$m/s)",hue = "Matrigel Concentration (mg/ml)",palette = "Paired",errorbar="se",capsize=0.1,saturation=1)
#plt.title("From combined anomalous model")

plt.savefig(r"Figures/Flow velocities.tif",dpi=100,bbox_inches="tight")
plt.show()
plt.close()


sns.barplot(data = means,x="Q ($\mu$l/min)",y=r"Diffusion exponent ($\alpha$)",hue = "Matrigel Concentration (mg/ml)",palette = "hls",errorbar="se",capsize=0.1,saturation=1)
#plt.title("From combined anomalous model")
plt.savefig(r"Figures/Diffusion exponents.tif",dpi=100,bbox_inches="tight")
plt.show()
plt.close()



sns.barplot(data = means,x="Q ($\mu$l/min)",y="G$_0$ anomalous diffusion model",hue = "Matrigel Concentration (mg/ml)",palette = "Set2",errorbar="se",capsize=0.1,saturation=1)
#plt.title("From combined anomalous model")
plt.ylabel("G$_0$")
plt.savefig(r"Figures/G0.tif",dpi=100,bbox_inches="tight")
plt.show()
plt.close()


Mat5 = results_combined_anomalous.loc[results_combined_anomalous.loc[:,"Matrigel Concentration (mg/ml)"]=="5.0 mg_ml",:]


ROI = Mat5.iloc[:,1:-1].groupby(by=["Experiment","Q ($\mu$l/min)"],as_index=False).mean()


populations_D = []
populations_v = []
populations_a = []

for Q in [0,10,20,50]:

    sel = list(ROI.loc[ROI.loc[:,"Q ($\mu$l/min)"]==Q,"D$_{anomalous}$ ($\mu$m$^2$/s$^{\alpha}$)"])

    populations_D.append(sel)
    
    sel_v = list(ROI.loc[ROI.loc[:,"Q ($\mu$l/min)"]==Q,"Flow velocity ($\mu$m/s)"])

    populations_v.append(sel_v)
    
    sel_a = list(ROI.loc[ROI.loc[:,"Q ($\mu$l/min)"]==Q,r"Diffusion exponent ($\alpha$)"])

    populations_a.append(sel_a)
    

stats_D = tukey_hsd(*populations_D)

print(r"Statistics for D anomalous 5.0 mg/ml")


print(stats_D)


stats_v = tukey_hsd(*populations_v)

print(r"Statistics for vf anomalous 5.0 mg/ml")


print(stats_v)


stats_a = tukey_hsd(*populations_a)

print(r"Statistics for a anomalous 5.0 mg/ml")


print(stats_a)



Mat8_7 = results_combined_anomalous.loc[results_combined_anomalous.loc[:,"Matrigel Concentration (mg/ml)"]=="8.7 mg_ml",:]

ROI = Mat8_7.iloc[:,1:-1].groupby(by=["Experiment","Q ($\mu$l/min)"],as_index=False).mean()


populations_D = []
populations_v = []
populations_a = []

for Q in [0,10,20,50]:

    sel = list(ROI.loc[ROI.loc[:,"Q ($\mu$l/min)"]==Q,"D$_{anomalous}$ ($\mu$m$^2$/s$^{\alpha}$)"])

    populations_D.append(sel)
    
    sel_v = list(ROI.loc[ROI.loc[:,"Q ($\mu$l/min)"]==Q,"Flow velocity ($\mu$m/s)"])

    populations_v.append(sel_v)
    
    sel_a = list(ROI.loc[ROI.loc[:,"Q ($\mu$l/min)"]==Q,r"Diffusion exponent ($\alpha$)"])

    populations_a.append(sel_a)
    

stats_D = tukey_hsd(*populations_D)

print(r"Statistics for D anomalous 8.7 mg/ml")


print(stats_D)


stats_v = tukey_hsd(*populations_v)

print(r"Statistics for vf anomalous 8.7 mg/ml")


print(stats_v)


stats_a = tukey_hsd(*populations_a)

print(r"Statistics for a anomalous 8.7 mg/ml")


print(stats_a)



