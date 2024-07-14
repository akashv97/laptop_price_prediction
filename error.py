import streamlit as stm
import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

stm.title("Laptop Price Predictor")

company = stm.selectbox('Brand', df['Company'].unique())

type = stm.selectbox('Type', df['TypeName'].unique())

ram = stm.selectbox('Ram_Variant_In_GB', df['Ram'].unique())

# Ram = stm.selectbox('Ram_Variant_In_GB', [1, 2, 4, 6, 8, 12, 14])

weight = stm.number_input("Weight Of The Laptop")

touchscreen = stm.selectbox("TouchScreen", ['Yes', "No"])

ips = stm.selectbox("IPS_Display", ['Yes', "No"])

screen_size = stm.number_input('Screen Size')

resolution = stm.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

cpu = stm.selectbox("CPU Brand", df['Cpu_brnd'].unique())

# hdd = stm.selectbox('HDD_In_GB', [0,128,256,512,1024,2048])

# ssd= stm.selectbox('SSD_In_GB', [0,8,128,256,512,1024])

ssd= stm.selectbox('SSD_In_GB', df['SSD'].unique())

hdd = stm.selectbox('HDD_In_GB', df['HDD'].unique())

gpu = stm.selectbox("GPU", df['Gpu_brnd'].unique())

os = stm.selectbox("OS", df['OS_typ'].unique())

if stm.button("Predict Price"):
    # query
    ppi = None
    if touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

    if ips == 'Yes':
        Ips = 1
    else:
        Ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = round(((X_res**2) + (Y_res**2))**0.5/screen_size,2)
    # query = np.array([Company,TypeName,Ram,Weight,Touchscreen,Ips,ppi,cpu,hdd,ssd,gpu,os])
    query_df = pd.DataFrame([[company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os]],columns=['Company','TypeName','Ram','Weight',
    'Touchscreen','Ips','ppi','Cpu_brnd','HDD','SSD','Gpu_brnd','OS_typ'])
    output=pipe.predict(query_df)
    b=np.exp(output)
    stm.title(str(b[0]))

    # print(query)

    # query = query.reshape(1,12)
    # stm.title(str(pipe.predict(query)))
    # print(str(pipe.predict(query)[0]))
    # stm.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

