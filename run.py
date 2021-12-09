import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import xgboost
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import lime
import lime.lime_tabular

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False




st.sidebar.header('預計展店相關資料')
with st.sidebar.form(key ='form1'):
    longitude = st.text_input(label = '預計展店店址經度', placeholder = '範例格式: 119.567972')
    latitude = st.text_input(label = '預計展店店址緯度', placeholder = '範例格式: 23.569646')
    
    sells_ground = st.text_input(label = '預計展店賣場坪數', placeholder = '範例格式: 33.0')
    dining_seats_in = st.text_input(label = '預計展店室內用餐區座位數', placeholder = '範例格式: 6')
    dining_seats_out = st.text_input(label = '預計展店室外用餐區座位數', placeholder = '範例格式: 3')
    parking_lot	= st.text_input(label = '預計展店停車格數', placeholder = '範例格式: 5')
    external_lavatory_in = st.text_input(label = '預計展店廁所間數(由賣場出入)', placeholder = '範例格式: 1')
    external_lavatory_out = st.text_input(label = '預計展店廁所間數(由室外出入)', placeholder = '範例格式: 2')
 
    do_name = st.selectbox('預計展店區課', ('三民區', '三重區', '中和區', '中園區',
                                            '中壢區', '中山區', '中彰區', '中投區',
                                            '中正區', '中港區', '中濱區', '中科區',
                                            '中興區', '信義區', '內湖區', '八德區',
                                            '前鎮區', '北宜區', '北屯區', '北投區',
                                            '北桃區', '南投區', '南港區', '南科區',
                                            '台中區', '台南區', '台東區', '嘉新區',
                                            '嘉義區', '土城區', '基隆區', '墾丁區',
                                            '士林區', '大仁區', '大同區', '大安區',
                                            '安澎區', '宜蘭區', '屏東區', '岡山區',
                                            '左營區', '府城區', '彰化區', '彰濱區',
                                            '文山區', '新店區', '新泰區', '新營區',
                                            '新竹區', '新豐區', '松山區', '板城區',
                                            '板橋區', '林口區', '桃園區', '桃新區',
                                            '楊新區', '楠梓區', '樹林區', '永和區',
                                            '永康區', '汐止區', '淡水區', '竹北區',
                                            '竹東區', '竹科區', '竹苗區', '花蓮區',
                                            '苓雅區', '苗栗區', '蘆洲區', '蘆竹區',
                                            '豐原區', '雅潭區', '雲彰區', '雲林區',
                                            '鳳山區', '龍潭區')
                            )   
    
    store_type = st.selectbox('預計展店店性型態', ('FC1',
                                                  'FC2',
                                                  'RC')       
                              )
    store_status = st.selectbox('預計展店門市狀態', ('一般',
                                                    '位移新開',
                                                    '擴店')       
                              )
    business_time = st.selectbox('預計展店營業時間', ('一般',
                                                     '非24小時營業')       
                              )
    area_type = st.selectbox('預計展店商圈類型', ('交通轉運站---其他', '交通轉運站---客運',
                                                '交通轉運站---捷運站(內)', '交通轉運站---捷運站(外)',
                                                '交通轉運站---火車站(內)', '交通轉運站---火車站(外)',
                                                '交通轉運站---高鐵站', '住宅區---交通轉運站', '住宅區---商業區',
                                                '住宅區---娛樂區', '住宅區---工業區', '住宅區---幹道型',
                                                '住宅區---文教區', '住宅區---純住宅型', '住宅區---辦公商圈',
                                                '住宅區---醫院', '住宅區---風景區', '商業區---市場',
                                                '商業區---百貨公司.購物商場', '商業區---補習班',
                                                '商業區---觀光飯店', '商業區---電影院.KTV.保齡球館',
                                                '商業區---餐飲業.商店街', '外島---住宅', '外島---商業',
                                                '外島---幹道', '外島---文教', '外島---醫院',
                                                '工業區---傳統工業', '工業區---高科技業(封閉型)',
                                                '工業區---高科技業(開放型)', '幹道型---交通轉運站',
                                                '幹道型---住宅區', '幹道型---商業區', '幹道型---娛樂區',
                                                '幹道型---工業區', '幹道型---文教區', '幹道型---純幹道型',
                                                '幹道型---辦公商圈', '幹道型---醫院', '幹道型---風景區',
                                                '文教區---大專院校(校內)', '文教區---大專院校(校外)',
                                                '文教區---高中職(含)以下(校內)', '文教區---高中職(含)以下(校外)',
                                                '辦公商圈---封閉型', '辦公商圈---開放型', '醫院---其他',
                                                '醫院---區域醫院(內)', '醫院---區域醫院(外)',
                                                '醫院---醫學中心(內)', '醫院---醫學中心(外)',
                                                '風景區---著名景點', '風景區---風景線中繼站')       
                            )

       
    submitted1 = st.form_submit_button(label = '送出資料進行PSD預測')

@st.cache
def model(input):   
    data = pd.read_csv("8月人流_10月前PSD_精準展店用大表_20211201.csv")
    data = data.dropna(how='any',axis=0)

    X_data = data.drop(['LON', 'LAT', 
                    'STORE_NO','LASTEST_STORE_NO','STORE_NAME',
                    'GRID_LON', 'GRID_LAT', 'FLOW_INDEX', 'AVG_PSD',
                    'fm_100', 'hl_100', 'ok_100', 'px_100', 'sm_100',
                    'fm_250', 'hl_250', 'ok_250', 'px_250', 'sm_250',
                    'fm_500', 'hl_500', 'ok_500', 'px_500', 'sm_500',
                    'fm_1000', 'hl_1000', 'ok_1000', 'px_1000', 'sm_1000'], axis=1)

    Y_data = data['AVG_PSD']

    X_data_processed = pd.get_dummies(X_data)
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(X_data_processed, Y_data)
    
    return rf.predict(input)
    #LIME
    #explainer = lime.lime_tabular.LimeTabularExplainer(X_data_processed.to_numpy(), feature_names=X_data_processed.columns, class_names=['AVG_PSD'], verbose=True, mode='regression')
    #exp = explainer.explain_instance(input[0], rf.predict, num_features=10)
    #return rf.predict(input), exp.as_pyplot_figure()

    

    
if submitted1:
    with st.spinner('預測模型運算中......'):   
        if float(longitude) < 118.3300 or float(longitude) > 121.9900 or float(latitude) < 21.9000 or float(latitude) > 26.3000:
            st.error('請填入位於台灣範圍內的經緯度位址：經度介於118.330000至121.990000；緯度介於21.900000至26.300000')
        elif longitude and latitude and sells_ground and dining_seats_in and dining_seats_out and parking_lot and external_lavatory_in and external_lavatory_out and do_name and store_type and store_status and business_time and area_type:
            data = pd.read_csv("8月人流_10月前PSD_精準展店用大表_20211201.csv")
            lat_s = 21.9000
            lat_n = 26.3000 
            lon_w = 118.3300
            lon_e = 121.9900
            
            longitude = float(longitude)
            latitude = float(latitude)
            
            lon_grid = round(((math.ceil((longitude - lon_w)/0.01))*0.01 + round(lon_w,2) - 0.005), 3)
            lat_grid = round(((math.ceil((latitude - lat_s)/0.01))*0.01 + round(lat_s,2) - 0.005), 3)
            
            grid_info = data.loc[(data['GRID_LON'] == lon_grid) & (data['GRID_LAT'] == lat_grid)][['PEOPLE_FLOW','STORE_AMT','COMPETE_STORE_AMT']]
            if grid_info.shape[0] == 0:
                grid_info = [0]*3
            else:
                grid_info = list(grid_info.iloc[0,:])
            
            
            do_name_dummy = ['三民區', '三重區', '中和區', '中園區',
                            '中壢區', '中山區', '中彰區', '中投區',
                            '中正區', '中港區', '中濱區', '中科區',
                            '中興區', '信義區', '內湖區', '八德區',
                            '前鎮區', '北宜區', '北屯區', '北投區',
                            '北桃區', '南投區', '南港區', '南科區',
                            '台中區', '台南區', '台東區', '嘉新區',
                            '嘉義區', '土城區', '基隆區', '墾丁區',
                            '士林區', '大仁區', '大同區', '大安區',
                            '安澎區', '宜蘭區', '屏東區', '岡山區',
                            '左營區', '府城區', '彰化區', '彰濱區',
                            '文山區', '新店區', '新泰區', '新營區',
                            '新竹區', '新豐區', '松山區', '板城區',
                            '板橋區', '林口區', '桃園區', '桃新區',
                            '楊新區', '楠梓區', '樹林區', '永和區',
                            '永康區', '汐止區', '淡水區', '竹北區',
                            '竹東區', '竹科區', '竹苗區', '花蓮區',
                            '苓雅區', '苗栗區', '蘆洲區', '蘆竹區',
                            '豐原區', '雅潭區', '雲彰區', '雲林區',
                            '鳳山區', '龍潭區']
            do_name_dummy = [int(i == do_name) for i in do_name_dummy] 
            
            store_type_dummy = ['FC1',
                                'FC2',
                                'RC']
            store_type_dummy = [int(i == store_type) for i in store_type_dummy]
            
            store_status_dummy = ['一般',
                                '位移新開',
                                '擴店']
            store_status_dummy = [int(i == store_status) for i in store_status_dummy]
            
            business_time_dummy = ['一般',
                                '非24小時營業']
            business_time_dummy = [int(i == business_time) for i in business_time_dummy]
            
            area_type_dummy = ['交通轉運站---其他', '交通轉運站---客運',
                            '交通轉運站---捷運站(內)', '交通轉運站---捷運站(外)',
                            '交通轉運站---火車站(內)', '交通轉運站---火車站(外)',
                            '交通轉運站---高鐵站', '住宅區---交通轉運站', '住宅區---商業區',
                            '住宅區---娛樂區', '住宅區---工業區', '住宅區---幹道型',
                            '住宅區---文教區', '住宅區---純住宅型', '住宅區---辦公商圈',
                            '住宅區---醫院', '住宅區---風景區', '商業區---市場',
                            '商業區---百貨公司.購物商場', '商業區---補習班',
                            '商業區---觀光飯店', '商業區---電影院.KTV.保齡球館',
                            '商業區---餐飲業.商店街', '外島---住宅', '外島---商業',
                            '外島---幹道', '外島---文教', '外島---醫院',
                            '工業區---傳統工業', '工業區---高科技業(封閉型)',
                            '工業區---高科技業(開放型)', '幹道型---交通轉運站',
                            '幹道型---住宅區', '幹道型---商業區', '幹道型---娛樂區',
                            '幹道型---工業區', '幹道型---文教區', '幹道型---純幹道型',
                            '幹道型---辦公商圈', '幹道型---醫院', '幹道型---風景區',
                            '文教區---大專院校(校內)', '文教區---大專院校(校外)',
                            '文教區---高中職(含)以下(校內)', '文教區---高中職(含)以下(校外)',
                            '辦公商圈---封閉型', '辦公商圈---開放型', '醫院---其他',
                            '醫院---區域醫院(內)', '醫院---區域醫院(外)',
                            '醫院---醫學中心(內)', '醫院---醫學中心(外)',
                            '風景區---著名景點', '風景區---風景線中繼站']
            area_type_dummy = [int(i == area_type) for i in area_type_dummy]
            
            
            #start of data processing
            input = [float(sells_ground), 
                    float(dining_seats_in), 
                    float(dining_seats_out), 
                    float(parking_lot), 
                    float(external_lavatory_in), 
                    float(external_lavatory_out)]
        
            input.extend(grid_info)
            input.extend(do_name_dummy)
            input.extend(store_type_dummy)
            input.extend(store_status_dummy)
            input.extend(business_time_dummy)
            input.extend(area_type_dummy)
            
            input = np.array(input)
            input = input.reshape(1, -1)
            #end of data processing
            
            output = model(input)
            #output, output_exp = model(input)
            output = output.item(0)    #why is it output.item((0,0))?
            st.title('PSD預測結果')
            st.write(output, ' 元')
            
            #LIME
            st.title('PSD預測依據')
            st.write('優化中')
            #st.pyplot(output_exp)

            

        else:
            st.error('請確實於左欄填入預期展店門市的相關資料')
        
#模型變成@st.cache (?)     
#還需設其他提示, 例如經緯度不在台灣範圍 (?)
#把等待提示改成進度條?
#需再調整字型大小?

#需還有等待提示 (O)
#需調整output格式 (O)
#如果沒輸入值就submit, 給予提示 (O)