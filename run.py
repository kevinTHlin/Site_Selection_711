import streamlit as st
import pandas as pd
import os
import xgboost
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics



st.sidebar.header('預計展店相關資料')
with st.sidebar.form(key ='form1'):
    longitude = st.text_input(label = '預計展店店址經度', placeholder = '範例格式: 119.567972')
    latitude = st.text_input(label = '預計展店店址緯度', placeholder = '範例格式: 23.569646')
    
    store_type = st.selectbox('預計展店店性型態', ('FC1',
                                                  'FC2',
                                                  'RC')       
                              )
    store_status = st.selectbox('預計展店門市狀態', ('一般',
                                                    '擴店',
                                                    '位移新開')       
                              )
    
    business_time = st.selectbox('預計展店營業時間', ('一般',
                                                     '非24小時營業')       
                              )
    area_type = st.selectbox('預計展店商圈類型', ('商業區---百貨公司.購物商場',
                                                '交通轉運站---其他',
                                                '幹道型---住宅區',
                                                '商業區---餐飲業.商店街',
                                                '外島---醫院',
                                                '文教區---大專院校(校外)',
                                                '住宅區---幹道型',
                                                '工業區---高科技業(開放型)',
                                                '醫院---醫學中心(外)',
                                                '醫院---醫學中心(內)',
                                                '文教區---大專院校(校內)',
                                                '商業區---補習班',
                                                '交通轉運站---捷運站(內)',
                                                '工業區---高科技業(封閉型)',
                                                '商業區---觀光飯店',
                                                '辦公商圈---開放型',
                                                '住宅區---商業區',
                                                '住宅區---工業區',
                                                '外島---住宅',
                                                '幹道型---文教區',
                                                '幹道型---純幹道型',
                                                '工業區---傳統工業',
                                                '住宅區---辦公商圈',
                                                '交通轉運站---火車站(內)',
                                                '醫院---區域醫院(內)',
                                                '交通轉運站---高鐵站',
                                                '住宅區---娛樂區',
                                                '辦公商圈---封閉型',
                                                '幹道型---醫院',
                                                '幹道型---交通轉運站',
                                                '幹道型---風景區',
                                                '住宅區---交通轉運站',
                                                '風景區---風景線中繼站',
                                                '幹道型---工業區',
                                                '商業區---市場', 
                                                '商業區---電影院.KTV.保齡球館', 
                                                '醫院---區域醫院(外)', 
                                                '外島---商業', 
                                                '幹道型---商業區', 
                                                '幹道型---娛樂區', 
                                                '交通轉運站---火車站(外)', 
                                                '文教區---高中職(含)以下(校外)', 
                                                '文教區---高中職(含)以下(校內)', 
                                                '幹道型---辦公商圈', 
                                                '外島---文教', 
                                                '住宅區---醫院', 
                                                '外島---幹道', 
                                                '住宅區---純住宅型', 
                                                '住宅區---文教區', 
                                                '醫院---其他', 
                                                '交通轉運站---捷運站(外)', 
                                                '交通轉運站---客運', 
                                                '風景區---著名景點', 
                                                '住宅區---風景區')       
                            )
    do_name = st.selectbox('預計展店區課', ('文山區', 
                                            '大同區', 
                                            '嘉新區', 
                                            '大安區', 
                                            '雅潭區', 
                                            '岡山區',
                                            '三民區',
                                            '中山區',
                                            '蘆竹區',
                                            '竹北區',
                                            '新店區',
                                            '墾丁區',
                                            '北宜區',
                                            '豐原區',
                                            '新竹區',
                                            '台中區', 
                                            '樹林區',
                                            '竹苗區',
                                            '永和區',
                                            '新營區',
                                            '北桃區',
                                            '土城區',
                                            '新豐區',
                                            '花蓮區',
                                            '中興區',
                                            '永康區',
                                            '基隆區',
                                            '竹東區',
                                            '府城區',
                                            '宜蘭區',
                                            '北屯區',
                                            '林口區',
                                            '台東區',
                                            '內湖區',
                                            '鳳山區',
                                            '中投區',
                                            '淡水區',
                                            '中彰區',
                                            '彰濱區',
                                            '楠梓區',
                                            '嘉義區',
                                            '中濱區',
                                            '中港區',
                                            '左營區',
                                            '雲彰區',
                                            '南港區',
                                            '楊新區',
                                            '前鎮區',
                                            '板橋區',
                                            '北投區',
                                            '新泰區',
                                            '台南區',
                                            '苗栗區',
                                            '雲林區',
                                            '中壢區',
                                            '苓雅區',
                                            '屏東區',
                                            '桃新區',
                                            '八德區',
                                            '中園區',
                                            '三重區',
                                            '南科區',
                                            '安澎區',
                                            '中科區',
                                            '板城區',
                                            '南投區',
                                            '信義區',
                                            '汐止區',
                                            '中和區',
                                            '蘆洲區',
                                            '竹科區',
                                            '彰化區',
                                            '松山區',
                                            '中正區',
                                            '士林區',
                                            '桃園區',
                                            '大仁區',
                                            '龍潭區')
                            )
       
    sells_ground = st.text_input(label = '預計展店賣場坪數', placeholder = '範例格式: 33.0')
    
    Dining_seats_in = st.text_input(label = '預計展店室內用餐區座位數', placeholder = '範例格式: 6')
    
    Dining_seats_out = st.text_input(label = '預計展店室外用餐區座位數', placeholder = '範例格式: 3')
    
    parking_lot	= st.text_input(label = '預計展店停車格數', placeholder = '範例格式: 5')
    
    External_lavatory_in = st.text_input(label = '預計展店廁所間數(由賣場出入)', placeholder = '範例格式: 1')
    
    External_lavatory_out = st.text_input(label = '預計展店廁所間數(由室外出入)', placeholder = '範例格式: 2')
    
    submitted1 = st.form_submit_button(label = '送出資料進行PSD預測')
    
    
if submitted1:
    st.title('PSD預測結果')
    st.write('數值:  ')
    
    st.title('PSD預測依據')
    st.write('圖示:  ')