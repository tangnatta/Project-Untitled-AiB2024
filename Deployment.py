from catboost import CatBoostClassifier, Pool
import streamlit as st
from streamlit_geolocation import streamlit_geolocation
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely

LATLONG_CRS = "EPSG:4326"

FEATURE_SOIL_GROUP = ['grp_properties_upper_organic', 'grp_properties_upper_N', 'grp_properties_upper_P', 'grp_properties_upper_K', 'grp_properties_lower_organic', 'grp_properties_lower_N',
                      'grp_properties_lower_P', 'grp_properties_lower_K', 'grp_properties_upper_pH_upper', 'grp_properties_upper_pH_lower', 'grp_properties_lower_pH_upper',
                      'grp_properties_lower_pH_lower', 'grp_awc_min', 'grp_awc_max', 'grp_awc_avg']

X_FEATURE = ['grp_properties_upper_organic',
             'grp_properties_upper_N', 'grp_properties_upper_P',
             'grp_properties_upper_K', 'grp_properties_lower_organic',
             'grp_properties_lower_N', 'grp_properties_lower_P',
             'grp_properties_lower_K', 'grp_properties_upper_pH_upper',
             'grp_properties_upper_pH_lower', 'grp_properties_lower_pH_upper',
             'grp_properties_lower_pH_lower', 'grp_awc_min', 'grp_awc_max',
             'grp_awc_avg']

# #! load soil data
# soil_CH_gdf = gpd.read_file(
#     ".\\Soil_Chon Buri\\Soil_Chon Buri\\Soil_Chon Buri.shp", encoding="tis-620")
# soil_CH_gdf.to_crs(crs=LATLONG_CRS, inplace=True)

# soil_CHS_gdf = gpd.read_file(
#     ".\\Soil_Chachoengsao\\Soil_Chachoengsao\\Soil_Chachoengsao.shp", encoding="tis-620")
# soil_CHS_gdf.to_crs(crs=LATLONG_CRS, inplace=True)

# soil_JB_gdf = gpd.read_file(
#     ".\\Soil_Chanthaburi\\Soil_Chanthaburi\\Soil_จ.จันทบุรี.shp", encoding="tis-620")
# soil_JB_gdf.to_crs(crs=LATLONG_CRS, inplace=True)

# soil_RY_gdf = gpd.read_file(
#     ".\\Soil_Rayong\\Soil_Rayong\\Soil_จ.ระยอง.shp", encoding="tis-620")
# soil_RY_gdf.to_crs(crs=LATLONG_CRS, inplace=True)

# soil_gdf = gpd.GeoDataFrame(pd.concat(
#     [soil_CH_gdf, soil_CHS_gdf, soil_RY_gdf, soil_JB_gdf], ignore_index=True), geometry="geometry")
# del soil_CH_gdf, soil_CHS_gdf, soil_JB_gdf, soil_RY_gdf

soil_gdf = gpd.read_parquet('soil_gdf.parquet')

soil_grp_df = pd.read_parquet('soil_group_data.parquet').fillna(0)
soil_grp_df[FEATURE_SOIL_GROUP] = soil_grp_df[FEATURE_SOIL_GROUP].map(
    lambda x: 0 if x == -1 else x)


# #! load landuse data
# landuse_CH_gdf = gpd.read_file(
#     ".\\Landuse_Chon Buri_2564\\Landuse_Chon Buri_2564\\Landuse_Chon Buri_2564.shp", encoding='tis-620')
# landuse_CH_gdf.to_crs(crs=LATLONG_CRS, inplace=True)

# landuse_CHS_gdf = gpd.read_file(
#     ".\\Landuse_Chachoengsao_2561\\Landuse_Chachoengsao_2561\\Landuse_ฉะเชิงเทรา_2561.shp", encoding="tis-620")
# landuse_CHS_gdf.to_crs(crs=LATLONG_CRS, inplace=True)

# landuse_JB_gdf = gpd.read_file(
#     ".\\Landuse_Chanthaburi_2561\\Landuse_Chanthaburi_2561\\Landuse_จันทบุรี_2561.shp", encoding="tis-620")
# landuse_JB_gdf.to_crs(crs=LATLONG_CRS, inplace=True)

# landuse_RY_gdf = gpd.read_file(
#     ".\\Landuse_Rayong_2561\\Landuse_Rayong_2561\\LU_ระยอง_2561.shp", encoding="tis-620")
# landuse_RY_gdf.to_crs(crs=LATLONG_CRS, inplace=True)

# landuse_gdf = gpd.GeoDataFrame(pd.concat(
#     [landuse_CH_gdf, landuse_CHS_gdf, landuse_RY_gdf, landuse_JB_gdf], ignore_index=True), geometry="geometry")

#! Load catboost model
model = CatBoostClassifier()
model.load_model("lastest_cat_boost_model.cbm")

soil_ids_search = []
soil_ids_search_grp_id = []

for idx, i in enumerate(soil_grp_df['soil_series_id'].to_list()):
    soil_ids_search += [j.lower() for j in i]
    soil_ids_search_grp_id += [idx+1]*len(i)


def find_soil_group_data(soil_series_id):
    # ex soil_series_id = 48
    soil_series_id = str(soil_series_id).strip()
    if soil_series_id[0].isnumeric():
        return soil_series_id[:2]
    try:
        return soil_ids_search_grp_id[soil_ids_search.index(soil_series_id.lower())]
    except:
        return 61


# def get_soil_group_data(grp_ids):
#     # arr = [0 * len(FEATURE_SOIL_GROUP)]
#     arr = []
#     if len(grp_ids.split(',')) > 1:
#         grp_ids = [find_soil_group_data(i) for i in grp_ids.split(',')]
#         grp_id = grp_ids[0]
#         # for i in grp_ids:
#         #     soil_grp_df.loc[soil_grp_df['grp_id'] == i][FEATURE_SOIL_GROUP]
#     else:
#         grp_id = find_soil_group_data(grp_ids)
#     # print(grp_id)
#     return soil_grp_df[soil_grp_df['grp_id'] == int(grp_id)][FEATURE_SOIL_GROUP].values[0]

def predict(lat, lon):
    #! Find soil group
    point = shapely.geometry.Point(lon, lat)
    soil_grp = soil_gdf[soil_gdf.contains(point)]
    if len(soil_grp) == 0:
        return [[-1]]
    soil_grp = soil_grp.iloc[0]['soilgroup']

    soil_grp = find_soil_group_data(soil_grp)

    #! Find soil group data
    soil_grp_data = soil_grp_df[soil_grp_df['grp_id'] == soil_grp]

    #! Predict
    soil_grp_data = soil_grp_data[FEATURE_SOIL_GROUP]
    soil_grp_data = soil_grp_data.to_numpy().reshape(1, -1)
    if soil_grp_data.shape[1] == 0:
        return [[-1]]
    print(soil_grp_data)
    result = model.predict_proba(soil_grp_data)
    # result = "test"

    return result


CLASSES = ['sugar', 'cassava', 'maize', 'rice']


def result_arr_2_str(arr):
    i = arr[0].argmax()
    return CLASSES[i]


def main():
    st.title("What plant should be plant here")
    st.write("This app support only location with in Chon Buri, Chachoengsao, Chanthaburi and Rayong province")
    label, btn = st.columns(2)

    # location = streamlit_geolocation()

    with label:
        st.write("Use current position: ")
    with btn:
        location = streamlit_geolocation()

    if location["latitude"] is None and location["longitude"] is None:
        location["latitude"] = 13.361389
        location["longitude"] = 100.984722

    # Input fields for latitude and longitude
    lat, lon = st.columns(2)

    with lat:
        lat = st.number_input(
            "Enter Latitude", format='%.7f', value=location["latitude"])
    with lon:
        lon = st.number_input(
            "Enter Longitude", format='%.7f', value=location["longitude"])

    res = predict(lat, lon)
    if len(res[0]) <= 1:
        st.subheader("This location is not in the supported area")
        return

    result = result_arr_2_str(res)

    # Output the result
    # st.write(f"The latitude is {lat} and the longitude is {lon}")
    st.subheader(f"Best plant to plant here is {result}")
    st.write(
        f"The probability of {result} is {np.round(res[0].max()*100, 2)} or {CLASSES}:{np.round(res[0]*100, 2)}")


if __name__ == "__main__":
    main()
