import pandas as pd
import numpy as np
import string
import re
import scipy.stats as stats
from collections import defaultdict
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class preprocess:
    
    '''preprocess class gets the primary dataframe into a desired format. This class also adds some feature engineering.'''
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    
    def select_country(self, country_code):
        
        '''Added functionality to select US only'''
        
        mask = self.dataframe['country_code'] == country_code
        self.dataframe = self.dataframe[mask]
        
        
    def clean_category(self):
        
        '''fill category list with unknown rather than nan and split categories into base categories'''
        
        self.dataframe['category_list'].fillna(value='unknown', inplace=True) # replace nan with unknown
        self.dataframe['category_list'] = self.dataframe['category_list'].apply(lambda x: x.split('|')[0].lower()) # select first category
  

    def median_funding(self):
        
        '''select median funding of companies to fill missing funding information'''
        
        mask = self.dataframe['funding_total_usd'] != '-' # mask dataframe to only have true values
        masked_df = self.dataframe[mask]
        
        # calculate median funding of companies
        masked_df['funding_total_usd'] = pd.to_numeric(masked_df['funding_total_usd']) 
        masked_df['average_funding'] = masked_df['funding_total_usd'] / masked_df['funding_rounds']
        median_funding = np.median(masked_df['average_funding'])
        
        # replace missing funding info with median funding
        self.dataframe['funding_total_usd'] = self.dataframe['funding_total_usd'].replace(['-',], str(median_funding))
        self.dataframe['funding_total_usd'] = pd.to_numeric(self.dataframe['funding_total_usd'])
        
        # select average funding per company now that null values have been filled
        self.dataframe['average_funding'] = self.dataframe['funding_total_usd'] / self.dataframe['funding_rounds']

        
    def feature_engineering_name(self):
        
        '''creating a feature based on length of company name'''
        
        # fill nan names with average name length
        filtered_df = self.dataframe[self.dataframe['name'].notnull()]
        mean_len = int(np.mean(filtered_df['name'].apply(lambda x: len(x))))
        mean_word = 'a' * mean_len
        
        # create length of name column
        temp = self.dataframe
        temp['name'] = self.dataframe['name'].fillna(mean_word)
        self.dataframe['len_of_company_name'] = self.dataframe['name'].apply(lambda x : len(x))
        
        
    def feature_engineering_url(self): 
        filtered_df = self.dataframe[self.dataframe['homepage_url'].notnull()]
        mean_len = int(np.mean(filtered_df['homepage_url'].apply(lambda x: len(x))))
        mean_url = 'a' * mean_len
        
        temp = self.dataframe
        temp['homepage_url'] = self.dataframe['homepage_url'].fillna(mean_url)
        self.dataframe['len_of_company_url'] = self.dataframe['homepage_url'].apply(lambda x : len(x))
    
    def feature_engineering_categories(self):
        filtered_df = self.dataframe[self.dataframe['category_list'].notnull()]
        filtered_df['category_list'] = filtered_df['category_list'].apply(lambda x: len(x.split('|')))
        mean_len = int(np.mean(filtered_df['category_list']))
        
        self.dataframe['num_of_company_cat'] = self.dataframe['category_list'].apply(lambda x : len(x.split('|')) if isinstance(x, str) else mean_len)
    
    def feature_engineering_dates(self):
        self.dataframe['founded_at'] = pd.to_datetime(self.dataframe['founded_at'], errors='coerce')
        self.dataframe['first_funding_at'] = pd.to_datetime(self.dataframe['first_funding_at'], errors='coerce')
        self.dataframe['last_funding_at'] = pd.to_datetime(self.dataframe['last_funding_at'], errors='coerce')
        
        
        mean_funding = int(np.mean(self.dataframe['funding_rounds']))
        self.dataframe['funding_rounds'].fillna(value=mean_funding)
        
        
        self.dataframe['first_last_funding'] = (self.dataframe['first_funding_at'] - self.dataframe['last_funding_at']).dt.days
        self.dataframe['first_last_funding']= self.dataframe['first_last_funding'].fillna(0)
        self.dataframe['first_last_funding'] = self.dataframe['first_last_funding'] / self.dataframe['funding_rounds']
        self.dataframe['first_last_funding'] = self.dataframe['first_last_funding'].apply(lambda x: np.abs(x))
        
        
#         # bfill selected as it seems to maintain the poisson shape the best
        self.dataframe['first_last_funding'].replace(to_replace=0, method='bfill', inplace=True)
        
        median_val = int(np.median(self.dataframe['first_last_funding']))
        self.dataframe.replace({'first_last_funding': {0: median_val}}, inplace=True)

    def cols_to_drop(self, drop_lst):
        self.dataframe.drop(drop_lst, inplace=True, axis=1)
        
    def feature_engineering_region_count(self):
        self.dataframe.region.fillna(value='unspecified', inplace=True) # fill missing values
        
        region_count_dct = dict(self.dataframe.region.value_counts()) # value count dict
        self.dataframe['region_count'] = self.dataframe['region'].apply(lambda x: region_count_dct[x]) # map values to regions
        
        funding_dct = dict(self.dataframe.groupby('region').sum().funding_total_usd)
        self.dataframe['funding_total_per_region'] = self.dataframe['region'].apply(lambda x: funding_dct[x])
        
        self.dataframe['funding_per_region_avg'] = self.dataframe['funding_total_per_region'] / self.dataframe['region_count']
        
        avg_funding_dct = dict(region_avg_funding = self.dataframe.groupby('region').sum().average_funding)
        self.dataframe['first_funding_total_per_region'] = self.dataframe['region'].apply(lambda x: funding_dct[x])
        
        self.dataframe['avg_first_funding_per_region'] = self.dataframe['first_funding_total_per_region'] / self.dataframe['region_count']
        
    def feature_engineering_state_count(self):
        self.dataframe.state_code.fillna(value='unspecified', inplace=True) # fill missing values
        
        state_code_count_dct = dict(self.dataframe.state_code.value_counts()) # value count dict
        self.dataframe['state_code_count'] = self.dataframe['state_code'].apply(lambda x: state_code_count_dct[x]) # map values to regions
        
        city_count_dct = defaultdict(int, self.dataframe.city.value_counts()) # value count dict
        self.dataframe['city_count'] = self.dataframe['city'].apply(lambda x: city_count_dct[x])
        
        funding_dct = dict(self.dataframe.groupby('state_code').sum().funding_total_usd)
        self.dataframe['funding_total_per_state_code'] = self.dataframe['state_code'].apply(lambda x: funding_dct[x])
        
        self.dataframe['funding_per_state_code_avg'] = self.dataframe['funding_total_per_state_code'] / self.dataframe['state_code_count']
        
        avg_funding_dct = dict(region_avg_funding = self.dataframe.groupby('state_code').sum().average_funding)
        self.dataframe['first_funding_total_per_state_code'] = self.dataframe['state_code'].apply(lambda x: funding_dct[x])
        
        self.dataframe['avg_first_funding_per_state_code'] = self.dataframe['first_funding_total_per_state_code'] / self.dataframe['state_code_count']
        
    def map_target(self):
        self.dataframe['status'] = self.dataframe.status.map({'closed': 0, 'operating':1, 'acquired':2, 'ipo':3})
        
        
    def industry_count(self):
        cat_dict = defaultdict(list)

        for city, cat in zip(self.dataframe.city, self.dataframe.category_list):
            cat_dict[city].append(cat)
            
        industry_count_dict = defaultdict(int)
        industry_name_dict = defaultdict(str)
        
        for k, v in cat_dict.items():
            industry_count_dict[k] = len(list(set(v)))
        
        for k, v in cat_dict.items():
            industry_name_dict[k] = ",".join(list(set(v)))
            
        self.dataframe['unique_industry_count_in_city'] = self.dataframe['city'].apply(lambda x: industry_count_dict[x])
        self.dataframe['unique_industry_names_in_city'] = self.dataframe['city'].apply(lambda x: industry_name_dict[x])
    
#     def remove_outliers(self):
#         self.dataframe[(np.abs(stats.zscore(self.dataframe[''])) < 3)]
    
    def pipeline(self, country_code, drop_lst):
        self.select_country(country_code)
        self.median_funding()
        self.feature_engineering_name()
        self.feature_engineering_url()
        self.feature_engineering_categories()
        self.feature_engineering_dates()
        self.clean_category()
        self.feature_engineering_region_count()
        self.feature_engineering_state_count()
        self.industry_count()
        self.map_target()
        self.cols_to_drop(drop_lst)
        return self.dataframe


class merge_mult_df:
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def merge_gdp(self, gdp_df):
        '''
        Using gdp csv to calculate gdp in percent and usd for each state in original dataframe
        '''

        us_state_abbrev = {
            'Alabama': 'AL',
            'Alaska': 'AK',
            'American Samoa': 'AS',
            'Arizona': 'AZ',
            'Arkansas': 'AR',
            'California': 'CA',
            'Colorado': 'CO',
            'Connecticut': 'CT',
            'Delaware': 'DE',
            'District of Columbia': 'DC',
            'Florida': 'FL',
            'Georgia': 'GA',
            'Guam': 'GU',
            'Hawaii': 'HI',
            'Idaho': 'ID',
            'Illinois': 'IL',
            'Indiana': 'IN',
            'Iowa': 'IA',
            'Kansas': 'KS',
            'Kentucky': 'KY',
            'Louisiana': 'LA',
            'Maine': 'ME',
            'Maryland': 'MD',
            'Massachusetts': 'MA',
            'Michigan': 'MI',
            'Minnesota': 'MN',
            'Mississippi': 'MS',
            'Missouri': 'MO',
            'Montana': 'MT',
            'Nebraska': 'NE',
            'Nevada': 'NV',
            'New Hampshire': 'NH',
            'New Jersey': 'NJ',
            'New Mexico': 'NM',
            'New York': 'NY',
            'North Carolina': 'NC',
            'North Dakota': 'ND',
            'Northern Mariana Islands':'MP',
            'Ohio': 'OH',
            'Oklahoma': 'OK',
            'Oregon': 'OR',
            'Pennsylvania': 'PA',
            'Puerto Rico': 'PR',
            'Rhode Island': 'RI',
            'South Carolina': 'SC',
            'South Dakota': 'SD',
            'Tennessee': 'TN',
            'Texas': 'TX',
            'Utah': 'UT',
            'Vermont': 'VT',
            'Virgin Islands': 'VI',
            'Virginia': 'VA',
            'Washington': 'WA',
            'West Virginia': 'WV',
            'Wisconsin': 'WI',
            'Wyoming': 'WY'
        }
        
        gdp_df['State_Boundaries_NAME'] = gdp_df['State_Boundaries_NAME'].apply(lambda x: us_state_abbrev[x])
        
        self.dataframe = self.dataframe.merge(gdp_df, left_on='state_code', right_on='State_Boundaries_NAME')
        
        self.dataframe = self.dataframe.drop(['FID', 'State_Boundaries_GEO_ID', 'State_Boundaries_STATE', 'State_Boundaries_NAME',
                 'GDP_by_State___97_to_16__dollar', 'GDP_by_State___97_to_16__doll_1'], axis=1)
        
        grouped_by_state = self.dataframe.groupby('state_code').mean()
        start_percent_col = grouped_by_state.columns.get_loc("GDP_by_State___97_to_16__doll_2")
        stop_percent_col = grouped_by_state.columns.get_loc("GDP_by_State___97_to_16__dol_22")
        
        percent_df = grouped_by_state.iloc[:, start_percent_col:stop_percent_col]
        percent_dict = dict(round(percent_df.mean(axis=1), 2))

        

        start_total_col = grouped_by_state.columns.get_loc("GDP_by_State___97_to_16__dol_22")
        stop_total_col = grouped_by_state.columns.get_loc("SHAPE_Length")

        total_df = grouped_by_state.iloc[:, start_total_col:stop_total_col]
        total_dict = dict(round(total_df.mean(axis=1), 2))
        
        pop_dict = dict(grouped_by_state.State_Boundaries_CENSUSAREA)
        
        
        self.dataframe['percent_gdp_by_state'] = self.dataframe['state_code'].apply(lambda x: percent_dict[x])
        self.dataframe['total_gdp_by_state'] = self.dataframe['state_code'].apply(lambda x: total_dict[x])
        self.dataframe['population_by_state'] = self.dataframe['state_code'].apply(lambda x: pop_dict[x])
        
        drop_idx_1 = self.dataframe.columns.get_loc("State_Boundaries_LSAD")
        drop_idx_2 = self.dataframe.columns.get_loc("percent_gdp_by_state")
        
        part_1 = self.dataframe.iloc[:, :drop_idx_1]
        part_2 = self.dataframe.iloc[:, drop_idx_2:]
        
        self.dataframe = pd.concat([part_1, part_2], axis=1)
        
        
        
    def merge_zip(self, zip_df):
        
        self.dataframe = self.dataframe[self.dataframe['city'].notna()] # no nan's in df.city
        
        # standardize city names
        zip_df['city'] = zip_df.city.apply(lambda x: " ".join(x.title().split())) 
        self.dataframe['city'] = self.dataframe.city.apply(lambda x: " ".join(x.title().split()))
        
        # selecting only the cities in original dataframe
        unique_city_lst = self.dataframe.city.unique().tolist()
        mask = zip_df.city.apply(lambda x: x in unique_city_lst)
        zip_df = zip_df[mask]
        
        pop_density = zip_df.groupby('city').sum() # agg pop and density with sum
        zip_lat = zip_df.groupby('city').mean() # agg zip, lat, long with mean

        pop_dict = defaultdict(int, pop_density.population) # dict of sum population per city
        density_dict = defaultdict(int, pop_density.density) # dict of sum density per city

        zip_dict = defaultdict(int, zip_lat['zip']) # dict of mean zip codes among cities
        lat_dict = defaultdict(int, zip_lat.lat) # dict of mean lat among cities
        long_dict = defaultdict(int, zip_lat.lng) # dict of mean long among cities
        
        self.dataframe['pop_per_city'] = self.dataframe['city'].apply(lambda x: pop_dict[x])
        self.dataframe['pop_density_per_city'] = self.dataframe['city'].apply(lambda x: density_dict[x])
        self.dataframe['zip'] = self.dataframe['city'].apply(lambda x: int(zip_dict[x]))
        self.dataframe['lat'] = self.dataframe['city'].apply(lambda x: lat_dict[x])
        self.dataframe['long'] = self.dataframe['city'].apply(lambda x: long_dict[x])
        
    def merge_edu(self, edu_df):
        
        edu_df['location_city'] = edu_df['Location'].apply(lambda x: x.split(',')[0])
        edu_df['location_state'] = edu_df['Location'].apply(lambda x: x.split(',')[1])

        edu_city = edu_df.groupby('location_city').count()
        edu_state = edu_df.groupby('location_state').count()
        
        education_count_dict = defaultdict(int, edu_city.Rank)
        education_bool_dict = defaultdict(bool, edu_city.Rank != 0)
        education_count_state_dict = defaultdict(int, edu_state.Name)
        
        self.dataframe['num_top_univ_per_city'] = self.dataframe['city'].apply(lambda x: education_count_dict[x])
        self.dataframe['top_univ_in_city'] = self.dataframe['city'].apply(lambda x: education_bool_dict[x])
        self.dataframe['num_top_univ_per_state'] = self.dataframe['state_code'].apply(lambda x: education_count_state_dict[x])
    
    def merge_pipeline(self, gdp_df, zip_df, edu_df):
        self.merge_gdp(gdp_df)
        self.merge_zip(zip_df)
        self.merge_edu(edu_df)
        return self.dataframe
    

class round_two_engineering:
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def success_fail_by_region(self):
        success_mask = (self.dataframe.status > 0)
        success_count_by_region = self.dataframe[success_mask].groupby('region').count().region_count
        success_count_by_region_dict = defaultdict(int, success_count_by_region)


        fail_mask = (self.dataframe.status == 0)

        fail_count_by_region = self.dataframe[fail_mask].groupby('region').count().region_count
        fail_count_by_region_dict = defaultdict(int, fail_count_by_region)



        total_count = self.dataframe.groupby('region').count().region_count
        total_count_dict = defaultdict(int, total_count)

        
        self.dataframe['success_count_by_region'] = self.dataframe['region'].apply(lambda x: success_count_by_region[x])
        self.dataframe['fail_count_by_region'] = self.dataframe['region'].apply(lambda x: fail_count_by_region_dict[x])
        self.dataframe['total_company_count'] = self.dataframe['region'].apply(lambda x: total_count_dict[x])




        self.dataframe['ratio_success_by_region'] = self.dataframe['success_count_by_region'] / self.dataframe['total_company_count']
        self.dataframe['ratio_fail_by_region'] = self.dataframe['fail_count_by_region'] / self.dataframe['total_company_count']


    def engineering_pipeline(self):
        self.success_fail_by_region()
        return self.dataframe
    
    
    

def plot_regions(target, ur_df, sr_df, title_text, filename):

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Bar(name='Top Ten Least Successful Regions',x=ur_df.index, y=ur_df[target]),
        row=1, col=1)

    fig.add_trace(
        go.Bar(name='Top Ten Successful Regions', x=sr_df.index, y=sr_df[target]),
        row=1, col=2)

    fig.update_layout(
        height=550, width=1000, 
        title_text=title_text,
        showlegend=True,
        title_x=0.25)


    fig.update_xaxes(title_text="Unsuccessful Regions", row=1, col=1)
    fig.update_xaxes(title_text="Successful Regions", row=1, col=2)

    fig.update_yaxes(title_text="Number of Start-Ups in Region", row=1, col=1)
    # fig.update_yaxes(title_text="Number of Start-Ups in Region", row=1, col=2)


    fig.write_image(f"../images/{filename}.png")
    fig.show()
    
if __name__ == "__main__":
    print('main')