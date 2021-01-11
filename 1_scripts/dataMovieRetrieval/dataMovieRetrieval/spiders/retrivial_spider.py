#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Régis Gbenou
@email: regis.gbenou@outlook.fr


This script allows us to retrieve, from ImDb, the following movie information:
    - title
    - runtime
    - year
    - country 
    - genre
    - budget
    - cumulative worldwide gross

"""
###############################################################################
#                             LIBRAIRIES
###############################################################################

import scrapy         # Library to crawl web information on sites.
import pandas as pd   # Library for data frame manipulation.
import re             # Library for regex operations.


###############################################################################
#                             FUNCTIONS
###############################################################################

def strg_fct(pattern, s):
    w = re.search(pattern, s)
    if w:
        word = (w.group(0).replace(' ', '')).replace('\n', u'')
    else:
        word = ''
    return word


###############################################################################
#                             INITIALIZATION
###############################################################################

'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
1) 
THIS PART ALLOW US TO INITIALIZE DATA.
'''
year_start, year_end = 2000, 2019

iter_csv = pd.read_csv('../../../2_data/1_original/0_title.basics.tsv', iterator=True, sep="\t",
                        chunksize=10000)

  
tab = pd.concat(
    [chunk[(chunk["titleType"] == "movie") &\
           (~chunk["runtimeMinutes"].str.contains("\\\\N", na=False)) &\
           (~chunk["genres"].str.contains("\\\\N", na=False))]\
     for chunk in iter_csv]
   )
        
tab["startYear"] = tab["startYear"].replace("\\N", "0")
tab["startYear"] = tab["startYear"].astype("int32")
tab2 = tab[(tab.startYear >= year_start) & (tab.startYear<=year_end)]
id_list = list(tab2.tconst)


tab2.to_csv(f'../../2_data/1_rawData{year_start}_{year_end}.csv', sep=';', index=False)



tab0 = pd.read_csv(
    f'../../../2_data/1_rawData{year_start}_{year_end}.csv', sep=';')                 # contains all of the references which have the release year between 2000 and 2020.

id_list = list(tab0.tconst)               
lclass = ['Budget', 'Opening Weekend USA', 
          'Gross USA', 'Cumulative Worldwide Gross']
lclass2 = [k+' curr' for k in lclass]
l_col = 'title,year,runtime,genre,note,nbre_note'.split(',') +\
    lclass2 + lclass + 'country,Directors,Writers,Stars'.split(',') # List of keys for the <amount_set_tot>.
amount_set_tot = {k:[] for k in l_col}                              # Dictionary that will gather of the desired information.


'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
2) 
THIS PART ALLOW US TO FETCH DESIRED DATA.
'''
class MovieInformationSpider(scrapy.Spider):
    name = "movieInformation"
    start_urls = [''.join(['https://www.imdb.com/title/',k,
                           '/']) for k in id_list]           # contains the web address associated with each movie having its reference in <id_list>.


    def parse(self, response):
        resp_title = response.css('div.title_wrapper')
        if not resp_title:                                   # if we cannot reach the title, its surely means taht information are not utilizable.
            pass
        else:
            # a. Getting Runtime and Title ------------------------------------
            title = resp_title.css('h1::text').get(
                default='').replace(u'\xa0', u'')
            amount_set_tot['title'].append(title)            # gets the movie title.
            
            resp_time = resp_title.css('div.subtext')
            duration = resp_time.css('time::text').get(
                default='').replace(u'\n', u'').strip()
            amount_set_tot['runtime'].append(duration)       # gets the movie duration.
            
            
            # b. Getting Year, Country and Genre ------------------------------
            info_list = resp_time.css('a::text').getall()    # contains information relative to genre, release year and country.
            key_list = ['year', 'country', 'genre']    
            p_year, p_country, p_genre = '\d{4}', '(?<=\()\D+(?=\))', '[A-Za-z-|]+'
            result_list = ['' for k in key_list]
            if len(info_list) > 1:
                word_list = [re.search(p_year, info_list[-1]),
                             re.search(p_country, info_list[-1]),
                             re.search(p_genre, '|'.join(info_list[:-1]))]
            else:
                if len(info_list) == 1:
                    word_list = [re.search(p_year, info_list[-1]),
                                 re.search(p_country, info_list[-1]),
                                 re.search(p_genre, info_list[-1])]
                else:
                    word_list = [re.search('', '\w') for i in range(3)]                               
            for i,k in enumerate(key_list):
                if word_list[i]:
                    result_list[i] = word_list[i].group(0)
                else:
                    result_list[i] = ''
                amount_set_tot[k].append(result_list[i])  
            
            
            # c. Getting Note and Vote number ---------------------------------
            resp_note = response.css(
                'div.ratings_wrapper div.ratingValue')            
            note = resp_note.xpath(
                '//span[@itemprop="ratingValue"]/text()').get(default='')
            nbre_note = resp_note.xpath(
                '//span[@itemprop="ratingCount"]/text()'
                ).get(default='').replace(',', '')
            amount_set_tot['note'].append(note)
            amount_set_tot['nbre_note'].append(nbre_note)
            
            
            # d. Getting Directors, Writers and Stars -------------------------
            resp_info = response.css('div.plot_summary div.credit_summary_item')
            dws_list = ['Directors', 'Writers', 'Stars']
            for k in dws_list:
                amount_set_tot[k].append('')
            n = len(dws_list)
            for i in range(n):
                key, val = dws_list[i], ''
                try:
                    key = resp_info.css(
                        'h4.inline::text')[i].get().replace(':', '')
                except IndexError:
                    pass
                try:
                    val = '|'.join(re.findall(
                        r'(?<=>)[A-Z]\w+\s[A-Z]\w+', resp_info[i].get()))
                except IndexError:
                    pass 
                if list(key)[-1] == 's':
                    pass
                else:
                    key += 's'
                if key in dws_list:                
                    amount_set_tot[key][-1] = val
                else:                
                    amount_set_tot[dws_list[i]][-1] = val
            
            
            # e. Getting Budget and Cumulative Worldwide Gross ----------------
            resp = response.css(
                'body.fixed div.redesign div.flatland div.article h3.subheading') 
            money_list = [quote.replace(':', '') for quote in\
                          resp.xpath('following::*').css('h4.inline::text'
                                                         )[:8][::2].getall()] 
            value_list = [] 
            currency_list = []
            if len(money_list) == 0: 
                for i,k in enumerate(lclass): 
                    amount_set_tot[k].append('') 
                    amount_set_tot[lclass2[i]].append('')
            else: 
                for i,k in enumerate(resp.xpath('following::*')[:10].re(
                        r'h4>\D+\d[\d,]*\d')): 
                    
                    pattern_currency = '(?<=h4>)\D{1,3}'
                    currency_list.append(strg_fct(pattern_currency, k))
                    
                    pattern_amount_str = '(?<=h4>)\D+\d[\d,]*'
                    amount_str = strg_fct(pattern_amount_str, k)
                    
                    pattern_amount = '\d[\d,]*'
                    value_list.append(
                        strg_fct(pattern_amount, amount_str).replace(',',''))
                for i,k in enumerate(lclass): 
                    value, curr = '', ''
                    if k in money_list: 
                        try: 
                            value = value_list[money_list.index(k)] 
                            curr = currency_list[money_list.index(k)]
                        except IndexError: 
                            pass             
                    else:
                        pass          
                    amount_set_tot[k].append(value) 
                    amount_set_tot[lclass2[i]].append(curr)


        # f. Storing information in csv file ----------------------------------
        yield pd.DataFrame(
            amount_set_tot).to_csv('../../../2_data/2_imdbRetrieval.csv', sep=';',
                                   index=False)