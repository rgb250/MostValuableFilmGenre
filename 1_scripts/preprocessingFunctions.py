#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Régis Gbenou

The aim here is to build a dictionary that collects all of the exchange rates
based on US dollars ($) in a given time periode.
"""

###############################################################################
#                             LIBRARIES
###############################################################################
import numpy as np             # library for matrix manipulation.
import re                      # library for regular expressions.
import requests                # library for requesting web pages.
from bs4 import BeautifulSoup  # library for getting information from HTML pages.
from ast import literal_eval   # library for interpreting python objects.


###############################################################################
#                             FUNCTIONS
###############################################################################

## Recognization of python objects.
def literal_ev(row):
    '''
    <lit_ev> abréviation for litteral_eval, is a function which interprets 
    python object, here sets and lists.
    
    Parameters
    ----------
    row : str
        Any character string.

    Returns
    -------
        Python object
        DESCRIPTION.

    '''
    if row == "set()":                       # Instead of '{}' to signal the presence of an empty set there is 'set()'
        row = "{}"
    else:
        pass
    return literal_eval(row) 


## Getting Currency.
pattern_currency = "\D+"                     # sequence of non digit characters whose length >= 0
prog_currency = re.compile(pattern_currency) # transforms <pattern_currency> in search criterion.

def get_currency(row, name):
  '''
    Getting the first non-numerical character sequence of the <row> dictionary 
    at <name> key.

    Parameters
 <   ----------
    row : dict
        Dictionary.
    name : str
        One of the keys of <row>.

    Returns
    -------
    str
        Frst non numerical chararcter sequence.

    '''
  x = name
  if x in list(row):
    return prog_currency.match(row[x]).group(0) # First sequence that matches with <pattern_currency>.
  else:
    return "non-existent"
  

## Getting money amount.
pattern_amount = "(\d+[\d,]*)*\d"               # Either sequence of number that can contains comas.
prog_amount = re.compile(pattern_amount)        # transforms <pattern_amount> in search criterion.

def get_amount(row, name):
    '''
    Getting the number the firtst numerical sequence of the <row> dictionnary 
    at <name> key.

    Parameters
    ----------
    row : dict
        Dictionary.
    name : str
        One of the keys of <row>.

    Returns
    -------
    str
        First numerical sequence.

    '''
    x = name
    if x in list(row):
      y = prog_amount.search(row[x]).group(0)   # First sequence that matches with <pattern_amount>.
      return int(re.sub(",", "", y))
    else:
      return np.nan
 
    
## Standadization
def build_dict_currency(year_start, year_end):
    '''
    <build_dict_currency> builds a dictionary collecting all of available exchange 
    rates in the https://www.xe.com/ website at each year between <year_start> 
    and <year_end>. Precisely at the date year-01-25.

    Parameters
    ----------
    year_start : int
        The firtst year from which we want to collect exchange rates.
    year_end : int
        The last year from which we want to collect exchange rates.

    Returns
    -------
    set_set : list
        Dictionary indexed by year and containing at each entry a dictionary 
        that contains  all of available exchange rates at the given year 
        between <year_start> and <year_end>.
    '''
    set_set = dict() 
    for year in np.arange(year_start, year_end+1):
        page = requests.get("https://www.xe.com/currencytables/?from=USD&"+\
                            f"date={year}-12-31")
        soup = BeautifulSoup(page.content, "html.parser")                      # accesses to the html page structure.
        """ Use of tags to found the intersting data. """
        body = soup.body
        content = body.find(id="content")
        frame_table = content.find(attrs=({"class": "historicalRateTable-wrap"}))
        table = frame_table.find(id="historicalRateTbl")
        body_table = table.find_all("tr")
        table_list = [k.find_all("td") for k in body_table]
        set_set[str(year)] = {k[0].get_text(): float(k[3].get_text()) for\
                              k in table_list[1:]}
        set_set[str(year)]["$"] = 1.0
        ''' Special processing for the currencies that have different name in 
        our data and the website.'''
        currency_list = [k[0].get_text() for k in table_list[1:]]
        currency_ue = ['ATS','BYR', 'BGL','DEM', 'EEK', 'ESP', 'FIM', 'FRF',   # Some coutries of UE have abandonned their currency for UE currency.
                      'IEP', 'ITL', 'LTL', 'LVL', 'PTE', 'ROL', 'SKK', 'YUM']
        for k in currency_ue:                                                 
            if k in currency_list:                                             # If they are present in currency list 
                pass                                                           # there is nothing to do.
            else:                                                              # Otherwise,
                set_set[str(year)][k] = set_set[str(year)]['EUR']              # we will associate them to the UE currency.
        
        corr_dict_int = {'NLG':'ANG', 'RUR':'RUB', 'TRL':'TRY', 'VEB':'VEF'}   # Some of the currencies in our data have a name different from those present in <currency_list>
        for k in corr_dict_int:                                                # besides these currencies appear only from a certain year.
            if corr_dict_int[k] in currency_list:                              # if the equivalent is present
                set_set[str(year)][k] = set_set[str(year)][corr_dict_int[k]]   # we affect the equivalent exchange rate.
            else:
                pass
        print(f'{year}: downloaded exchange rates.')
    return set_set