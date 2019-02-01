import re
import requests
from time import sleep
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date
from random import randint
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import date, timedelta
import time
import datetime

d1 = datetime.datetime.strptime((input('When do you want to start looking for flights? '
                                       '(Please type in YYYY-MM-DD format): ')),
                                '%Y-%m-%d').date()
d2 = d1 + datetime.timedelta(days=90)

delta = d2 - d1
departures = []
for i in range(delta.days + 1):
    departure_dates = d1 + timedelta(i)
    departures.append(str(departure_dates))

dates = enumerate(departures)

vaclength = int(input('How many days do you want to travel? '))
alz = vaclength # alternate zero
albo = vaclength - 1 # alternate back one
albt = vaclength - 2 # alternate back two
albth = vaclength - 3 # alternate back three
albf = vaclength - 4 # alternate back four
alfo = vaclength + 1 # alternate forward one
alft = vaclength + 2 # alternate forward two
alfth = vaclength + 3 # alternate forward three
alff = vaclength + 4 # alternate forward four
vacation_approximates = [alz, albo, albt, albth, albf, alfo, alft, alfth, alff]

depart_date_list = []
return_date_list = []
for index, date in dates:
    try:
        for i in vacation_approximates:
            depart_date = departures[index]
            depart_date_list.append(depart_date)
            return_date = departures[index + i]
            return_date_list.append(return_date)
    except IndexError:
        pass
flight_dates = zip(depart_date_list, return_date_list)

origin = 'SLC'
destination = ['KUL', 'NRT', 'FRA', 'CDG', 'ITM', 'JFK', 'PEK', 'SFO', 'HNL',
               'IAD', 'PRG', 'TXL', 'FCO', 'ICN', 'SEA', 'MRU', 'LHR', 'PPT', 'CMH']
destination_dict = {'KUL': 'Kuala Lumpur', 'NRT': 'Narita, Japan', 'FRA': 'Frankfurt',
                   'CDG': 'Paris', 'ITM': 'Kyoto', 'JFK': 'New York',
                   'PEK': 'Beijing', 'SFO': 'San Francisco', 'HNL': 'Honolulu',
                    'IAD': 'D.C.', 'PRG': 'Prague', 'TXL': 'Berlin', 'FCO': 'Rome',
                    'ICN': 'Seoul', 'SEA': 'Seattle', 'MRU': 'Mauritius',
                    'LHR': 'London', 'PPT': 'Tahiti', 'CMH': 'Columbus, Ohio'}


today = datetime.datetime.now().date()
driver = webdriver.Chrome()

# For testing
# static_depart = ['2019-03-22', '2019-03-23']
# static_return = ['2019-03-29', '2019-03-30']
# flight_dates = zip(static_depart, static_return)

for fdepart, freturn in flight_dates:
    print(f'Collecting results from {fdepart} to {freturn}.')
    for dest in destination:
        url = ('https://www.google.com/flights?hl=en#flt=' + origin + '.'
               + dest + '.' + fdepart + '*' + dest + '.'
               + origin + '.' + freturn + ';c:USD;e:1;sd:1;t:f')
        driver.get(url)
        sleep(5)
        page = driver.page_source
        soup = BeautifulSoup(page, 'html.parser')
        results_list = []
        for tmp in soup.findAll('span', {'id': re.compile(r'flt-i-.*')}):
            for jsl in tmp.findAll('jsl'):
                results_list.append(str(jsl.string))
        print(f'{len(results_list)} flights were found for {destination_dict[dest]}.')
        for i in results_list:
            annotated_flight = soup.new_tag('X')
            if 'Total price is unavailable' in i:
                results_list.pop()
            else:
                search = re.search(r'(.+?stops?)\sflight\sby\s(.+?)\.', i)
                searchtwo = re.search(r'Departure time: (.+?)\.', i)
                searchthree = re.search(r'Arrival time: (.+?)\.', i)
                searchfour = re.search(r'From (\$.+?)\s(.+?)\stotal', i)
                searchfive = re.search(r'Trip duration: (.+?)\.', i)
                searchsix = re.search(r'\.(\d{1,2}h?\s?\d{1,2}m) layover in (.+?)\.(?:(\d{1,2}h?\s?\d{1,2}m) layover in (.+?)\.)?(?:(\d{1,2}h?\s?\d{1,2}m) layover in (.+?)\.)?', i)
                if search:
                    stop_num = search.group(1)
                    annotated_flight['stops'] = stop_num
                    airline = search.group(2)
                    airline = re.sub(r'(and)', r' \1 ', airline)
                    annotated_flight['airline'] = airline
                if searchtwo:
                    departure_time = searchtwo.group(1)
                    annotated_flight['dtime'] = departure_time
                if searchthree:
                    arrival_time = searchthree.group(1)
                    annotated_flight['atime'] = arrival_time
                if searchfour:
                    price = searchfour.group(1)
                    annotated_flight['price'] = price
                    flight_type = searchfour.group(2)
                    annotated_flight['type'] = flight_type
                if searchfive:
                    total_duration = searchfive.group(1)
                    annotated_flight['length'] = total_duration
                if searchsix:
                    layoverone_time = searchsix.group(1)
                    annotated_flight['layover1time'] = layoverone_time
                    layoverone_location = searchsix.group(2)
                    annotated_flight['layover1loc'] = layoverone_location
                    layovertwo_time = searchsix.group(3)
                    annotated_flight['layover2time'] = layovertwo_time
                    layovertwo_location = searchsix.group(4)
                    annotated_flight['layover2loc'] = layovertwo_location
                    layoverthree_time = searchsix.group(5)
                    annotated_flight['layover3time'] = layoverthree_time
                    layoverthree_location = searchsix.group(6)
                    annotated_flight['layover3loc'] = layoverthree_location
                annotated_flight['ddate'] = fdepart
                annotated_flight['rdate'] = freturn
                annotated_flight['origin'] = origin
                annotated_flight['dest'] = dest
                annotated_flight['href'] = url
                annotated_flight.string = f'{origin} to {dest}'
                xan_str = str(annotated_flight)
                xan_str = re.sub(r'(layover1?2?3?(time|loc) )', '', xan_str)
                xanresults = open(str(today) + ' ' + origin + '-' + dest + '.txt', 'a', encoding='utf-8')
                masterresults = open('Master Results.txt', 'a', encoding='utf-8')
                print(xan_str, file=xanresults)
                print(xan_str, file=masterresults)
driver.close()




# port = 465  # For SSL
# smtp_server = "smtp.gmail.com"
# sender_email = "dev.xanimyle@gmail.com"  # Enter your address
# receiver_email = "vincenjes@gmail.com"  # Enter receiver address
# password = 'dnalmaerd'
#
# message = MIMEMultipart("alternative")
# message["Subject"] = "Latest Flight Results"
# message["From"] = sender_email
# message["To"] = receiver_email

#         message = 'Lowest Price Today: ' + minimum
#         message = str(message)
#         print(message)
# context = ssl.create_default_context()
# with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
#     server.login(sender_email, password)
#     server.sendmail(sender_email, receiver_email, message)
