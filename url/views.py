from django.shortcuts import render
import pandas as pd
import numpy as np
import requests
from requests import ReadTimeout
from tld import get_tld, is_tld
from urllib.parse import urlparse
import re, string
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import sklearn
import scipy.stats

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
import os.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



# general functions
class generate_lexical_features:
    def __init__(self, url):
        self.url = url

    def url_len(self):
        return len(str(self))

    def domain(self):
        try:
    #         Extract the top level domain (TLD) from the URL given
            res = get_tld(self.url, as_object=True,
                          fail_silently=False, fix_protocol=True)
            pri_domain = res.parsed_url.netloc
        except:
            pri_domain = None
        return pri_domain

    def just_tld(self):
        try:
            # Extract the top level domain (TLD) from the URL given
            res = get_tld(self.url, as_object=True,
                          fail_silently=False, fix_protocol=True)
            pri_domain = res.tld
        except:
            pri_domain = None
        return pri_domain

    def subdomain(self):
        try:
            #         Extract the top level domain (TLD) from the URL given
            res = get_tld(self.url, as_object=True,
                          fail_silently=False, fix_protocol=True)
            sub_domain = res.subdomain
        except:
            sub_domain = None
        return sub_domain
    # here we check if the len of the subdomain is zero then there is no subdomain associated to the url.

    def is_subdomain(self):
        if len(str(self.subdomain())) == 0:
            return 0
        else:
            return 1

    def httpSecure(self):
        # It supports the following URL schemes: file , ftp , gopher , hdl ,
        htp = urlparse(self.url).scheme
        # http , https ... from urllib.parse
        match = str(htp)
        if match == 'https':
            # print match.group()
            return 1
        else:
            # print 'No matching pattern found'
            return 0

    def digit_count(self):
        digits = 0
        for i in self.url:
            if i.isnumeric():
                digits = digits + 1
        return digits

    def letter_count(self):
        letters = 0
        for i in self.url:
            if i.isalpha():
                letters = letters + 1
        return letters

    def Shortining_Service(self):
    
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                        'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                        'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                        'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                        'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                        'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                        'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                        'tr\.im|link\.zip\.net',
                        self.url)
        if match:
            return 1
        else:
            return 0

    def having_ip_address(self):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4 with port
            # IPv4 in hexadecimal
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
            '([0-9]+(?:\.[0-9]+){3}:[0-9]+)|'
            '((?:(?:\d|[01]?\d\d|2[0-4]\d|25[0-5])\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d|\d)(?:\/\d{1,2})?)', self.url)  # Ipv6
        if match:
            return 1
        else:
            return 0


def lexical_features(url):

    features = generate_lexical_features(url)
 
    urldata = pd.DataFrame({'url':url,'url_len':features.url_len(),}, index=[0])
    feature = ['@', '?', '-', '=', '.', '#',
        '%', '+', '$', '!', '*', ',', '//']
    for a in feature:
        urldata[a] = urldata['url'].apply(lambda i: i.count(a))
    urldata['https'] = features.httpSecure()
    urldata['digits'] = features.digit_count()
    urldata['letters'] =features.letter_count()
    urldata['Shortining_Service'] = features.Shortining_Service()
    urldata['having_ip_address'] = features.having_ip_address() 
    urldata['tld'] =  features.just_tld()
    urldata['len_subdomain'] = len(str(features.subdomain()))
    urldata['is_subdomain'] =  features.is_subdomain()
    urldata['ratioUrlDomLen'] = features.url_len()/len(str(features.domain()))
    urldata['count_spl'] = urldata[['@', '?', '-', '=', '.',
        '#', '%', '+', '$', '!', '*', ',', '//']].sum(axis=1)
    urldata['ratioSpltinurl'] = urldata['count_spl'] / urldata['url_len']
    pipeline_lexical = pickle.load(open('C:\\Users\\user\\projects\\medusa\\url\\pipeline_model.sav', 'rb'))
   
    X_lexical = pipeline_lexical.transform(urldata.drop('url', axis=1))
    return X_lexical


def generate_performance_features(url):
    try:
        res = requests.get(
            f'https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url=http://{url}&key=AIzaSyDRnF-59dT83Z-ZlbnILiBUc8u4W-_u4go', timeout=60)
        if res.status_code == 200:
            print('ok')
            try:
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedFirstPaint']:
                    observedFirstPaint = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedFirstPaint']
                else:
                    observedFirstPaint = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['firstContentfulPaint']:
                    firstContentfulPaint = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['firstContentfulPaint']
                else:
                    firstContentfulPaint = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedFirstContentfulPaint']:
                    observedFirstContentfulPaint = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedFirstContentfulPaint']
                else:
                    observedFirstContentfulPaint = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['firstMeaningfulPaint']:
                    firstMeaningfulPaint = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['firstMeaningfulPaint']
                else:
                    firstMeaningfulPaint = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedFirstMeaningfulPaint']:
                    observedFirstMeaningfulPaint = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedFirstMeaningfulPaint']
                else:
                    observedFirstMeaningfulPaint = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['cumulativeLayoutShift']:
                    cumulativeLayoutShift = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['cumulativeLayoutShift']
                else:
                    cumulativeLayoutShift = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedCumulativeLayoutShift']:
                    observedCumulativeLayoutShift = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedCumulativeLayoutShift']
                else:
                    observedCumulativeLayoutShift = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['cumulativeLayoutShiftMainFrame']:
                    cumulativeLayoutShiftMainFrame = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['cumulativeLayoutShiftMainFrame']
                else:
                    cumulativeLayoutShiftMainFrame = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedCumulativeLayoutShiftMainFrame']:
                    observedCumulativeLayoutShiftMainFrame = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedCumulativeLayoutShiftMainFrame']
                else:
                    observedCumulativeLayoutShiftMainFrame = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['totalCumulativeLayoutShift']:
                    totalCumulativeLayoutShift = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['totalCumulativeLayoutShift']
                else:
                    totalCumulativeLayoutShift = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedTotalCumulativeLayoutShift']:
                    observedTotalCumulativeLayoutShift = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedTotalCumulativeLayoutShift']
                else:
                    observedTotalCumulativeLayoutShift = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['interactive']:
                    interactive = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['interactive']
                else:
                    interactive = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedDomContentLoaded']:
                    observedDomContentLoaded = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedDomContentLoaded']
                else:
                    observedDomContentLoaded = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['largestContentfulPaint']:
                    largestContentfulPaint = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['largestContentfulPaint']
                else:
                    largestContentfulPaint = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedLargestContentfulPaint']:
                    observedLargestContentfulPaint = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedLargestContentfulPaint']
                else:
                    observedLargestContentfulPaint = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedLoad']:
                    observedLoad = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedLoad']
                else:
                    observedLoad = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['totalBlockingTime']:
                    totalBlockingTime = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['totalBlockingTime']
                else:
                    totalBlockingTime = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['speedIndex']:
                    speedIndex = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['speedIndex']
                else:
                    speedIndex = 0
                if res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedSpeedIndex']:
                    observedSpeedIndex = res.json()["lighthouseResult"]["audits"]["metrics"]["details"]["items"][0]['observedSpeedIndex']
                else:
                    observedSpeedIndex = 0
            except KeyError:
                print('bad')
                value = [0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  0]

                return value

            scores = [observedFirstPaint, firstContentfulPaint, observedFirstContentfulPaint,  firstMeaningfulPaint, observedFirstMeaningfulPaint, cumulativeLayoutShift, observedCumulativeLayoutShift, cumulativeLayoutShiftMainFrame, observedCumulativeLayoutShiftMainFrame,
                totalCumulativeLayoutShift,  observedTotalCumulativeLayoutShift,  interactive, observedDomContentLoaded, largestContentfulPaint, observedLargestContentfulPaint,  observedLoad, totalBlockingTime, speedIndex, observedSpeedIndex]

            return scores
        if res.status_code == 500:
            print('bad')
            value = [0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            return value
    except ReadTimeout:
        print('bad')
        value = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        return value

#convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

 
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)


#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
        
# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
# Create your views here.


def index(request):
    return render(request, 'url/index.html')


def url_validator(request):
    if request.method == 'POST':
        url = request.POST.get('url')
        url = str(url)
        ###############################################
        ######     Lexical feature generation  ########
        ###############################################
       
        X_lexical= lexical_features(url)

        ###############################################
        ###### Performance feature generation  ########
        ###############################################
        lx = generate_lexical_features(url)
        performance_metric = generate_performance_features(lx.domain())
        pm_data = pd.DataFrame({'observedFirstPaint': performance_metric[0], 'firstContentfulPaint':performance_metric[1], 'observedFirstContentfulPaint':performance_metric[2],  'firstMeaningfulPaint':performance_metric[3], 'observedFirstMeaningfulPaint': performance_metric[4], 'cumulativeLayoutShift': performance_metric[5], 
                                                            'observedCumulativeLayoutShift':performance_metric[6], 'cumulativeLayoutShiftMainFrame':performance_metric[7], 'observedCumulativeLayoutShiftMainFrame':performance_metric[8],
                                                             'totalCumulativeLayoutShift':performance_metric[9],  'observedTotalCumulativeLayoutShift':performance_metric[10],  'interactive':performance_metric[11], 'observedDomContentLoaded':performance_metric[12], 'largestContentfulPaint':performance_metric[13], 
                                                             'observedLargestContentfulPaint':performance_metric[14],  'observedLoad':performance_metric[15], 'totalBlockingTime':performance_metric[16], 'speedIndex':performance_metric[17], 'observedSpeedIndex':performance_metric[18]}, index=[0])
        
        pm_pipeline = pickle.load(open('C:\\Users\\user\\projects\\medusa\\url\\pm_pipeline.sav', 'rb'))
        X_pm = pm_pipeline.transform(pm_data)
        

        ###############################################
        ###### Bag of words feature generation  ########
        ###############################################

        pm_data['tfidf_features'] = finalpreprocess(str(url))
        
        tfidf_vectorizer = pickle.load(open('C:\\Users\\user\\projects\\medusa\\url\\tfidf_vectorizer_model.sav', 'rb'))
        X_nlp = tfidf_vectorizer.transform(pm_data['tfidf_features'])

        ############################################
        ######       model summation     ###########
        ############################################

        xgb_lexical = pickle.load(open('C:\\Users\\user\\projects\\medusa\\url\\xgB_lexical_model.sav', 'rb'))
        rf_pm = pickle.load(open('C:\\Users\\user\\projects\\medusa\\url\\rf_pm_model.sav', 'rb'))
        lr_tfidf = pickle.load(open('C:\\Users\\user\\projects\\medusa\\url\\lr_tfidf_model.sav', 'rb'))

        ##############################################
        ######## Evaluating URL AND Prediction #######
        ##############################################

        y_pred_lexical = xgb_lexical.predict(X_lexical)
        y_pred_pm = rf_pm.predict(X_pm)
        y_pred_nlp = lr_tfidf.predict(X_nlp)
        y_predict_arr = [np.array(y_pred_lexical), np.array(y_pred_pm), np.array(y_pred_nlp)]
        y_predict = scipy.stats.mode(np.stack(y_predict_arr), axis=0)
        print(y_predict.mode.tolist()[0][0])
    context = {'pm0' : performance_metric[0],'pm1' : performance_metric[1],
                'pm2' : performance_metric[2],'pm3' : performance_metric[3],
                'pm4' : performance_metric[4],'pm5' : performance_metric[5],
                'pm6' : performance_metric[6],'pm7' : performance_metric[7],
                'pm8' : performance_metric[8],'pm9' : performance_metric[9],
                'pm10' : performance_metric[10],'pm11' : performance_metric[11],
                'pm12' : performance_metric[12],'pm13' : performance_metric[13],
                'pm14' : performance_metric[14],'pm15' : performance_metric[15],
                'pm16' : performance_metric[16],'pm17' : performance_metric[17],
                'pm18' : performance_metric[18], 'https' : lx.httpSecure(), 
                'digits':lx.digit_count(), 'letters':lx.letter_count, 
                'url_len':lx.url_len(), 'len_subdomain':len(lx.subdomain()),
                'is_subdomain':lx.is_subdomain(), 'Shortining_Service': lx.Shortining_Service(), 
                'having_ip_address':lx.having_ip_address(), 'tld':lx.just_tld(),
                'prediction': y_predict.mode.tolist()[0][0], 
                'url': url, 
                }
    return render(request, 'url/index.html', context)