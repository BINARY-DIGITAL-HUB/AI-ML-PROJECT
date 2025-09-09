from flask import Flask
from flask import render_template
import json
from flask import request
import joblib



# FLASK RUN
# =====================================================================
app = Flask(__name__)


if __name__ == '__main__': 
    app.run()


# HELPER METHOD TO PREPROCESS AND PREDICT CAR PRICE
# =======================================================================

# function to load json data... 
def load_feature_json(path):
    features = None
    with open(path, 'r') as file: 
        features = json.load(file)

    return features



# function to convert feature
def feature_converter(ld_feature, company, body_cond, mecha_cond, no_cylind, trans_type, model, year, horse_p): 
    
    ld_feature = load_feature_json('feature.json')
    company_feature = ld_feature['company'][company]
    body_condi = ld_feature['body_condition'][body_cond]
    mecha_condi = ld_feature['mechanical_condition'][mecha_cond]
    trans_types =  ld_feature['transmission_type'][trans_type]
    models =  ld_feature['model'][model]
    horse_po = ld_feature['horsepower'][horse_p]
   
    # non diction feature... 
    years = year
    no_cylinds = no_cylind

    return [company_feature, body_condi, int(mecha_condi), int(no_cylinds), trans_types, models, float(years), horse_po]


# function to convert feature
def load_features_data(): 
    
    ld_feature = load_feature_json('feature.json')
    company_feature = ld_feature['company']
    body_condi = ld_feature['body_condition']
    mecha_condi = ld_feature['mechanical_condition']
    trans_types =  ld_feature['transmission_type']
    models =  ld_feature['model']
    horse_po = ld_feature['horsepower']
   
    # non diction feature... 
    years = [2021., 2015., 2014., 2020., 2016., 2018., 2017., 2003., 2011.,
       2012., 2010., 2019., 1989., 2013., 2009., 2007., 2006., 2004.,
       2008., 1998., 1980., 1983., 2005., 2001., 1991., 2002., 1990.,
       1973., 1999., 2000., 1971., 1979., 1975., 1969., 1997., 1984.,
       1992., 1972., 1965., 1977., 1996., 1995., 1953., 1982., 1987.,
       1954., 1970.]
    
    no_cylinds = ['4', '8', '6', '12', '5', '3', '10']
    no_cylinds = [int(cy) for cy in no_cylinds]


    return [company_feature, body_condi, mecha_condi, no_cylinds, trans_types, models, years, horse_po]

# load machine learning model
def load_machine_model(): 

    rf_model = joblib.load('Radomforest.joblib')
    gb_model = joblib.load('GradientBoost.joblib')
    knn_model = joblib.load('KNeighbors.joblib')
    ensemble = joblib.load('ensemble.joblib')

    return (rf_model, gb_model, knn_model, ensemble)


# function to predict
def predict(model, features): 

    prediction = model.predict([features])[0]

    return prediction






# HTML HANDLERS (FRONT END DESIGNS)
# ======================================================================
@app.route('/', methods=['POST', 'GET'] )
def index(): 

    # init val
    collective_prediction = [0, 0, 0, 0]

    features_data = load_features_data()
    com, body, mecho, cylin, transmi, model, yrs, hrs = features_data[0],features_data[1],features_data[2], features_data[3] ,features_data[4],features_data[5],features_data[6], features_data[7] 


    if request.method == "POST": 
        form = request.form

        # getting for data 
        company = form['com']
        body_condition = form['body']
        mechanical_condidtion = form['mecho']
        cylinder_number = form['cylin']
        transmition_type = form['transmi']
        models = form['model']
        years = form['yrs']
        hoursepower= form['hrs']

        # deburgin .. printing to see the output.. .
        print('{}\n{}\n{}\n{}\n{}'.format(company, mechanical_condidtion, transmition_type, years, hoursepower))


        # ld_feature, company, body_cond, mecha_cond, no_cylind, trans_type, model, year, horse_p
        # convering data to feature
        numeric_features = feature_converter(features_data, company, body_condition, mechanical_condidtion, cylinder_number, transmition_type, models, years, hoursepower)
        print("Numeric Feature sample : {}".format(numeric_features))
        print([type(v) for v in numeric_features])

        # load model using helper method
        randam_forest, gradient_boost, knn, ensemble = load_machine_model()


        # prediction using predict helper function
        rf_prediction = predict(randam_forest, numeric_features)
        gb_predictions = predict(gradient_boost, numeric_features)
        knn_predictions = predict(knn, numeric_features)
        ensemble_predictions = predict(ensemble, numeric_features)

        collective_prediction = [rf_prediction, gb_predictions, knn_predictions, ensemble_predictions]

        


    return render_template('index.html', com=com, body=body, mecho=mecho, 
                           cylin=cylin, transmi=transmi, model=model, yrs=yrs, hrs=hrs, 
                           collective_prediction=collective_prediction)









