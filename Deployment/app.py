from flask import Flask, request
import pandas as pd
import numpy as np
import gc
import pickle
import sqlite3
import flask
#pipeline contains the entire pipeline for prediction of query point
from pipeline import final_pipeline

app = Flask(__name__)

#instantiating the pipeline
file_directory = 'Final Pipeline Files/'
test_predictor_class = final_pipeline(file_directory = file_directory)

with open(file_directory + 'application_table_dtypes.pkl', 'rb') as f:
	application_dtypes = pickle.load(f)
	
#home page
@app.route('/', methods = ['GET'])
def home_page():
    return flask.render_template('home-page.html')

#prediction page
@app.route('/home', methods = ['POST', 'GET'])
def inputs_page():
	return flask.render_template('predict.html')

#results page
@app.route('/predict', methods = ['POST'])
def prediction():
	conn = sqlite3.connect(file_directory + 'HOME_CREDIT_DB.db')
    #getting the SK_ID_CURR from user
	sk_id_curr = request.form.to_dict()['SK_ID_CURR']
	sk_id_curr = int(sk_id_curr)
	test_datapoint = pd.read_sql_query(f'SELECT * FROM applications WHERE SK_ID_CURR == {sk_id_curr}', conn)
	test_datapoint = test_datapoint.replace([None], np.nan)
	test_datapoint = test_datapoint.astype(application_dtypes)
	predicted_proba, predicted_class, data_for_display = test_predictor_class.predict(test_datapoint)

	if predicted_class == 1:
		prediction = 'a Potential Defaulter'
	else:
		prediction = 'not a Defaulter'
		predicted_proba = 1 - predicted_proba

	data_for_display = pd.concat([test_datapoint[['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']], data_for_display.reset_index(drop = True)], axis = 1)
	data_for_display = data_for_display.to_html(classes = 'data', header = 'true', index = False)

	conn.close()
	gc.collect()

	return flask.render_template('result_and_inference.html', tables = [data_for_display],
			output_proba = predicted_proba, output_class = prediction, sk_id_curr = sk_id_curr)

if __name__ == '__main__':
	app.run(host = '0.0.0.0', port = 5000)
